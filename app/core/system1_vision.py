import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from ultralytics import YOLO
from datetime import datetime
import uuid

from app.config import settings
from app.utils.logger import app_logger
from app.core.dataloader import InspectionDataLoader
from app.core.cost_estimator import CostEstimator, CostBreakdown
from app.api.v1.schema import DamageRecord, SeverityLabel, DamageCategory, RecommendedAction, RepairPriority

class System1Vision:
    COCO_TO_DAMAGE = {
        2: ("car", "dent"), 
        5: ("bus", "structural"),
        7: ("truck", "structural"),
        8: ("boat", "corrosion"),
    }
    
    PANEL_KEYWORDS = {
        "front_bumper": ["bumper", "front", "grille"],
        "side_mirror": ["mirror", "side"],
        "left_door": ["door", "left", "driver"],
        "roof": ["roof", "top"],
        "hood": ["hood", "bonnet", "engine"],
        "trunk": ["trunk", "boot", "rear"],
        "windshield": ["windshield", "glass", "window"],
        "wheel": ["wheel", "tire", "rim"],
    }

    def __init__(self):
        app_logger.info("Initializing System 1 Vision Engine...")
        self.model = YOLO(settings.YOLO_MODEL_PATH)
        self.data_loader = InspectionDataLoader(settings.REPAIR_COST_PATH)
        self.cost_estimator = CostEstimator()
        self.seg_model = None
        if settings.YOLO_SEG_MODEL_PATH and Path(settings.YOLO_SEG_MODEL_PATH).exists():
            try:
                self.seg_model = YOLO(settings.YOLO_SEG_MODEL_PATH)
                app_logger.info(f"Segmentation model loaded: {settings.YOLO_SEG_MODEL_PATH}")
            except Exception as e:
                app_logger.warning(f"Segmentation model load failed (bbox fallback): {e}")
                self.seg_model = None
        app_logger.info("System 1 ready")

    def run_inference(self, image_bytes: bytes, metadata: dict) -> Tuple[np.ndarray, List[DamageRecord], Dict, Dict]:
        """
        Returns: (heatmap_np, damage_records, summary_dict, routing_decision)
        Routing decision includes confidence calibration for metacognitive layer.
        """
        try:
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Invalid image format")
            h, w = img.shape[:2]

            results = self.model(img, conf=settings.CONFIDENCE_THRESH, verbose=False)[0]
            seg_masks = self._segment_masks(img) if self.seg_model else None
            damage_records = self._parse_to_schema(results, metadata, img.shape, seg_masks)
            heatmap = self._generate_heatmap(img, damage_records)
            summary = self._compute_summary(damage_records)
            
            # Metacognitive routing: aggregate confidence across detections
            avg_conf = np.mean([r.ai_confidence for r in damage_records]) if damage_records else 0.0
            routing = {
                "status": "CONFIDENT" if avg_conf >= 0.75 else "UNCERTAIN",
                "confidence": round(avg_conf, 3),
                "action": "PROCEED" if avg_conf >= 0.75 else "FLAG_HUMAN",
                "requires_human_review": avg_conf < 0.75
            }
            
            return heatmap, damage_records, summary, routing
            
        except Exception as e:
            app_logger.error(f"System 1 inference failed: {e}")
            # Return deterministic fallback
            fallback_route = {"status": "UNCERTAIN", "confidence": 0.0, "action": "ERROR", "requires_human_review": True}
            return np.zeros((480, 640, 3), dtype=np.uint8), [], {"total_cost_median": 0, "detection_count": 0}, fallback_route

    def _parse_to_schema(self, results, metadata: dict, img_shape: tuple, seg_masks=None) -> List[DamageRecord]:
        records = []
        boxes = results.boxes
        if boxes is None:
            return records

        mask_list = None
        if seg_masks is not None:
            try:
                mask_list = seg_masks.cpu().numpy() if hasattr(seg_masks, "cpu") else np.asarray(seg_masks)
            except Exception:
                mask_list = None

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            
            panel, category = self._infer_panel_category(cls_id, (x1,y1,x2,y2), img_shape)
            severity = self._estimate_severity(conf, category)
            
            area_cm2 = self._calc_area_cm2((x1,y1,x2,y2), img_shape)
            if mask_list is not None and i < len(mask_list):
                try:
                    mask_area_px = float(mask_list[i].sum())
                    if mask_area_px > 0:
                        px_to_cm = 450 / max(img_shape[1], 1)
                        mask_area_cm2 = mask_area_px * (px_to_cm ** 2)
                        bbox_area_px = max(0, x2-x1) * max(0, y2-y1)
                        coverage = mask_area_px / max(bbox_area_px, 1)
                        # Use mask-derived area when it meaningfully overlaps the box
                        if coverage > 0.15:
                            area_cm2 = mask_area_cm2
                            # Larger mask coverage within panel → higher severity
                            if coverage > 0.6 and severity == SeverityLabel.MINOR:
                                severity = SeverityLabel.MODERATE
                            elif coverage > 0.8 and severity in [SeverityLabel.MINOR, SeverityLabel.MODERATE]:
                                severity = SeverityLabel.MAJOR
                except Exception:
                    pass
            
            # Use production CostEstimator
            cost: CostBreakdown = self.cost_estimator.estimate(
                panel=panel,
                severity=severity.value,
                area_cm2=area_cm2,
                city_tier=metadata.get("city_tier", "tier_2"),
                vehicle_age=metadata.get("vehicle_age", 2024 - metadata.get("manufacture_year", 2018)),
                odometer_km=metadata.get("odometer_km", 50000),
                brand=metadata.get("vehicle_brand", "generic")
            )
            
            labor = cost.labor_inr / self.cost_estimator.LABOR_RATES.get(metadata.get("city_tier", "tier_2"), 680)
            rust_prob = 0.3 if category == DamageCategory.CORROSION else 0.05
            structural_score = 0.95 if category not in [DamageCategory.STRUCTURAL] else 0.6
            
            record = DamageRecord(
                inspection_id=f"INS_{uuid.uuid4().hex[:8].upper()}",
                panel_affected=panel,
                damage_category=category,
                damage_subtype=metadata.get("damage_subtype"),
                severity_label=severity,
                severity_score=self._score_severity(severity, conf),
                damage_area_cm2=round(self._calc_area_cm2((x1,y1,x2,y2), img_shape), 1),
                repair_method=self._infer_repair_method(panel, category, severity),
                repair_cost_min_inr=cost.cost_range_inr[0],
                repair_cost_max_inr=cost.cost_range_inr[1],
                labor_hours=round(labor, 1),
                paint_required=cost.paint_inr > 0,
                parts_replacement=severity == SeverityLabel.MAJOR,
                safety_risk=self._assess_safety(category, severity),
                drivable_status=severity != SeverityLabel.CRITICAL,
                rust_probability=rust_prob,
                structural_integrity_score=structural_score,
                battery_health_impact=0.0,
                ai_confidence=round(conf, 3),
                inspection_confidence=round(conf * 0.95, 3),
                recommended_action=self._recommend_action(severity, cost.total_estimated_inr),
                predicted_resale_impact_inr=-cost.total_estimated_inr * 1.2,
                expected_profit_after_repair_inr=cost.total_estimated_inr * 0.3,
                repair_priority=self._priority(severity, category),
                agent_consensus_score=round(conf * 0.92, 3)
            )
            # Attach bbox for heatmap rendering
            record.bbox = [round(c, 1) for c in [x1, y1, x2, y2]]
            records.append(record)
        
        return records

    def _infer_panel_category(self, cls_id: int, bbox: tuple, img_shape: tuple) -> Tuple[str, DamageCategory]:
        x1, y1, x2, y2 = bbox
        h, w = img_shape
        center_x, center_y = (x1+x2)/2, (y1+y2)/2
        
        if center_y < h * 0.3:
            panel = "hood" if center_x < w/2 else "roof"
        elif center_y > h * 0.7:
            panel = "front_bumper" if center_x < w/2 else "trunk"
        elif center_x < w * 0.3:
            panel = "left_door"
        elif center_x > w * 0.7:
            panel = "right_door"
        else:
            panel = "side_mirror"

        category = DamageCategory.DENT if cls_id in [2, 5, 7] else DamageCategory.SCRATCH
        return panel, category

    def _estimate_severity(self, conf: float, category: DamageCategory) -> SeverityLabel:
        if conf > 0.85:
            return SeverityLabel.MAJOR if category in [DamageCategory.STRUCTURAL, DamageCategory.CORROSION] else SeverityLabel.MODERATE
        elif conf > 0.6:
            return SeverityLabel.MODERATE
        return SeverityLabel.MINOR

    def _score_severity(self, label: SeverityLabel, conf: float) -> float:
        base = {"MINOR": 2.5, "MODERATE": 5.0, "MAJOR": 7.5, "CRITICAL": 9.0}[label.value]
        return round(min(10.0, base + (conf - 0.5) * 2), 1)

    def _calc_area_cm2(self, bbox: tuple, img_shape: tuple) -> float:
        x1, y1, x2, y2 = bbox
        h, w = img_shape
        px_to_cm = 450 / max(w, 1)
        area_px = max(0, x2-x1) * max(0, y2-y1)
        return area_px * (px_to_cm ** 2)

    def _segment_masks(self, img: np.ndarray):
        """Run instance segmentation; returns mask data tensor aligned with detections."""
        try:
            seg_results = self.seg_model(img, verbose=False)[0]
            if seg_results.masks is None:
                return None
            return seg_results.masks.data
        except Exception as e:
            app_logger.warning(f"Segmentation inference failed: {e}")
            return None

    def _infer_repair_method(self, panel: str, category: DamageCategory, severity: SeverityLabel) -> str:
        if category == DamageCategory.CORROSION:
            return "anti_rust_treatment" if severity != SeverityLabel.MAJOR else "panel_replacement"
        if category == DamageCategory.GLASS:
            return "windshield_repair" if severity == SeverityLabel.MINOR else "windshield_replacement"
        if severity == SeverityLabel.MINOR:
            return "touchup_paint"
        if severity == SeverityLabel.MODERATE:
            return "panel_repair"
        return "panel_replacement"

    def _assess_safety(self, category: DamageCategory, severity: SeverityLabel) -> str:
        if category == DamageCategory.STRUCTURAL or severity == SeverityLabel.CRITICAL:
            return "critical"
        if category in [DamageCategory.ELECTRICAL, DamageCategory.GLASS] and severity != SeverityLabel.MINOR:
            return "high"
        return "low"

    def _recommend_action(self, severity: SeverityLabel, est_cost: float) -> RecommendedAction:
        if severity == SeverityLabel.CRITICAL or est_cost > 50000:
            return RecommendedAction.ESCALATE_TO_HUMAN
        if severity == SeverityLabel.MINOR and est_cost < 5000:
            return RecommendedAction.REFURBISH
        if est_cost > 20000:
            return RecommendedAction.SELL_AS_IS
        return RecommendedAction.REFURBISH

    def _priority(self, severity: SeverityLabel, category: DamageCategory) -> RepairPriority:
        if severity in [SeverityLabel.MAJOR, SeverityLabel.CRITICAL]:
            return RepairPriority.HIGH
        if category in [DamageCategory.STRUCTURAL, DamageCategory.ELECTRICAL]:
            return RepairPriority.MEDIUM
        return RepairPriority.LOW

    def _compute_summary(self, records: List[DamageRecord]) -> dict:
        if not records:
            return {"total_cost_median": 0, "detection_count": 0, "priority_breakdown": {}, "avg_confidence": 0.0}
        costs = [(r.repair_cost_min_inr + r.repair_cost_max_inr)/2 for r in records]
        return {
            "total_cost_median": round(sum(costs), 2),
            "detection_count": len(records),
            "priority_breakdown": {
                p.value: sum(1 for r in records if r.repair_priority == p)
                for p in RepairPriority
            },
            "avg_confidence": round(sum(r.ai_confidence for r in records)/len(records), 3)
        }

    def _generate_heatmap(self, img: np.ndarray, records: List[DamageRecord]) -> np.ndarray:
        overlay = img.copy()
        color_map = {
            "MINOR": (0, 255, 0),
            "MODERATE": (0, 165, 255),
            "MAJOR": (0, 0, 255),
            "CRITICAL": (0, 0, 139)
        }
        for r in records:
            if not hasattr(r, 'bbox') or not r.bbox:
                continue
            x1, y1, x2, y2 = map(int, r.bbox)
            color = color_map.get(r.severity_label.value, (255, 255, 255))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
            label = f"{r.panel_affected}: ₹{r.repair_cost_min_inr:.0f}-{r.repair_cost_max_inr:.0f}"
            cv2.putText(overlay, label, (max(x1, 5), max(y1-5, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return cv2.addWeighted(overlay, settings.HEATMAP_OPACITY, img, 1 - settings.HEATMAP_OPACITY, 0)