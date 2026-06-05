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
        self.cost_df = self._load_cost_lookup()
        app_logger.info("System 1 ready")

    def _load_cost_lookup(self) -> pd.DataFrame:
        """Load or generate repair cost lookup table"""
        return pd.DataFrame({
            "panel": ["side_mirror", "roof", "front_bumper", "left_door", "windshield"],
            "category": ["scratch", "corrosion", "dent", "scratch", "glass"],
            "severity": ["MINOR", "MODERATE", "MAJOR", "MINOR", "CRITICAL"],
            "cost_min": [2000, 8000, 25000, 3000, 15000],
            "cost_max": [8000, 25000, 60000, 12000, 45000],
            "labor_hrs": [1.5, 6.0, 12.0, 3.0, 4.0]
        })

    def run_inference(self, image_bytes: bytes, metadata: dict) -> Tuple[np.ndarray, List[DamageRecord], Dict]:
        try:
            # 1. Preprocess image
            nparr = np.frombuffer(image_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Invalid image format")
            h, w = img.shape[:2]

            # 2. Run YOLO inference
            results = self.model(img, conf=settings.CONFIDENCE_THRESH, verbose=False)[0]
            
            # 3. Parse detections into your schema
            damage_records = self._parse_to_schema(results, metadata, img.shape)
            
            # 4. Generate annotated heatmap
            heatmap = self._generate_heatmap(img, damage_records)
            
            # 5. Compute summary metrics
            summary = self._compute_summary(damage_records)
            
            return heatmap, damage_records, summary
            
        except Exception as e:
            app_logger.error(f"System 1 inference failed: {e}")
            raise

    def _parse_to_schema(self, results, metadata: dict, img_shape: tuple) -> List[DamageRecord]:
        """Convert YOLO output to your rich DamageRecord schema"""
        records = []
        boxes = results.boxes
        if boxes is None:
            return records

        for i in range(len(boxes)):
            cls_id = int(boxes.cls[i])
            conf = float(boxes.conf[i])
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            
            # Map COCO class → damage category (placeholder logic)
            panel, category = self._infer_panel_category(cls_id, (x1,y1,x2,y2), img_shape)
            severity = self._estimate_severity(conf, category)
            
            # Lookup costs from historical data
            cost_range = self.data_loader.get_repair_cost_range(panel, category, severity.value)
            labor = self._estimate_labor(panel, category, severity)
            
            # Compute derived fields
            area_cm2 = self._calc_area_cm2((x1,y1,x2,y2), img_shape)
            rust_prob = 0.3 if category == "corrosion" else 0.05
            structural_score = 0.95 if category not in ["structural"] else 0.6
            
            record = DamageRecord(
                inspection_id=f"INS_{uuid.uuid4().hex[:8].upper()}",
                panel_affected=panel,
                damage_category=category,
                damage_subtype=metadata.get("damage_subtype"),
                severity_label=severity,
                severity_score=self._score_severity(severity, conf),
                damage_area_cm2=round(area_cm2, 1),
                repair_method=self.data_loader.get_panel_repair_methods(panel)[0],
                repair_cost_min_inr=cost_range["min"],
                repair_cost_max_inr=cost_range["max"],
                labor_hours=labor,
                paint_required=category in ["scratch", "corrosion"],
                parts_replacement=severity == SeverityLabel.MAJOR,
                safety_risk=self._assess_safety(category, severity),
                drivable_status=severity != SeverityLabel.CRITICAL,
                rust_probability=rust_prob,
                structural_integrity_score=structural_score,
                battery_health_impact=0.0,  # Extend for EV logic
                ai_confidence=round(conf, 3),
                inspection_confidence=round(conf * 0.95, 3),  # Simulated calibration
                recommended_action=self._recommend_action(severity, cost_range["median"]),
                predicted_resale_impact_inr=-cost_range["median"] * 1.2,
                expected_profit_after_repair_inr=cost_range["median"] * 0.3,
                repair_priority=self._priority(severity, category),
                agent_consensus_score=round(conf * 0.92, 3)
            )
            records.append(record)
        
        return records

    def _infer_panel_category(self, cls_id: int, bbox: tuple, img_shape: tuple) -> Tuple[str, DamageCategory]:
        """Heuristic panel+category inference (replace with fine-tuned model later)"""
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
            return SeverityLabel.MAJOR if category in ["structural", "corrosion"] else SeverityLabel.MODERATE
        elif conf > 0.6:
            return SeverityLabel.MODERATE
        return SeverityLabel.MINOR

    def _score_severity(self, label: SeverityLabel, conf: float) -> float:
        base = {"MINOR": 2.5, "MODERATE": 5.0, "MAJOR": 7.5, "CRITICAL": 9.0}[label.value]
        return round(min(10.0, base + (conf - 0.5) * 2), 1)

    def _calc_area_cm2(self, bbox: tuple, img_shape: tuple) -> float:
        x1, y1, x2, y2 = bbox
        h, w = img_shape
        px_to_cm = 450 / w
        area_px = (x2-x1) * (y2-y1)
        return area_px * (px_to_cm ** 2)

    def _estimate_labor(self, panel: str, category: DamageCategory, severity: SeverityLabel) -> float:
        base = {"MINOR": 2, "MODERATE": 5, "MAJOR": 10, "CRITICAL": 20}[severity.value]
        panel_mult = {"windshield": 1.5, "roof": 1.3, "side_mirror": 0.8}.get(panel, 1.0)
        return round(base * panel_mult, 1)

    def _assess_safety(self, category: DamageCategory, severity: SeverityLabel) -> str:
        if category == "structural" or severity == SeverityLabel.CRITICAL:
            return "critical"
        if category in ["electrical", "glass"] and severity != SeverityLabel.MINOR:
            return "high"
        return "low"

    def _recommend_action(self, severity: SeverityLabel, est_cost: float) -> RecommendedAction:
        if severity == SeverityLabel.CRITICAL or est_cost > 50000:
            return RecommendedAction.ESCALATE_TO_HUMAN
        if severity == SeverityLabel.MINOR and est_cost < 5000:
            return RecommendedAction.REFURBISH
        return RecommendedAction.REFURBISH

    def _priority(self, severity: SeverityLabel, category: DamageCategory) -> RepairPriority:
        if severity in [SeverityLabel.MAJOR, SeverityLabel.CRITICAL]:
            return RepairPriority.HIGH
        if category in ["structural", "electrical"]:
            return RepairPriority.MEDIUM
        return RepairPriority.LOW

    def _compute_summary(self, records: List[DamageRecord]) -> dict:
        if not records:
            return {"total_cost_median": 0, "detection_count": 0, "priority_breakdown": {}}
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
        for r in records:
            x1, y1, x2, y2 = map(int, [100, 100, 300, 200]) 
            color_map = {
                "MINOR": (0, 255, 0),
                "MODERATE": (0, 165, 255),
                "MAJOR": (0, 0, 255),
                "CRITICAL": (0, 0, 139)
            }
            color = color_map.get(r.severity_label.value, (255, 255, 255))
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 3)
            label = f"{r.panel_affected}: ₹{r.repair_cost_min_inr:.0f}-{r.repair_cost_max_inr:.0f}"
            cv2.putText(overlay, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return cv2.addWeighted(overlay, settings.HEATMAP_OPACITY, img, 1 - settings.HEATMAP_OPACITY, 0)