import os
import json
import ollama
from mem0 import Memory
from typing import List, Dict
from app.config import settings
from app.utils.logger import app_logger
from app.api.v1.schema import AgentRecommendation

class System2Agent:
    def __init__(self):
        self.model = settings.SYSTEM2_MODEL_NAME
        # mem0 defaults to local SQLite if no API key provided
        self.mem0 = Memory()
        app_logger.info(f"✅ System 2 Agent initialized: {self.model} + mem0 (local)")

    def _load_context(self, vehicle_id: str, metadata: dict) -> str:
        try:
            memories = self.mem0.search(
                query=f"Past inspection decisions for {vehicle_id}",
                user_id=vehicle_id,
                limit=2
            )
            context = f"Vehicle: {metadata['vehicle_brand']} {metadata['vehicle_model']} ({metadata['manufacture_year']})"
            if memories:
                context += "\nHistorical context:\n" + "\n".join([f"- {m['memory']}" for m in memories])
            return context
        except Exception as e:
            app_logger.warning(f"⚠️ mem0 context load failed: {e}")
            return f"Vehicle: {metadata['vehicle_brand']} {metadata['vehicle_model']}"

    def analyze_tradeoffs(self, damages: List[dict], metadata: dict, inventory_target_days: int = 45) -> AgentRecommendation:
        context = self._load_context(metadata.get("inspection_id", "UNKNOWN"), metadata)
        
        damage_lines = []
        for d in damages:
            damage_lines.append(
                f"- Panel: {d['panel_affected']}, Severity: {d['severity_label']}, "
                f"Cost: ₹{d['repair_cost_min_inr']}-{d['repair_cost_max_inr']}, "
                f"AI Confidence: {d['ai_confidence']:.2f}"
            )

        prompt = f"""You are a Refurbishment Strategy Engine for a used-car marketplace.
Analyze inspection data and output ONLY valid JSON matching this exact schema:
{{
  "recommendation": "REPAIR | AS_IS | PARTIAL",
  "reasoning": "<1 sentence business justification>",
  "expected_net_margin_pct": <float>,
  "turnover_risk": "LOW | MED | HIGH",
  "repair_priority_items": ["<panel1>", "<panel2>"]
}}

VEHICLE CONTEXT:
{context}

DAMAGE REPORT:
{chr(10).join(damage_lines)}

BUSINESS CONSTRAINTS:
- Inventory turnover target: {inventory_target_days} days
- EV flag: {metadata.get('is_ev', False)}
- City tier: {metadata.get('city_tier', 'tier_2')}
- Market margin threshold: 8%
"""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                format="json",
                options={"temperature": 0.3, "num_predict": 256}
            )
            result = json.loads(response["message"]["content"])
            
            rec = result.get("recommendation", "PARTIAL")
            if rec not in ["REPAIR", "AS_IS", "PARTIAL"]:
                rec = "PARTIAL"

            return AgentRecommendation(
                recommendation=rec,
                reasoning=result.get("reasoning", "Balanced margin vs turnover risk"),
                expected_net_margin_pct=round(float(result.get("expected_net_margin_pct", 0)), 1),
                turnover_risk=result.get("turnover_risk", "MED"),
                repair_priority_items=result.get("repair_priority_items", [])
            )
        except Exception as e:
            app_logger.error(f"❌ System 2 LLM failed: {e}")
            # Deterministic fallback
            total_cost = sum(d["repair_cost_max_inr"] for d in damages)
            return AgentRecommendation(
                recommendation="AS_IS" if total_cost > 50000 else "REPAIR",
                reasoning="Fallback: High/Low cost threshold applied",
                expected_net_margin_pct=-5.0 if total_cost > 50000 else 12.0,
                turnover_risk="HIGH" if total_cost > 50000 else "LOW",
                repair_priority_items=[d["panel_affected"] for d in damages[:2]]
            )

    def log_decision(self, case_id: str, rec: AgentRecommendation):
        try:
            self.mem0.add(
                f"Case {case_id}: {rec.recommendation} | Margin: {rec.expected_net_margin_pct}% | Risk: {rec.turnover_risk}",
                user_id=case_id
            )
        except Exception as e:
            app_logger.warning(f"⚠️ mem0 logging failed: {e}")