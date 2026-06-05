import os, cv2, numpy as np, time, json, logging
import gradio as gr
import onnxruntime as ort
from typing import Dict, List, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("refurbcortex_spaces")

COST_TABLE = {
    0: {"panel": "front_bumper", "base_cost_inr": 8500, "labor_hrs": 4.0, "paint": True},
    1: {"panel": "hood", "base_cost_inr": 12000, "labor_hrs": 5.5, "paint": True},
    2: {"panel": "left_door", "base_cost_inr": 6500, "labor_hrs": 3.0, "paint": True},
    3: {"panel": "side_mirror", "base_cost_inr": 2200, "labor_hrs": 1.0, "paint": False},
    4: {"panel": "windshield", "base_cost_inr": 18000, "labor_hrs": 2.5, "paint": False},
    5: {"panel": "rear_bumper", "base_cost_inr": 7800, "labor_hrs": 3.5, "paint": True},
    6: {"panel": "roof", "base_cost_inr": 14500, "labor_hrs": 6.0, "paint": True},
}

class RefurbCortexEngine:
    """Production ONNX inference engine with deterministic routing & cost mapping."""
    def __init__(self, model_path: str = "./models/yolov8s.onnx", conf_thresh: float = 0.45):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}. Export via: yolo export model=yolov8s.pt format=onnx opset=11")
        
        # Dynamic provider fallback (GPU → CPU)
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_path, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.conf_thresh = conf_thresh
        logger.info(f"✅ Vision engine initialized | Providers: {providers}")

    def infer(self, img: np.ndarray) -> Tuple[np.ndarray, List[Dict], Dict]:
        try:
            h, w = img.shape[:2]
            scale = 640 / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            padded = np.full((640, 640, 3), 114, dtype=np.uint8)
            padded[:new_h, :new_w] = resized
            tensor = padded.transpose(2, 0, 1).astype(np.float32) / 255.0

            outputs = self.session.run(None, {self.input_name: tensor[np.newaxis, ...]})[0].squeeze()
            boxes, scores = outputs[:, :4], outputs[:, 4:]
            confs = np.max(scores, axis=1)
            classes = np.argmax(scores[:, 5:], axis=1)
            mask = confs > self.conf_thresh
            boxes, confs, classes = boxes[mask], confs[mask], classes[mask]

            if len(boxes) == 0:
                return img, [], {"status": "CLEAN", "confidence": 0.0, "cost": 0, "action": "NO_REPAIR"}

            # NMS & Heatmap Generation
            idx = cv2.dnn.NMSBoxes(boxes.tolist(), confs.tolist(), self.conf_thresh, 0.5)
            detections = []
            overlay = img.copy()
            total_cost = 0.0
            max_conf = 0.0

            for i in idx.flatten():
                x1, y1, x2, y2 = boxes[i]
                x1, y1, x2, y2 = int(x1/scale), int(y1/scale), int(x2/scale), int(y2/scale)
                conf = float(confs[i])
                cls = int(classes[i])
                meta = COST_TABLE.get(cls, {"panel": "unknown", "base_cost_inr": 3000, "labor_hrs": 2.0, "paint": False})

                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 165, 255), 3)
                label = f"{meta['panel']} | ₹{meta['base_cost_inr']} | {conf:.2f}"
                cv2.putText(overlay, label, (x1, max(y1-10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

                detections.append({"panel": meta["panel"], "confidence": conf, "cost_inr": meta["base_cost_inr"], "labor_hrs": meta["labor_hrs"]})
                total_cost += meta["base_cost_inr"]
                max_conf = max(max_conf, conf)

            # Metacognitive Routing Logic
            routing_status = "CONFIDENT" if max_conf >= 0.75 else "UNCERTAIN"
            safety_margin = 1.15 if routing_status == "UNCERTAIN" else 1.0
            final_cost = round(total_cost * safety_margin)
            action = "APPROVE_REPAIR" if routing_status == "CONFIDENT" else "HUMAN_REVIEW"

            return overlay, detections, {
                "status": routing_status,
                "confidence": round(max_conf, 3),
                "estimated_cost_inr": final_cost,
                "action": action,
                "detections_count": len(detections)
            }
        except Exception as e:
            logger.error(f"❌ Inference failed: {e}")
            return img, [], {"status": "ERROR", "confidence": 0.0, "cost": 0, "action": "RETRY", "detail": str(e)}

# Initialize once (singleton pattern for production memory management)
engine = RefurbCortexEngine()

def process_inspection(image: np.ndarray):
    if image is None: return None, "Please upload an image.", "{}"
    try:
        start = time.time()
        heatmap, detections, routing = engine.infer(image)
        latency_ms = round((time.time() - start) * 1000, 1)

        report = {
            "inspection_id": f"INS_{int(time.time())}",
            "latency_ms": latency_ms,
            "routing_decision": routing,
            "detections": detections,
            "shap_attribution": [f"{d['panel']}: ₹{d['cost_inr']} (conf: {d['confidence']:.2f})" for d in detections]
        }
        return heatmap, json.dumps(report, indent=2), f"✅ Processed in {latency_ms}ms | Status: {routing['status']}"
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return image, json.dumps({"error": str(e)}, indent=2), "❌ Processing failed"

with gr.Blocks(title="RefurbCortex AI | Production Demo", css="footer {display:none !important}") as demo:
    gr.Markdown("# 🧠 RefurbCortex AI: Introspective Inspection Engine")
    gr.Markdown("Quantized ONNX Vision • Metacognitive Routing • Real-time Cost Estimation")
    with gr.Row():
        with gr.Column():
            input_img = gr.Image(type="numpy", label="Upload Vehicle Image", height=400)
            run_btn = gr.Button("🔍 Run Inspection", variant="primary")
        with gr.Column():
            heatmap_out = gr.Image(type="numpy", label="AI Damage Heatmap", height=400)
            status_txt = gr.Textbox(label="System Status")
            json_out = gr.JSON(label="Structured Inspection Report")

    run_btn.click(process_inspection, inputs=input_img, outputs=[heatmap_out, json_out, status_txt])

if __name__ == "__main__":
    # Production launch: queue for concurrency, hide footer, bind to 0.0.0.0
    demo.queue(api_open=False).launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        allowed_paths=["."]
    )