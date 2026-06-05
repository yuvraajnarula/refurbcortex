import os, cv2, numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import onnxruntime as ort
from app.utils.logger import app_logger

class QuantizedYOLOEngine:
    def __init__(self, model_path: str = "./models/yolov8s.onnx", conf_thresh: float = 0.45, iou_thresh: float = 0.5):
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.model_path = model_path
        
        if not os.path.exists(self.model_path):
            app_logger.warning(f"{self.model_path} not found. Run: yolo export model=yolov8s.pt format=onnx opset=11")
            raise FileNotFoundError
        
        # Dynamic provider selection
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
        self.session = ort.InferenceSession(
            self.model_path, 
            providers=providers,
            sess_options=ort.SessionOptions()
        )
        self.input_shape = self.session.get_inputs()[0].shape
        app_logger.info(f"Quantized YOLO loaded | Providers: {providers} | Shape: {self.input_shape}")
        self._warmup()

    def _warmup(self):
        dummy = np.random.randn(1, *self.input_shape[1:]).astype(np.float32)
        self.session.run(None, {"images": dummy})

    def preprocess(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[float, float], float]:
        h, w = img.shape[:2]
        scale = 640 / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((640, 640, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized
        normalized = (padded.transpose(2, 0, 1) / 255.0).astype(np.float32)
        return normalized[np.newaxis, ...], (scale, 0.0), 640.0

    def postprocess(self, outputs: np.ndarray, input_shape: Tuple[float, float], orig_h: int, orig_w: int) -> List[Dict]:
        outputs = outputs.squeeze()
        boxes = outputs.T[:, :4]
        scores = outputs.T[:, 4:5] * outputs.T[:, 5:]
        classes = np.argmax(scores, axis=1)
        confs = np.max(scores, axis=1)
        
        mask = confs > self.conf_thresh
        boxes, classes, confs = boxes[mask], classes[mask], confs[mask]
        if len(boxes) == 0: return []

        indices = cv2.dnn.NMSBoxes(boxes.tolist(), confs.tolist(), self.conf_thresh, self.iou_thresh)
        results = []
        scale, _ = input_shape
        for idx in indices:
            x1, y1, x2, y2 = boxes[idx] * scale
            results.append({
                "bbox": [float(round(x1)), float(round(y1)), float(round(x2)), float(round(y2))],
                "class_id": int(classes[idx]),
                "confidence": float(confs[idx]),
                "area_cm2": float(((x2-x1)*(y2-y1))/100)  # px to cm² approx
            })
        return results

    def run(self, img: np.ndarray) -> List[Dict]:
        try:
            h, w = img.shape[:2]
            tensor, scale, _ = self.preprocess(img)
            outputs = self.session.run(None, {"images": tensor})[0]
            return self.postprocess(outputs, scale, h, w)
        except Exception as e:
            app_logger.error(f"Quantized YOLO inference failed: {e}")
            return []