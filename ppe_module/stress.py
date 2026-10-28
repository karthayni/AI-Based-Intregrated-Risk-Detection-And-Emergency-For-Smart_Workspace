"""Module 2: stress detection with a Teachable Machine image model.

How it fits with Module 1 (ppe_detect.py):
  1. ppe_detect.py finds each person.
  2. This file finds that person's face, asks the Teachable Machine model "Stressed or Normal?",
     smooths the answer over a few frames, and hands it back.
  3. ppe_detect.py combines the stress result with the PPE / danger-zone result.

Test the Teachable Machine model on its own first:
    python stress.py --model model_unquant.tflite --labels labels.txt
(press q to quit)
"""
import argparse
import urllib.request
from collections import deque
from pathlib import Path

import cv2
import numpy as np

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
YUNET_URLS = [
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/" + YUNET_FILE,
]


# ----------------------------------------------------------------------------- labels
def read_labels(path):
    """Teachable Machine labels.txt looks like:  '0 Stressed' / '1 Normal'. Returns ['Stressed', 'Normal']."""
    labels = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        head, _, tail = line.partition(" ")
        labels.append(tail.strip() if head.isdigit() and tail else line)
    return labels


# ----------------------------------------------------------------------------- model loading
def _load_tflite(path):
    """Try the small LiteRT package first, then full TensorFlow. Returns a function image->probabilities."""
    Interpreter = None
    try:
        from ai_edge_litert.interpreter import Interpreter          # pip install ai-edge-litert
    except Exception as e1:
        try:
            from tensorflow.lite.python.interpreter import Interpreter  # pip install tensorflow
        except Exception as e2:
            import sys
            raise RuntimeError(
                "Cannot load the .tflite model.\n"
                f"  Python used : {sys.executable}  ({sys.version.split()[0]})\n"
                f"  ai-edge-litert problem : {type(e1).__name__}: {e1}\n"
                f"  tensorflow problem     : {type(e2).__name__}: {e2}\n"
                f"  Fix: run   \"{sys.executable}\" -m pip install ai-edge-litert")
    it = Interpreter(model_path=str(path))
    it.allocate_tensors()
    inp, out = it.get_input_details()[0], it.get_output_details()[0]
    size = int(inp["shape"][1]), int(inp["shape"][2])

    def run(rgb):
        x = cv2.resize(rgb, (size[1], size[0]))
        if inp["dtype"] == np.uint8:                                  # quantised Teachable Machine model
            x = x.astype(np.uint8)
        else:                                                         # float model: scale to -1..1
            x = (x.astype(np.float32) / 127.5) - 1.0
        it.set_tensor(inp["index"], x[None, ...])
        it.invoke()
        y = it.get_tensor(out["index"])[0].astype(np.float32)
        if out["dtype"] == np.uint8:                                  # de-quantise
            scale, zero = out["quantization"]
            y = (y - zero) * scale
        return y
    return run


def _load_keras(path):
    """Teachable Machine 'keras_model.h5'. Newer Keras versions reject one old argument, so we patch it."""
    try:
        import tensorflow as tf
    except Exception:
        raise RuntimeError("To run keras_model.h5 install TensorFlow (pip install tensorflow), "
                           "or export the TensorFlow Lite model instead and use: pip install ai-edge-litert")

    class DepthwiseConv2D(tf.keras.layers.DepthwiseConv2D):
        def __init__(self, *args, **kwargs):
            kwargs.pop("groups", None)
            super().__init__(*args, **kwargs)

    model = tf.keras.models.load_model(str(path), compile=False,
                                       custom_objects={"DepthwiseConv2D": DepthwiseConv2D})
    size = model.input_shape[1:3]

    def run(rgb):
        x = cv2.resize(rgb, (size[1], size[0])).astype(np.float32)
        x = (x / 127.5) - 1.0
        return np.asarray(model(x[None, ...], training=False))[0]
    return run


def load_model(path):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Stress model not found: {path}")
    return _load_tflite(path) if path.suffix.lower() == ".tflite" else _load_keras(path)


# ----------------------------------------------------------------------------- face finding
class FaceFinder:
    """Finds faces. Order of preference:
       1. YuNet (a small 230 KB file, downloaded once next to the script),
       2. Haar cascade (comes with older OpenCV 4 versions),
       3. nothing: the caller then uses the head region of the person box instead."""

    def __init__(self, model_dir=None, download=True):
        model_dir = Path(model_dir) if model_dir else Path(__file__).resolve().parent   # next to the scripts
        self.kind = None
        self.haar = None
        self.yunet = None
        yn = Path(model_dir) / YUNET_FILE
        if not yn.exists() and download and hasattr(cv2, "FaceDetectorYN"):
            for url in YUNET_URLS:
                try:
                    with urllib.request.urlopen(url, timeout=15) as r:
                        data = r.read()
                    if len(data) > 100_000:                          # otherwise it was an error page
                        yn.write_bytes(data)
                        break
                except Exception:
                    pass
        if yn.exists() and hasattr(cv2, "FaceDetectorYN"):
            try:
                self.yunet = cv2.FaceDetectorYN.create(str(yn), "", (320, 320), 0.7, 0.3, 50)
                self.kind = "yunet"
                return
            except Exception:
                self.yunet = None
        try:
            path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            if path.exists():
                c = cv2.CascadeClassifier(str(path))
                if not c.empty():
                    self.haar, self.kind = c, "haar"
        except Exception:
            pass

    def find(self, bgr):
        """Returns a list of (x1, y1, x2, y2) face boxes."""
        if self.kind == "haar":
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            f = self.haar.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
            return [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in f]
        if self.kind == "yunet":
            h, w = bgr.shape[:2]
            self.yunet.setInputSize((w, h))
            _, faces = self.yunet.detect(bgr)
            if faces is None:
                return []
            return [(int(f[0]), int(f[1]), int(f[0] + f[2]), int(f[1] + f[3])) for f in faces]
        return []


def head_region(box, frame_shape):
    """Fallback when no face detector is available: the top part of the person box."""
    x1, y1, x2, y2 = box
    h, w = y2 - y1, x2 - x1
    side = min(w, 0.30 * h)
    cx = (x1 + x2) / 2
    hx1, hx2 = int(cx - side / 2), int(cx + side / 2)
    hy1, hy2 = int(y1), int(y1 + side * 1.1)
    H, W = frame_shape[:2]
    return max(hx1, 0), max(hy1, 0), min(hx2, W), min(hy2, H)


def square_crop(bgr, box, margin=0.25):
    """Crop a square around the box with some margin (Teachable Machine expects square images)."""
    H, W = bgr.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * (1 + margin)
    a, b = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
    c, d = int(min(cx + side / 2, W)), int(min(cy + side / 2, H))
    return bgr[b:d, a:c]


# ----------------------------------------------------------------------------- stress estimator
class StressEstimator:
    """Gives every person a stress result: {'high': bool, 'label': str, 'conf': float, 'face': box or None}."""

    def __init__(self, model_path, labels_path=None, high_label="Stressed",
                 every=5, threshold=0.6, history=8, model_dir=None):
        self.run = load_model(model_path)
        labels_path = labels_path or (Path(model_path).parent / "labels.txt")
        self.labels = read_labels(labels_path)
        lower = [l.lower() for l in self.labels]
        if high_label.lower() not in lower:
            raise SystemExit(f"--stress-label '{high_label}' is not in your labels {self.labels}. "
                             f"Use one of them, e.g. --stress-label \"{self.labels[0]}\".")
        self.high_idx = lower.index(high_label.lower())
        self.every, self.threshold = every, threshold
        self.history = history
        self.faces = FaceFinder(model_dir)
        self.tracks = {}              # id -> dict(center, size, hist, last_face_frame, face)
        self.next_id = 0
        self.frame_no = 0

    # tiny centroid tracker so each person keeps their own smoothing history
    def _match(self, persons):
        ids, used = [], set()
        for (x1, y1, x2, y2) in persons:
            c = ((x1 + x2) / 2, (y1 + y2) / 2)
            size = max(x2 - x1, y2 - y1)
            best, bd = None, 0.6 * size
            for tid, t in self.tracks.items():
                if tid in used:
                    continue
                d = ((c[0] - t["center"][0]) ** 2 + (c[1] - t["center"][1]) ** 2) ** 0.5
                if d < bd:
                    best, bd = tid, d
            if best is None:
                best = self.next_id
                self.next_id += 1
                self.tracks[best] = dict(hist=deque(maxlen=self.history), last_face=-999, face=None, seen=0)
            self.tracks[best].update(center=c, seen=self.frame_no)
            used.add(best)
            ids.append(best)
        for tid in [t for t, v in self.tracks.items() if self.frame_no - v["seen"] > 60]:
            del self.tracks[tid]
        return ids

    def update(self, frame, persons):
        """persons: list of (x1,y1,x2,y2). Returns a list (same order) of stress dicts or None."""
        self.frame_no += 1
        ids = self._match(persons)
        classify_now = (self.frame_no % self.every == 1) or self.every <= 1
        faces = self.faces.find(frame) if (classify_now and persons and self.faces.kind) else []
        out = []
        for box, tid in zip(persons, ids):
            t = self.tracks[tid]
            if classify_now:
                face = self._face_for_person(box, faces, frame.shape)
                if face is not None:
                    crop = square_crop(frame, face)
                    if crop.size and min(crop.shape[:2]) >= 30:
                        p = self.run(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                        t["hist"].append(float(p[self.high_idx]))
                        t["last_face"], t["face"] = self.frame_no, face
            if self.frame_no - t["last_face"] > 45 or not t["hist"]:
                out.append(None)                                   # no recent face: stress unknown
                continue
            p_high = sum(t["hist"]) / len(t["hist"])
            high = len(t["hist"]) >= 3 and p_high >= self.threshold
            out.append(dict(high=high, label=self.labels[self.high_idx] if high else "Normal",
                            conf=p_high if high else 1 - p_high, face=t["face"]))
        return out

    def _face_for_person(self, box, faces, shape):
        x1, y1, x2, y2 = box
        h = y2 - y1
        best, area = None, 0
        for f in faces:                                            # a face must sit in the upper part of the person
            cx, cy = (f[0] + f[2]) / 2, (f[1] + f[3]) / 2
            if x1 <= cx <= x2 and y1 <= cy <= y1 + 0.5 * h:
                a = (f[2] - f[0]) * (f[3] - f[1])
                if a > area:
                    best, area = f, a
        if best is None and not self.faces.kind:                   # no detector installed: use the head region
            return head_region(box, shape)
        return best


# ----------------------------------------------------------------------------- stand-alone test
def main():
    ap = argparse.ArgumentParser(description="Test your Teachable Machine stress model on the webcam")
    ap.add_argument("--model", required=True, help="model_unquant.tflite or keras_model.h5")
    ap.add_argument("--labels", default=None, help="labels.txt (default: next to the model)")
    ap.add_argument("--stress-label", default="Stressed")
    ap.add_argument("--source", default="0")
    a = ap.parse_args()
    src = int(a.source) if a.source.isdigit() else a.source
    est = StressEstimator(a.model, a.labels, a.stress_label, every=2, threshold=0.6)
    print("Labels:", est.labels, "| face finder:", est.faces.kind or "none (using head region)")
    cap = cv2.VideoCapture(src)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        res = est.update(frame, [(0, 0, W, H)])[0]            # treat the whole picture as one person
        if res:
            col = (50, 50, 230) if res["high"] else (60, 170, 60)
            if res["face"]:
                f = res["face"]
                cv2.rectangle(frame, (f[0], f[1]), (f[2], f[3]), col, 2)
            cv2.putText(frame, f'{res["label"]} {res["conf"]*100:.0f}%', (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 2)
        else:
            cv2.putText(frame, "no face found", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 2)
        cv2.imshow("Stress test", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
