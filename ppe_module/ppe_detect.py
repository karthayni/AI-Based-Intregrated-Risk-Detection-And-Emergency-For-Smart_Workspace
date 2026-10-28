"""Module 1 + Module 2 combined: PPE detection, danger zone and (optional) stress detection, with alerts.

Draw the danger zone once:   python ppe_detect.py --source 0 --draw-zone
Run detection:               python ppe_detect.py --weights runs/ppe/weights/best.pt --source 0
Run on a video file:         python ppe_detect.py --weights best.pt --source site.mp4
With stress detection (Teachable Machine model):
    python ppe_detect.py --weights best.pt --source 0 --stress-model model_unquant.tflite
Show the live picture in the dashboard website (the website's Live camera page starts this for you):
    python ppe_detect.py --weights best.pt --source 0 --stream 5001 --no-window
Press 'f' for full screen, 'q' to quit.
Bigger window:                python ppe_detect.py --weights best.pt --source 0 --fullscreen
"""
import argparse, csv, json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ---- CONFIG -----------------------------------------------------------------
ALIASES = {                            # model class name (lower-case) -> PPE category
    "helmet": ["hardhat", "hard hat", "helmet"],
    "vest":   ["safety vest", "vest"],
    "gloves": ["gloves", "glove"],
    "shoes":  ["safety shoes", "boots", "boot", "shoes"],
}
ZONE_FILE = "zone.json"
LOG_DIR = Path("logs")
COLORS = {"SAFE": (60, 170, 60), "CAUTION": (200, 120, 0), "VIOLATION": (0, 190, 255), "HIGH_RISK": (50, 50, 230)}
RANK = {"SAFE": 0, "CAUTION": 1, "VIOLATION": 2, "HIGH_RISK": 3}
# -----------------------------------------------------------------------------


def category(name):
    """Map a model class name to a PPE category, or None (ignores 'NO-Hardhat' style classes)."""
    n = name.lower().strip()
    if n.startswith("no-") or n.startswith("no "):
        return None
    for cat, words in ALIASES.items():
        if n in words:
            return cat
    return None


def in_zone(point, zone):
    if zone is None or len(zone) < 3:
        return False
    return cv2.pointPolygonTest(np.array(zone, np.int32), (float(point[0]), float(point[1])), False) >= 0


def evaluate(persons, ppe, zone, required, stress=None):
    """persons: [(x1,y1,x2,y2)], ppe: [(category,(x1,y1,x2,y2))], stress: optional list (one per person) of
    None or {'high': bool, ...} from stress.py  ->  list of result dicts.

    Status rules (from the project flowchart):
      PPE complete, stress not high ........ SAFE
      PPE complete, stress HIGH ............ CAUTION   (stress alert only)
      PPE missing, outside zone, stress ok . VIOLATION
      PPE missing, inside zone ............. HIGH_RISK
      PPE missing, outside zone, stress HIGH HIGH_RISK (escalated)
    """
    results = []
    for i, (x1, y1, x2, y2) in enumerate(persons):
        s = stress[i] if stress and i < len(stress) else None
        high_stress = bool(s and s.get("high"))
        h = max(y2 - y1, 1)
        found = set()
        for cat, (a1, b1, a2, b2) in ppe:
            cx, cy = (a1 + a2) / 2, (b1 + b2) / 2
            if not (x1 <= cx <= x2 and y1 <= cy <= y2):
                continue                                   # PPE is not on this person
            if cat == "helmet" and cy > y1 + 0.45 * h:     # helmet must be near the head
                continue
            if cat == "shoes" and cy < y1 + 0.70 * h:      # shoes must be near the feet
                continue
            found.add(cat)
        missing = [r for r in required if r not in found]
        foot = ((x1 + x2) / 2, y2)                         # bottom-centre = where the person stands
        inside = in_zone(foot, zone)
        if not missing:
            status = "CAUTION" if high_stress else "SAFE"
        elif inside or high_stress:
            status = "HIGH_RISK"
        else:
            status = "VIOLATION"
        results.append(dict(box=(x1, y1, x2, y2), missing=missing, in_zone=inside, status=status, stress=s))
    return results


def send_alert(status, details, snapshot):
    """Hook: replace the print with SMS / email / Telegram / dashboard API call."""
    print(f"[ALERT] {status}: {details}  -> {snapshot}")


def log_event(frame, status, details):
    (LOG_DIR / "snapshots").mkdir(parents=True, exist_ok=True)
    ts = datetime.now()
    snap = LOG_DIR / "snapshots" / f"{ts:%Y%m%d_%H%M%S}_{status}.jpg"
    cv2.imwrite(str(snap), frame)
    new = not (LOG_DIR / "events.csv").exists()
    with open(LOG_DIR / "events.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "status", "details", "snapshot"])
        w.writerow([ts.isoformat(timespec="seconds"), status, details, snap])
    send_alert(status, details, snap)


class Stream:
    """Serves the annotated frames as a live picture for the website: http://127.0.0.1:<port>/video (MJPEG),
    /status (JSON) and /shot (saves a screenshot). Only reachable from this computer."""

    def __init__(self, port):
        self.lock = threading.Lock()
        self.jpg, self.count = None, 0
        self.info = {"running": True, "status": "SAFE", "people": 0, "stress_high": 0, "fps": 0.0}
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _json(self, obj, code=200):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                path = self.path.split("?")[0]
                if path == "/status":
                    with outer.lock:
                        self._json(dict(outer.info))
                elif path == "/shot":
                    with outer.lock:
                        jpg, status = outer.jpg, outer.info["status"]
                    if jpg is None:
                        return self._json({"error": "no picture yet"}, 409)
                    Path("screenshots").mkdir(exist_ok=True)
                    name = f"{datetime.now():%Y%m%d_%H%M%S}_{status}.jpg"
                    (Path("screenshots") / name).write_bytes(jpg)
                    self._json({"saved": f"screenshots/{name}"})
                elif path == "/video":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    last = -1
                    try:
                        while True:
                            with outer.lock:
                                jpg, n, running = outer.jpg, outer.count, outer.info["running"]
                            if jpg is not None and n != last:
                                last = n
                                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                            elif not running:
                                break
                            else:
                                time.sleep(0.03)
                    except (BrokenPipeError, ConnectionError, OSError):
                        pass
                else:
                    self._json({"error": "use /video, /status or /shot"}, 404)

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def update(self, frame, **info):
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            with self.lock:
                self.jpg, self.count = buf.tobytes(), self.count + 1
                self.info.update(info)

    def stop(self):
        with self.lock:
            self.info["running"] = False


def draw_zone_tool(source, width=1280, height=720):
    cap = cv2.VideoCapture(source)
    if isinstance(source, int):                       # same size as live detection, so the zone lines up
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    for _ in range(5):                                # let the camera settle
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit("Cannot read from source")
    pts = []

    def click(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append([x, y])

    cv2.namedWindow("Draw danger zone")
    cv2.setMouseCallback("Draw danger zone", click)
    while True:
        img = frame.copy()
        for p in pts:
            cv2.circle(img, tuple(p), 5, (0, 0, 255), -1)
        if len(pts) > 1:
            cv2.polylines(img, [np.array(pts, np.int32)], len(pts) > 2, (0, 0, 255), 2)
        cv2.putText(img, "Click corners | s = save | r = reset | q = cancel", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("Draw danger zone", img)
        k = cv2.waitKey(30) & 0xFF
        if k == ord("r"):
            pts.clear()
        elif k == ord("s") and len(pts) >= 3:
            json.dump(pts, open(ZONE_FILE, "w"))
            print(f"Saved {len(pts)} points to {ZONE_FILE}")
            break
        elif k == ord("q"):
            break
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/ppe/weights/best.pt")
    ap.add_argument("--source", default="0", help="0 = webcam, or path to a video file")
    ap.add_argument("--person-weights", default="yolov8n.pt",
                    help="model that finds people (your PPE dataset has no 'person' class, so we use the standard COCO model)")
    ap.add_argument("--require", nargs="+", default=["helmet", "vest"],
                    help="PPE that every person must wear, e.g. --require helmet vest gloves shoes")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--persist", type=int, default=5, help="frames a status must persist before it is logged")
    ap.add_argument("--cooldown", type=float, default=10, help="seconds between logged alerts")
    ap.add_argument("--stress-model", default=None,
                    help="Teachable Machine model (model_unquant.tflite or keras_model.h5). Leave out to run PPE only.")
    ap.add_argument("--stress-labels", default=None, help="labels.txt (default: next to the model)")
    ap.add_argument("--stress-label", default="Stressed", help="name of the 'stressed' class in your model")
    ap.add_argument("--stress-threshold", type=float, default=0.6, help="average probability needed to call stress HIGH")
    ap.add_argument("--stress-every", type=int, default=5, help="check stress every N frames (higher = faster)")
    ap.add_argument("--draw-zone", action="store_true")
    ap.add_argument("--stream", type=int, default=0, help="port to share the live picture with the website (0 = off)")
    ap.add_argument("--no-window", action="store_true", help="do not open the OpenCV window (use with --stream)")
    ap.add_argument("--fullscreen", action="store_true", help="start in full screen (press f to toggle, q to quit)")
    ap.add_argument("--width", type=int, default=1280, help="camera width to request (bigger = sharper picture)")
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()
    source = int(a.source) if a.source.isdigit() else a.source

    if a.draw_zone:
        return draw_zone_tool(source, a.width, a.height)

    if a.stress_model:
        # load the stress runtime BEFORE torch/ultralytics: on Windows the other order can break its libraries
        try:
            import ai_edge_litert.interpreter  # noqa: F401
        except Exception as e:
            print(f"(note) ai-edge-litert could not be loaded first: {type(e).__name__}: {e}", flush=True)
    from ultralytics import YOLO
    model = YOLO(a.weights)                 # your trained PPE model (helmet, vest, gloves, boots...)
    person_model = YOLO(a.person_weights)   # standard model, class 0 = person
    zone = json.load(open(ZONE_FILE)) if Path(ZONE_FILE).exists() else None
    if zone is None:
        print("No zone.json found: danger-zone check disabled (run with --draw-zone first).")

    estimator = None
    if a.stress_model:
        from stress import StressEstimator
        estimator = StressEstimator(a.stress_model, a.stress_labels, a.stress_label,
                                    every=a.stress_every, threshold=a.stress_threshold)
        print(f"Stress detection ON | labels {estimator.labels} | face finder: {estimator.faces.kind or 'none (head region)'}")

    cap = cv2.VideoCapture(source)
    if isinstance(source, int):                       # ask the webcam for a larger picture
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source {a.source}. For the webcam try source 1, and close Zoom or Teams.")
    WIN = "PPE Monitoring"
    show_window = not a.no_window
    if show_window:
        cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)       # resizable window (drag corners or maximize button)
        cv2.resizeWindow(WIN, 1280, 720)
    full = a.fullscreen
    if full and show_window:
        cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    stream = Stream(a.stream) if a.stream else None
    if stream:
        print(f"Live picture for the website: http://127.0.0.1:{a.stream}/video", flush=True)
    bad_frames, last_log, frames, t_fps, fps = 0, 0, 0, time.time(), 0.0
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        persons, ppe = [], []
        pres = person_model(frame, conf=a.conf, classes=[0], verbose=False)[0]
        for b in pres.boxes:
            persons.append(tuple(float(v) for v in b.xyxy[0]))
        res = model(frame, conf=a.conf, verbose=False)[0]
        for b in res.boxes:
            cat = category(res.names[int(b.cls)])
            if cat:
                box = tuple(float(v) for v in b.xyxy[0])
                ppe.append((cat, box))
                cv2.rectangle(frame, tuple(int(v) for v in box[:2]), tuple(int(v) for v in box[2:]), (200, 200, 0), 1)

        stress = estimator.update(frame, persons) if estimator else None
        results = evaluate(persons, ppe, zone, a.require, stress)
        overall = max((r["status"] for r in results), key=RANK.get, default="SAFE")

        if zone:
            cv2.polylines(frame, [np.array(zone, np.int32)], True, (0, 0, 255), 2)
        for r in results:
            x1, y1, x2, y2 = (int(v) for v in r["box"])
            c = COLORS[r["status"]]
            cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
            txt = r["status"] if not r["missing"] else f'{r["status"]}: no {", ".join(r["missing"])}'
            cv2.putText(frame, txt, (x1, max(y1 - 8, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, c, 2)
            if estimator:
                s = r["stress"]
                if s:
                    sc = (50, 50, 230) if s["high"] else (60, 170, 60)
                    cv2.putText(frame, f'Stress: {"HIGH" if s["high"] else "LOW"} {s["conf"]*100:.0f}%',
                                (x1, min(y2 + 22, frame.shape[0] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sc, 2)
                    if s.get("face"):
                        fx1, fy1, fx2, fy2 = s["face"]
                        cv2.rectangle(frame, (fx1, fy1), (fx2, fy2), sc, 1)
                else:
                    cv2.putText(frame, "Stress: no face", (x1, min(y2 + 22, frame.shape[0] - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.rectangle(frame, (0, 0), (260, 34), COLORS[overall], -1)
        cv2.putText(frame, f"STATUS: {overall}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # log only if the problem persists for a few frames (avoids flicker) and respect a cooldown
        bad_frames = bad_frames + 1 if overall != "SAFE" else 0
        if bad_frames >= a.persist and time.time() - last_log > a.cooldown:
            def describe(r):
                text = f'missing {",".join(r["missing"]) or "none"} (in_zone={r["in_zone"]})'
                if r.get("stress"):
                    text += f' stress={"HIGH" if r["stress"]["high"] else "LOW"}'
                return text
            details = "; ".join(describe(r) for r in results if r["status"] != "SAFE")
            log_event(frame, overall, details)
            last_log = time.time()

        frames += 1
        if time.time() - t_fps >= 1:
            fps, frames, t_fps = frames / (time.time() - t_fps), 0, time.time()
        if stream:
            stream.update(frame, status=overall, people=len(results), fps=round(fps, 1),
                          stress_high=sum(1 for r in results if r.get("stress") and r["stress"]["high"]))
        if not show_window:
            continue
        shown = frame.copy()                          # key hints only on the screen, not in saved pictures
        cv2.putText(shown, "s: screenshot  f: full screen  q: quit", (8, shown.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow(WIN, shown)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):                             # save a screenshot for the report / PPT
            Path("screenshots").mkdir(exist_ok=True)
            shot = Path("screenshots") / f"{datetime.now():%Y%m%d_%H%M%S}_{overall}.jpg"
            ok_s, buf_s = cv2.imencode(".jpg", frame)
            if ok_s:
                buf_s.tofile(str(shot))
                print("Saved screenshot:", shot)
        if k == ord("f"):                             # toggle full screen
            full = not full
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if full else cv2.WINDOW_NORMAL)
    cap.release()
    if stream:
        stream.stop()
        time.sleep(0.3)
    if show_window:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
