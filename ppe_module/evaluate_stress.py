"""Measure how good your stress model is, on pictures it has NEVER seen (your test set).

Folder layout (folder names must match the class names in labels.txt):
    dataset_test/Stressed/*.jpg
    dataset_test/Normal/*.jpg

Run:
    python evaluate_stress.py --model model_unquant.tflite --labels labels.txt --data dataset_test

It prints the accuracy and saves, in the folder  stress_results/ :
    confusion_matrix.png   (put this on your Result slide)
    summary.txt            (accuracy, precision, recall and the best threshold)
    results.csv            (one line per picture, open it in Excel)

If your test pictures are whole photos (not face crops from collect_faces.py), add  --detect-faces .
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from stress import FaceFinder, load_model, read_labels, square_crop

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_image(path):
    """Reads any picture, also from folders with spaces or non-English letters (common on Windows)."""
    return cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)


def find_class_folders(root, labels):
    root = Path(root)
    if not root.is_dir():
        raise SystemExit(f"Test folder not found: {root}")
    folders = {p.name.lower(): p for p in root.iterdir() if p.is_dir()}
    missing = [l for l in labels if l.lower() not in folders]
    if missing:
        raise SystemExit(f"These class folders are missing inside {root}: {missing}\n"
                         f"Folders found: {sorted(p.name for p in folders.values())}\n"
                         f"The folder names must match your labels.txt: {labels}")
    return [folders[l.lower()] for l in labels]


def div(a, b):
    return a / b if b else 0.0


def draw_confusion(cm, labels, accuracy, path):
    n = len(labels)
    cell, left, top = 150, 210, 140
    W, H = left + n * cell + 40, top + n * cell + 100
    img = np.full((H, W, 3), 255, np.uint8)
    cv2.putText(img, "Confusion matrix (test pictures)", (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (99, 49, 33), 2)
    cv2.putText(img, f"Accuracy: {accuracy * 100:.1f}%", (20, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 60, 60), 2)
    cv2.putText(img, "Predicted by the model", (left, top - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1)
    cv2.putText(img, "Actual", (20, top + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 60, 60), 1)
    for j, name in enumerate(labels):
        cv2.putText(img, name[:14], (left + j * cell + 10, top - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
    for i, name in enumerate(labels):
        cv2.putText(img, name[:14], (20, top + i * cell + cell // 2 + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
        row = cm[i].sum()
        for j in range(n):
            frac = div(cm[i, j], row)
            shade = tuple(int(255 - frac * (255 - c)) for c in (99, 49, 33))        # white -> navy (BGR)
            x, y = left + j * cell, top + i * cell
            cv2.rectangle(img, (x, y), (x + cell - 4, y + cell - 4), shade, -1)
            txt = (255, 255, 255) if frac > 0.5 else (30, 30, 30)
            cv2.putText(img, str(cm[i, j]), (x + 20, y + 70), cv2.FONT_HERSHEY_SIMPLEX, 1.4, txt, 3)
            cv2.putText(img, f"{frac * 100:.0f}% of {name[:8]}", (x + 14, y + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.5, txt, 1)
    cv2.imwrite(str(path), img)


def main():
    ap = argparse.ArgumentParser(description="Evaluate the Teachable Machine stress model on a test folder")
    ap.add_argument("--model", required=True, help="model_unquant.tflite or keras_model.h5")
    ap.add_argument("--labels", default=None, help="labels.txt (default: next to the model)")
    ap.add_argument("--data", required=True, help="test folder that contains one sub-folder per class")
    ap.add_argument("--stress-label", default="Stressed", help="name of the 'stressed' class")
    ap.add_argument("--detect-faces", action="store_true", help="cut out the face first (for whole photos)")
    ap.add_argument("--out", default="stress_results")
    a = ap.parse_args()

    labels = read_labels(a.labels or Path(a.model).parent / "labels.txt")
    lower = [l.lower() for l in labels]
    if a.stress_label.lower() not in lower:
        raise SystemExit(f"--stress-label '{a.stress_label}' is not in your labels {labels}")
    high = lower.index(a.stress_label.lower())
    folders = find_class_folders(a.data, labels)
    run = load_model(a.model)
    finder = FaceFinder() if a.detect_faces else None
    if a.detect_faces and not finder.kind:
        raise SystemExit("--detect-faces needs the face detector file; connect to the internet once so it can download.")

    rows, unreadable, no_face = [], 0, 0                     # rows: (path, true_index, probabilities)
    for ti, folder in enumerate(folders):
        files = sorted(p for p in folder.glob("*") if p.suffix.lower() in IMG_EXT)
        if not files:
            print(f"WARNING: no pictures in {folder}")
        for p in files:
            img = read_image(p)
            if img is None:
                unreadable += 1
                continue
            if finder:
                faces = finder.find(img)
                if not faces:
                    no_face += 1
                    continue
                img = square_crop(img, max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1])))
            probs = run(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            if len(probs) != len(labels):
                raise SystemExit(f"The model gives {len(probs)} outputs but labels.txt has {len(labels)} classes {labels}. "
                                 f"Use the labels.txt that was exported together with this model.")
            rows.append((p, ti, probs))
    if not rows:
        raise SystemExit("No test pictures could be used.")

    n = len(labels)
    cm = np.zeros((n, n), int)
    for _, ti, pr in rows:
        cm[ti, int(np.argmax(pr))] += 1
    total = int(cm.sum())
    accuracy = div(np.trace(cm), total)

    lines = [f"Test pictures used: {total}   (unreadable: {unreadable}, no face found: {no_face})",
             f"Overall accuracy: {accuracy * 100:.1f}%", "",
             f"{'Class':14}{'Pictures':>9}{'Precision':>11}{'Recall':>9}{'F1':>7}"]
    for i, name in enumerate(labels):
        tp = cm[i, i]
        prec, rec = div(tp, cm[:, i].sum()), div(tp, cm[i].sum())
        lines.append(f"{name[:13]:14}{cm[i].sum():>9}{prec * 100:>10.1f}%{rec * 100:>8.1f}%{div(2 * prec * rec, prec + rec):>7.2f}")

    lines += ["", f"'{labels[high]}' versus everything else, at different thresholds",
              "(the live system calls a person stressed when the averaged probability is at least the threshold):",
              f"{'Threshold':>9}{'Precision':>11}{'Recall':>9}{'F1':>7}{'Accuracy':>10}"]
    best = (-1, 0.6)
    for t in (0.5, 0.6, 0.7, 0.8, 0.9):
        tp = sum(1 for _, ti, pr in rows if ti == high and pr[high] >= t)
        fp = sum(1 for _, ti, pr in rows if ti != high and pr[high] >= t)
        fn = sum(1 for _, ti, pr in rows if ti == high and pr[high] < t)
        tn = sum(1 for _, ti, pr in rows if ti != high and pr[high] < t)
        prec, rec = div(tp, tp + fp), div(tp, tp + fn)
        f1 = div(2 * prec * rec, prec + rec)
        lines.append(f"{t:>9.1f}{prec * 100:>10.1f}%{rec * 100:>8.1f}%{f1:>7.2f}{div(tp + tn, tp + tn + fp + fn) * 100:>9.1f}%")
        if f1 > best[0] + 1e-9:
            best = (f1, t)
    lines.append(f"Best F1 at threshold {best[1]:.1f}.  Use it with:  --stress-threshold {best[1]:.1f}")

    wrong = sorted([(float(np.max(pr)), p, ti, int(np.argmax(pr))) for p, ti, pr in rows if int(np.argmax(pr)) != ti], reverse=True)
    lines += ["", f"Most confident mistakes ({len(wrong)} wrong in total):"]
    lines += [f"   {p.parent.name}/{p.name}: really {labels[ti]}, model said {labels[pi]} ({c * 100:.0f}%)" for c, p, ti, pi in wrong[:12]] or ["   none"]

    verdict = ("Good: ready for the demo." if accuracy >= 0.85 else
               "Acceptable for a student project. More varied training pictures will help." if accuracy >= 0.70 else
               "Weak. Add more training pictures (more people, more lighting, clearer expressions), retrain, and test again.")
    lines += ["", "Verdict: " + verdict,
              "Note: this measures single pictures. In the live system the answer is averaged over several frames, which is usually steadier."]

    out = Path(a.out)
    out.mkdir(exist_ok=True)
    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    with open(out / "results.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["picture", "actual", "predicted", f"probability_{labels[high]}", "correct"])
        for p, ti, pr in rows:
            pi = int(np.argmax(pr))
            w.writerow([f"{p.parent.name}/{p.name}", labels[ti], labels[pi], f"{pr[high]:.3f}", ti == pi])
    draw_confusion(cm, labels, accuracy, out / "confusion_matrix.png")
    print("\n".join(lines))
    print(f"\nSaved: {out / 'confusion_matrix.png'}, {out / 'summary.txt'}, {out / 'results.csv'}")


if __name__ == "__main__":
    main()
