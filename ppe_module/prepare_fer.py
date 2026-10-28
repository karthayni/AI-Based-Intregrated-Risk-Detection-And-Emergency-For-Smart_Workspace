"""Turn the FER-2013 dataset into the two classes Teachable Machine needs: Stressed and Normal.

FER-2013 comes as folders like:
    <fer>/train/angry  disgust  fear  happy  neutral  sad  surprise
    <fer>/test/ ...same folders...

Run (change the path to where you unzipped FER-2013):
    python prepare_fer.py --fer "C:\\Users\\karth\\Downloads\\FER2013"

Result:
    dataset/Stressed   dataset/Normal         (for training in Teachable Machine)
    dataset_test/Stressed   dataset_test/Normal   (kept apart, for evaluate_stress.py)

Mapping (you can change it with --stressed / --normal):
    Stressed = angry, fear, sad, disgust        Normal = neutral, happy        (surprise is left out)

The pictures are 48x48 grey faces. They are enlarged to 224x224 and turned into 3-colour pictures
because that is what the model expects.
"""
import argparse
import random
from pathlib import Path

import cv2
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_split(root, names):
    """Find the folder (train / test) whatever its capital letters."""
    for p in root.rglob("*"):
        if p.is_dir() and p.name.lower() in names:
            return p
    return None


def emotion_folders(split):
    return {p.name.lower(): p for p in split.iterdir() if p.is_dir()}


def pick(folders, wanted, cap, rng):
    """Take pictures evenly from each emotion in `wanted` (so one emotion does not dominate)."""
    lists = []
    for name in wanted:
        if name in folders:
            files = [p for p in folders[name].glob("*") if p.suffix.lower() in IMG_EXT]
            rng.shuffle(files)
            lists.append(files)
    out, i = [], 0
    while len(out) < cap and any(lists):
        for fl in lists:
            if fl and len(out) < cap:
                out.append(fl.pop())
        i += 1
    return out


def convert(src, dst):
    img = cv2.imdecode(np.fromfile(str(src), np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
    if ok:
        buf.tofile(str(dst))
    return ok


def build(split, out_root, mapping, cap, rng):
    folders = emotion_folders(split)
    found = sorted(folders)
    print(f"  emotion folders found in {split.name}: {found}")
    for cls, wanted in mapping.items():
        missing = [w for w in wanted if w not in folders]
        if missing:
            print(f"  note: no folder named {missing} (skipped)")
        files = pick(folders, wanted, cap, rng)
        dest = Path(out_root) / cls
        dest.mkdir(parents=True, exist_ok=True)
        n = 0
        for k, src in enumerate(files):
            if convert(src, dest / f"{cls}_{src.parent.name}_{k:05d}.jpg"):
                n += 1
        print(f"  {cls:9} -> {n} pictures in {dest}")


def main():
    ap = argparse.ArgumentParser(description="Prepare FER-2013 for Teachable Machine (Stressed / Normal)")
    ap.add_argument("--fer", required=True, help="folder where you unzipped FER-2013 (it contains train and test)")
    ap.add_argument("--train-out", default="dataset")
    ap.add_argument("--test-out", default="dataset_test")
    ap.add_argument("--train-per-class", type=int, default=600, help="pictures per class for training")
    ap.add_argument("--test-per-class", type=int, default=150, help="pictures per class for testing")
    ap.add_argument("--stressed", nargs="+", default=["angry", "fear", "sad", "disgust"])
    ap.add_argument("--normal", nargs="+", default=["neutral", "happy"])
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    root = Path(a.fer)
    if not root.is_dir():
        raise SystemExit(f"Folder not found: {root}")
    train = find_split(root, {"train", "training"})
    test = find_split(root, {"test", "testing", "val", "validation", "publictest"})
    if train is None:
        raise SystemExit(f"No 'train' folder found inside {root}. Open it in File Explorer and check where the folders "
                         f"angry / happy / sad ... are, then pass that path (it may be one folder deeper).")
    mapping = {"Stressed": [s.lower() for s in a.stressed], "Normal": [s.lower() for s in a.normal]}
    rng = random.Random(a.seed)

    print("Training pictures:")
    build(train, a.train_out, mapping, a.train_per_class, rng)
    if test is not None:
        print("Test pictures:")
        build(test, a.test_out, mapping, a.test_per_class, rng)
    else:
        print("No test folder found. Skipping the test set (you can still train).")
    print("\nDone. Next: upload dataset/Stressed and dataset/Normal to Teachable Machine.")


if __name__ == "__main__":
    main()
