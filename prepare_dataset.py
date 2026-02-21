import argparse
import os
import random
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from cnn_model import CLASS_NAMES


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def find_source_images(source_root: Path) -> Dict[str, List[Path]]:
    result: Dict[str, List[Path]] = {name: [] for name in CLASS_NAMES}
    for cls in CLASS_NAMES:
        folder = source_root / cls
        if folder.is_dir():
            result[cls] = sorted([p for p in folder.glob("*.png") if p.is_file()])
    return result


def augment_square(gray: np.ndarray, out_size: int = 64) -> np.ndarray:
    h, w = gray.shape[:2]
    center = (w / 2.0, h / 2.0)
    angle = random.uniform(-8.0, 8.0)
    scale = random.uniform(0.95, 1.05)
    tx = random.uniform(-3.0, 3.0)
    ty = random.uniform(-3.0, 3.0)

    m = cv2.getRotationMatrix2D(center, angle, scale)
    m[0, 2] += tx
    m[1, 2] += ty
    warped = cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)

    alpha = random.uniform(0.85, 1.2)
    beta = random.uniform(-18, 18)
    adjusted = cv2.convertScaleAbs(warped, alpha=alpha, beta=beta)

    if random.random() < 0.4:
        k = random.choice([3, 5])
        adjusted = cv2.GaussianBlur(adjusted, (k, k), sigmaX=random.uniform(0.2, 1.2))

    if random.random() < 0.5:
        noise = np.random.normal(loc=0.0, scale=random.uniform(2.0, 9.0), size=adjusted.shape).astype(np.float32)
        adjusted = np.clip(adjusted.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    resized = cv2.resize(adjusted, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return resized


def split_train_val(paths: List[Path], val_ratio: float) -> (List[Path], List[Path]):
    if not paths:
        return [], []
    items = paths[:]
    random.shuffle(items)
    n_val = max(1, int(len(items) * val_ratio))
    val = items[:n_val]
    train = items[n_val:] if len(items) > n_val else items[:]
    return train, val


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare augmented chess-square dataset from base templates.")
    parser.add_argument("--source-root", default="grayscale_images", help="Input class folders with seed images.")
    parser.add_argument("--output-root", default="dataset", help="Output dataset root.")
    parser.add_argument("--target-per-class", type=int, default=120, help="Generated images per class.")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="Validation split ratio.")
    parser.add_argument("--image-size", type=int, default=64, help="Output square size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    train_root = output_root / "train"
    val_root = output_root / "val"

    for cls in CLASS_NAMES:
        ensure_dir(train_root / cls)
        ensure_dir(val_root / cls)

    sources = find_source_images(source_root)
    missing = [cls for cls, files in sources.items() if len(files) == 0]
    if missing:
        raise FileNotFoundError(
            f"Missing source images for classes: {missing}. "
            f"Expected folders under {source_root}."
        )

    for cls in CLASS_NAMES:
        seed_imgs = []
        for p in sources[cls]:
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            seed_imgs.append(img)
        if not seed_imgs:
            raise RuntimeError(f"No readable images for class {cls}.")

        generated_paths = []
        class_tmp = output_root / "_tmp" / cls
        ensure_dir(class_tmp)
        for i in range(args.target_per_class):
            base = random.choice(seed_imgs)
            aug = augment_square(base, out_size=args.image_size)
            out_path = class_tmp / f"{cls}_{i:04d}.png"
            cv2.imwrite(str(out_path), aug)
            generated_paths.append(out_path)

        train_paths, val_paths = split_train_val(generated_paths, args.val_ratio)
        for p in train_paths:
            p.replace(train_root / cls / p.name)
        for p in val_paths:
            p.replace(val_root / cls / p.name)

    tmp_root = output_root / "_tmp"
    if tmp_root.is_dir():
        for cls_dir in tmp_root.glob("*"):
            for f in cls_dir.glob("*"):
                f.unlink(missing_ok=True)
            cls_dir.rmdir()
        tmp_root.rmdir()

    total_train = sum(len(list((train_root / cls).glob("*.png"))) for cls in CLASS_NAMES)
    total_val = sum(len(list((val_root / cls).glob("*.png"))) for cls in CLASS_NAMES)
    print(f"Prepared dataset at '{output_root}'. train={total_train}, val={total_val}")


if __name__ == "__main__":
    main()
