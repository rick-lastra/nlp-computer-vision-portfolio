"""Compare simple OpenCV enhancement and image augmentation pipelines."""

import argparse
from pathlib import Path

import albumentations as A
import cv2


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    enhanced_dir = output_dir / "enhanced"
    augmented_dir = output_dir / "augmented"
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    augmented_dir.mkdir(parents=True, exist_ok=True)

    augmentation = A.Compose(
        [
            A.Affine(scale=(0.9, 1.1), translate_percent=(-0.05, 0.05), rotate=(-15, 15), p=0.8),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        ]
    )

    images = sorted(p for p in input_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"No supported images found in {input_dir}")

    for path in images:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping unreadable image: {path.name}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        cv2.imwrite(str(enhanced_dir / f"{path.stem}_gray_equalized.png"), equalized)
        cv2.imwrite(str(enhanced_dir / f"{path.stem}_bilateral.png"), denoised)

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for index in range(3):
            augmented = augmentation(image=rgb)["image"]
            bgr = cv2.cvtColor(augmented, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(augmented_dir / f"{path.stem}_aug_{index + 1}.jpg"), bgr)

    print(f"Processed {len(images)} images. Outputs saved under {output_dir}")


if __name__ == "__main__":
    main()
