"""Compare pretrained YOLO detection and segmentation prediction counts."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ultralytics import YOLO


def collect_counts(model_path: str, image_dir: Path, confidence: float) -> dict[str, int]:
    model = YOLO(model_path)
    counts = {}
    for result in model.predict(source=str(image_dir), conf=confidence, verbose=False):
        counts[Path(result.path).name] = 0 if result.boxes is None else len(result.boxes)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--detection-model", default="yolov8n.pt")
    parser.add_argument("--segmentation-model", default="yolov8n-seg.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--output-csv", default="waste_prediction_counts.csv")
    args = parser.parse_args()

    image_dir = Path(args.input_dir)
    detection = collect_counts(args.detection_model, image_dir, args.confidence)
    segmentation = collect_counts(args.segmentation_model, image_dir, args.confidence)
    common = sorted(set(detection) & set(segmentation))
    if not common:
        raise RuntimeError("The two models returned no matching image filenames")

    frame = pd.DataFrame(
        {
            "image": common,
            "detection_count": [detection[name] for name in common],
            "segmentation_count": [segmentation[name] for name in common],
        }
    )
    frame["absolute_count_difference"] = (
        frame["detection_count"] - frame["segmentation_count"]
    ).abs()
    frame.to_csv(args.output_csv, index=False)

    x = frame["detection_count"].to_numpy()
    y = frame["segmentation_count"].to_numpy()
    correlation = np.corrcoef(x, y)[0, 1] if len(frame) > 1 and x.std() and y.std() else float("nan")
    print(f"Images compared: {len(frame)}")
    print(f"Detection/segmentation count correlation: {correlation:.4f}")
    print(f"Mean absolute count difference: {frame['absolute_count_difference'].mean():.4f}")
    print(f"Per-image counts written to {args.output_csv}")


if __name__ == "__main__":
    main()
