"""Real-time facial emotion recognition with a trained CNN (webcam or one image).

    pip install -e ".[webcam]"
    python scripts/webcam_demo.py --weights results/face__fedavg__seed0_weights.npz
    python scripts/webcam_demo.py --weights ... --image photo.jpg --save annotated.jpg

Faces are found with OpenCV's Haar cascade, cropped, converted to 48x48
grayscale and classified. ``--no-detect`` classifies the whole image (for
already-cropped faces). Press q to quit the webcam view.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from fedmultiemo.config import load_config
from fedmultiemo.data import CLASSES
from fedmultiemo.models import build_cnn, set_named_weights

CASCADE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


def load_model(weights_path: str):
    model = build_cnn(load_config())
    arrays = np.load(weights_path)
    set_named_weights(model, {k.replace("__", "/"): arrays[k] for k in arrays.files})
    return model


def classify(model, gray_crop: np.ndarray) -> tuple[str, float]:
    face = cv2.resize(gray_crop, (48, 48)).astype(np.float32)[None, :, :, None] / 255.0
    proba = np.asarray(model(face, training=False))[0]
    return CLASSES[int(proba.argmax())], float(proba.max())


def annotate(model, detector, frame: np.ndarray, detect: bool = True) -> list[tuple]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    boxes = detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5) if detect else [(0, 0, *gray.shape[::-1])]
    found = []
    for x, y, w, h in boxes:
        label, conf = classify(model, gray[y : y + h, x : x + w])
        found.append((x, y, w, h, label, conf))
        if frame.ndim == 3:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(
                frame, f"{label} {conf:.0%}", (x, max(y - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 0), 2
            )
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", required=True, help="*_weights.npz written by `fedmultiemo face`")
    parser.add_argument("--image", help="classify this image instead of the webcam stream")
    parser.add_argument("--save", help="write the annotated image here (with --image)")
    parser.add_argument("--no-detect", action="store_true", help="treat the whole image as one face")
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()
    model = load_model(args.weights)
    detector = cv2.CascadeClassifier(CASCADE)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"cannot read {args.image}")
        for x, y, w, h, label, conf in annotate(model, detector, frame, detect=not args.no_detect):
            print(f"face at ({x},{y},{w},{h}): {label} ({conf:.1%})")
        if args.save:
            cv2.imwrite(args.save, frame)
        return

    cap = cv2.VideoCapture(args.camera)
    last = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        annotate(model, detector, frame)
        now = time.perf_counter()
        cv2.putText(
            frame, f"{1.0 / max(now - last, 1e-6):.1f} fps", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1
        )
        last = now
        cv2.imshow("FedMultiEmo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
