import argparse

import cv2

from utils.mask_detector import MaskDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Real-time face mask detection")
    parser.add_argument(
        "--model",
        type=str,
        default="models/mask_detector_model.h5",
        help="Path to trained mask detector model",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="models/labels.json",
        help="Path to labels json (optional)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="models/model_config.json",
        help="Path to model config json (optional)",
    )
    parser.add_argument(
        "--face-cascade",
        type=str,
        default=None,
        help="Path to Haar cascade (optional)",
    )
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument(
        "--max-width",
        type=int,
        default=800,
        help="Resize frame to this width for faster processing",
    )
    parser.add_argument(
        "--screenshot-dir",
        type=str,
        default="static/sample_images",
        help="Directory to save no-mask screenshots",
    )
    parser.add_argument(
        "--screenshot-cooldown",
        type=float,
        default=5.0,
        help="Cooldown in seconds between screenshots",
    )
    parser.add_argument(
        "--alert-cooldown",
        type=float,
        default=2.0,
        help="Cooldown in seconds between alert sounds",
    )
    parser.add_argument(
        "--no-alert",
        action="store_true",
        help="Disable sound alerts",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    detector = MaskDetector(
        model_path=args.model,
        labels_path=args.labels,
        config_path=args.config,
        cascade_path=args.face_cascade,
        alert=not args.no_alert,
        alert_cooldown=args.alert_cooldown,
        screenshot_dir=args.screenshot_dir,
        screenshot_cooldown=args.screenshot_cooldown,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {args.camera}. "
            "Allow camera access for your terminal or IDE and try again."
        )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if args.max_width and frame.shape[1] > args.max_width:
                scale = args.max_width / frame.shape[1]
                frame = cv2.resize(
                    frame, (args.max_width, int(frame.shape[0] * scale))
                )

            annotated, _ = detector.annotate_frame(frame)
            cv2.imshow("Face Mask Detector", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
