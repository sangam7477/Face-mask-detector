import argparse
import atexit
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

from utils.mask_detector import MaskDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Flask app for mask detection")
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
    parser.add_argument("--camera", type=int, default=0, help="Webcam index")
    parser.add_argument(
        "--max-width",
        type=int,
        default=800,
        help="Resize frame to this width for faster processing",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--no-alert",
        action="store_true",
        help="Disable sound alerts",
    )
    return parser.parse_args()


def create_app(args):
    app = Flask(__name__)

    detector = MaskDetector(
        model_path=args.model,
        labels_path=args.labels,
        config_path=args.config,
        alert=not args.no_alert,
        screenshot_dir=None,
    )

    camera_lock = threading.Lock()
    state = {
        "cap": None,
        "camera_error": None,
        "stream_active": False,
        "stats": {
            "current_faces": 0,
            "current_mask": 0,
            "current_no_mask": 0,
            "compliance": 0.0,
            "total_mask": 0,
            "total_no_mask": 0,
            "last_update": None,
        },
    }

    def open_camera():
        with camera_lock:
            if state["cap"] is not None and state["cap"].isOpened():
                return state["cap"]

            cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(args.camera, cv2.CAP_MSMF)

            print("Camera Opened:", cap.isOpened())
            if not cap.isOpened():
                state["camera_error"] = (
                    f"Could not open camera index {args.camera}. "
                    "Allow camera access for your terminal or IDE and try again."
                )
                cap.release()
                state["cap"] = None
                return None

            state["camera_error"] = None
            state["cap"] = cap
            return cap

    def release_camera():
        with camera_lock:
            if state["cap"] is not None and state["cap"].isOpened():
                state["cap"].release()
            state["cap"] = None

    atexit.register(release_camera)

    def gen_frames():
        cap = open_camera()
        if cap is None:
            placeholder = _build_status_frame(state["camera_error"])
            while True:
                ok, buffer = cv2.imencode(".jpg", placeholder)
                if not ok:
                    break
                frame_bytes = buffer.tobytes()

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
                time.sleep(1)
            return

        try:
            while True:
                success, frame = cap.read()
                print("Camera:", success)
                if not success:
                    break

                if args.max_width and frame.shape[1] > args.max_width:
                    scale = args.max_width / frame.shape[1]
                    frame = cv2.resize(
                        frame, (args.max_width, int(frame.shape[0] * scale))
                    )

                annotated, detections = detector.annotate_frame(frame)

                current_faces = len(detections)
                current_mask = sum(1 for det in detections if det.label == "Mask")
                current_no_mask = sum(1 for det in detections if det.label == "No Mask")
                total_mask = detector.total_mask
                total_no_mask = detector.total_no_mask
                total = current_mask + current_no_mask
                compliance = (current_mask / total) * 100 if total > 0 else 0.0

                with camera_lock:
                    state["stream_active"] = True
                    state["stats"].update(
                        {
                            "current_faces": current_faces,
                            "current_mask": current_mask,
                            "current_no_mask": current_no_mask,
                            "compliance": round(compliance, 1),
                            "total_mask": total_mask,
                            "total_no_mask": total_no_mask,
                            "last_update": datetime.now().strftime("%H:%M:%S"),
                        }
                    )

                ok, buffer = cv2.imencode(".jpg", annotated)
                if not ok:
                    continue
                frame_bytes = buffer.tobytes()

                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                )
        finally:
            release_camera()

    @app.route("/")
    def index():
        return render_template(
            "index.html",
            camera_error=state["camera_error"],
            stream_active=False,
        )

    @app.route("/video_feed")
    def video_feed():
        return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/stop_camera", methods=["POST"])
    def stop_camera():
        release_camera()
        with camera_lock:
            state["stream_active"] = False
        return jsonify({"ok": True})

    @app.route("/stats")
    def stats():
        with camera_lock:
            return jsonify(
                {
                    "stream_active": state["stream_active"],
                    "camera_error": state["camera_error"],
                    **state["stats"],
                }
            )

    return app


def _build_status_frame(message):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[:] = (15, 24, 38)
    cv2.putText(
        frame,
        "Camera Access Needed",
        (70, 180),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        (91, 183, 255),
        3,
    )
    cv2.putText(
        frame,
        "Grant camera permission to Terminal / IDE, then restart app.py",
        (70, 280),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.95,
        (255, 255, 255),
        2,
    )
    if message:
        cv2.putText(
            frame,
            message[:95],
            (70, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (180, 196, 214),
            2,
        )
    return frame


if __name__ == "__main__":
    args = parse_args()
    app = create_app(args)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)
