import cv2
import os

def extract_frames(video_path, output_dir, frame_skip=5):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % frame_skip == 0:
            path = os.path.join(output_dir, f"{saved:04d}.jpg")
            cv2.imwrite(path, frame)
            saved += 1

        count += 1

    cap.release()