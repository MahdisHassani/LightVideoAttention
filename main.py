from src.data.extract_frames import extract_frames
from src.data.compute_flow import process_video_frames
import os

raw_root = "data/raw"
processed_root = "data/processed"

splits = ["train", "val", "test"]

for split in splits:
    split_path = os.path.join(raw_root, split)

    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)

        for video in os.listdir(class_path):
            video_path = os.path.join(class_path, video)

            video_name = video.replace(".avi", "")

            frame_dir = os.path.join(
                processed_root, split, "frames", class_name, video_name
            )

            flow_dir = os.path.join(
                processed_root, split, "flows", class_name, video_name
            )

            print(f"Processing: {split} | {class_name} | {video}")

            extract_frames(video_path, frame_dir)
            process_video_frames(frame_dir, flow_dir)