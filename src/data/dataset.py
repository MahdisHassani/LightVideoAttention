import os
import cv2
import torch
from torch.utils.data import Dataset

class VideoDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []

        frame_root = os.path.join(root_dir, "frames")
        flow_root  = os.path.join(root_dir, "flows")

        for class_name in os.listdir(frame_root):
            class_frame_path = os.path.join(frame_root, class_name)
            class_flow_path  = os.path.join(flow_root, class_name)

            for video in os.listdir(class_frame_path):
                frame_dir = os.path.join(class_frame_path, video)
                flow_dir  = os.path.join(class_flow_path, video)

                if not os.path.exists(flow_dir):
                    continue

                frames = sorted(os.listdir(frame_dir))
                flows  = sorted(os.listdir(flow_dir))

                for i in range(min(len(frames)-1, len(flows))):
                    self.samples.append((
                        os.path.join(frame_dir, frames[i]),
                        os.path.join(flow_dir, flows[i])
                    ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, flow_path = self.samples[idx]

        img = cv2.imread(img_path)
        flow = cv2.imread(flow_path, 0)

        img = cv2.resize(img, (224, 224))
        flow = cv2.resize(flow, (224, 224))

        img = img / 255.0
        img = (img - 0.5) / 0.5

        flow = flow / 255.0

        flow = (flow > 0.05).astype("float32")

        img = torch.tensor(img).permute(2, 0, 1).float()
        flow = torch.tensor(flow).unsqueeze(0).float()

        return img, flow