import cv2
import os

def compute_flow(img1, img2):
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(
        g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    mag = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)

    mag = cv2.GaussianBlur(mag, (7, 7), 0)

    mag = (mag > 0.1).astype("float32")

    return mag


def process_video_frames(frame_dir, flow_dir):
    os.makedirs(flow_dir, exist_ok=True)

    frames = sorted(os.listdir(frame_dir))

    for i in range(len(frames) - 1):
        img1 = cv2.imread(os.path.join(frame_dir, frames[i]))
        img2 = cv2.imread(os.path.join(frame_dir, frames[i+1]))

        flow = compute_flow(img1, img2)

        flow = (flow * 255).astype("uint8")
        cv2.imwrite(os.path.join(flow_dir, f"{i:04d}.jpg"), flow)