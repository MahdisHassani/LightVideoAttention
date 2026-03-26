import cv2
import torch
import numpy as np
from src.models.model import LiteUNet


def preprocess(frame):
    frame = cv2.resize(frame, (224, 224))
    frame = frame / 255.0
    frame = (frame - 0.5) / 0.5
    frame = torch.tensor(frame).permute(2, 0, 1).float()
    return frame.unsqueeze(0)


def postprocess(pred):
    pred = torch.sigmoid(pred)
    pred = pred.squeeze().cpu().numpy()

    pred = (pred * 255).astype(np.uint8)
    pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)

    return pred


def overlay(frame, heatmap):
    heatmap = cv2.resize(heatmap, (frame.shape[1], frame.shape[0]))
    return cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)


def run_demo(video_path, max_frames=300):  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LiteUNet().to(device)
    model.load_state_dict(torch.load("outputs/models/best_model.pth", map_location=device))
    model.eval()

    cap = cv2.VideoCapture(video_path)

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame).to(device)

        with torch.no_grad():
            pred = model(input_tensor)

        heatmap = postprocess(pred)
        output_frame = overlay(frame, heatmap)

        display_frame = cv2.resize(frame, (400, 300))
        display_heatmap = cv2.resize(heatmap, (400, 300))
        display_overlay = cv2.resize(output_frame, (400, 300))

        cv2.imshow("Original", display_frame)
        cv2.imshow("Heatmap", display_heatmap)
        cv2.imshow("Overlay", display_overlay)

        key = cv2.waitKey(30) & 0xFF
        if key == 27 or key == ord('q'):
            break

        frame_count += 1
        if frame_count >= max_frames:
            print("✅ Demo finished automatically")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_demo(
        video_path="data/raw/test/BandMarching/v_BandMarching_g11_c05.avi",
        max_frames=300 
    )