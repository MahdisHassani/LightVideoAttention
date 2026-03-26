import cv2
import torch
import numpy as np
import imageio
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
    overlayed = cv2.addWeighted(frame, 0.6, heatmap, 0.4, 0)
    return overlayed


def run_inference(video_path, output_path, save_gif=False):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LiteUNet().to(device)
    model.load_state_dict(torch.load("outputs/models/best_model.pth", map_location=device))
    model.eval()

    cap = cv2.VideoCapture(video_path)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, 20.0,
                          (int(cap.get(3)), int(cap.get(4))))

    gif_frames = []  

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame).to(device)

        with torch.no_grad():
            pred = model(input_tensor)

        heatmap = postprocess(pred)
        output_frame = overlay(frame, heatmap)

        out.write(output_frame)

        if save_gif:
            gif_frame = cv2.cvtColor(output_frame, cv2.COLOR_BGR2RGB)
            gif_frames.append(gif_frame)

        cv2.imshow("Output", output_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    if save_gif:
        imageio.mimsave("outputs/videos/demo.gif", gif_frames, fps=10)
        print("✅ GIF saved!")
        
        
run_inference(
    video_path="data/raw/test/Basketball/v_Basketball_g06_c02.avi",
    output_path="outputs/videos/output.avi",
    save_gif=True)