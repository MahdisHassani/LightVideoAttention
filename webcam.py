import cv2
import torch
from src.models.model import LiteUNet
from predict_video import preprocess, postprocess, overlay


def run_webcam():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LiteUNet().to(device)
    model.load_state_dict(torch.load("outputs/models/best_model.pth", map_location=device))
    model.eval()

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        input_tensor = preprocess(frame).to(device)

        with torch.no_grad():
            pred = model(input_tensor)

        heatmap = postprocess(pred)
        output_frame = overlay(frame, heatmap)
        
        cv2.imshow("Webcam", output_frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_webcam()