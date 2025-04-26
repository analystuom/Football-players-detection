import cv2
import torch
import numpy as np
from ultralytics import YOLO
from torchvision.transforms import ToTensor, Resize, Normalize, Compose
import os
from tqdm import tqdm
from image_classification.players_model_ResNet50 import ResNet50
from image_classification.players_model_MyCNN import MyCNN

# input_path = "football_train/Match_1824_1_0_subclip_3/Match_1824_1_0_subclip_3.mp4"
input_path = "football_train/Match_2031_5_0_test/Match_2031_5_0_test.mp4"
output_path = "output_results"
detector_weight_path = "object_detection/best.pt"
# classifier_weight_path = "image_classification/checkpoints_mycnn/best.pt"
classifier_weight_path = "image_classification/checkpoints_resnet50/best.pt"
confidence_threshold = 0.7
num_classes = 12
device = torch.device("cuda" if torch.cuda.is_available() else "mps")

os.makedirs(output_path, exist_ok=True)

def load_models(device):
    detector = YOLO(detector_weight_path)
    detector = detector.to(device)
    detector.eval()

    classifier = ResNet50(num_classes=num_classes).to(device)
    classifier.load_state_dict(torch.load(classifier_weight_path, map_location=device))
    classifier.eval()

    return detector, classifier

def get_transform():
    transform = Compose([
        ToTensor(),
        Resize((224, 224)),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform


def main():
    detector, classifier = load_models(device)
    transform = get_transform()

    # Open input video
    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video file {input_path}")

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Define codec and set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video_path = os.path.join(output_path, "processed_video.mp4")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Processing frames
    frame_count = 0
    progress_bar = tqdm(total=total_frames, desc="Processing frames", unit="frame")

    for frame in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            raise ValueError(f"Could not read frame {frame}")

        frame_count += 1
        progress_bar.update(1)

        # Detect player in each frame
        detections = detector(frame, conf=confidence_threshold, verbose=False)[0]
        output_frame = frame.copy()

        for detection in detections.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, class_id = detection
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

            # Crop Players
            player_crop = frame[y1:y2, x1:x2]
            if player_crop.size == 0:
                continue

            # Process Player Crop
            player_crop_rgb = cv2.cvtColor(player_crop, cv2.COLOR_BGR2RGB)
            player_tensor = transform(player_crop_rgb).unsqueeze(0).to(device)

            # Classify Player
            with torch.no_grad():
                output = classifier(player_tensor)
                classifier_prediction = torch.argmax(output, dim=1)
                jersey_number = classifier_prediction.item()

            # Draw binding box and jersey number
            # Binding box
            cv2.rectangle(
                img=output_frame,
                pt1=(x1, y1),
                pt2=(x2, y2),
                color=(0, 0, 255),
                thickness=2
            )

            center_x = int((x1 + x2) / 2)

            text = f"{jersey_number}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1.2
            font_thickness = 3
            text_color = (255, 255, 255)

            (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

            text_x = center_x - (text_width // 2)
            text_y = y2 + text_height + 10

            cv2.putText(
                img=output_frame,
                text=text,
                org=(text_x, text_y),
                fontFace=font,
                fontScale=font_scale,
                color=text_color,
                thickness=font_thickness
            )

        out.write(output_frame)

    cap.release()
    out.release()

    print("Processed {} frames".format(frame_count))
    print("Video saved to {}".format(output_video_path))


if __name__ == '__main__':
    main()
