import os
import cv2
import json
import tqdm
import random
import shutil


def crop_players(root, output):
    # Create the directories
    os.makedirs(output, exist_ok=True)
    for path in ["initial", "train", "val"]:
        os.makedirs(os.path.join(output, path), exist_ok=True)

    sub_folder_paths = [os.path.join(output, f"initial/{i}") for i in range(0, 12)]
    for path in sub_folder_paths:
        os.makedirs(path, exist_ok=True)

    # Opening and process video files
    data_subdirectories = [i for i in os.listdir(root) if os.path.isdir(os.path.join(root, i))]
    for dir in data_subdirectories:
        print("Processing {}".format(dir))

        subdir_path = os.path.join(root, dir)
        video_path = os.path.join(subdir_path, f"{dir}.mp4")
        annotation_path = os.path.join(subdir_path, f"{dir}.json")

        # Read annotation json file
        with open(annotation_path, "r") as f:
            json_file = json.load(f)

        # Open video and read frames
        cap = cv2.VideoCapture(video_path)
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

            current_frame_annotations = [annotation for annotation in json_file["annotations"]
                                         if annotation['image_id'] == frame_count and annotation['category_id'] == 4]

            # Process each frame and crop the image
            for annotation in current_frame_annotations:
                x_min, y_min, w, h = annotation['bbox']
                x_max = x_min + w
                y_max = y_min + h
                player_image = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

                if player_image.size > 0:
                    player_jersey = 0
                    if annotation['attributes']:
                        num_visibility = annotation['attributes'].get('number_visible')
                        if num_visibility in ['invisible']:
                            player_jersey = 0
                        else:
                            player_jersey = int(annotation["attributes"]["jersey_number"])
                            if 1 <= player_jersey <= 10:
                                player_jersey = player_jersey
                            elif player_jersey >= 11:
                                player_jersey = 11

                    jersey_color = annotation['attributes'].get('team_jersey_color')
                    class_dir = os.path.join(output, "initial", str(player_jersey))
                    image_name = (f"Video_{dir}_frame_{frame_count}_jersey_player_{player_jersey}"
                                  f"_jersey_color_{jersey_color}.jpg")
                    cv2.imwrite(os.path.join(class_dir, image_name), player_image)


        print("Total frames: {}".format(frame_count))
        cap.release()


def train_val_split(initial_path, split_ratio):
    for label in range(12):
        player_dir = os.path.join(initial_path, str(label))
        images = os.listdir(player_dir)
        random.shuffle(images)

        temp = int(len(images) * split_ratio)
        train_images = images[:temp]
        val_images = images[temp:]

        # Save images to corresponding folders
        train_path = os.path.join("player_images/train", str(label))
        os.makedirs(train_path, exist_ok=True)
        for image in train_images:
            current_image_path = os.path.join(player_dir, image)
            target_image_path = os.path.join(train_path, image)
            shutil.copy(current_image_path, target_image_path)

        # Save images to val folder
        val_path = os.path.join("player_images/val", str(label))
        os.makedirs(val_path, exist_ok=True)
        for image in val_images:
            current_image_path = os.path.join(player_dir, image)
            target_image_path = os.path.join(val_path, image)
            shutil.copy(current_image_path, target_image_path)


if __name__ == '__main__':
    root = "football_train"
    output = "player_images"
    initial_path = "player_images/initial"

    crop_players(root, output)
    train_val_split(initial_path, 0.8)
