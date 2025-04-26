import cv2
import os
import json

def get_total_frames(root):
    cap = cv2.VideoCapture(root)

    if not cap.isOpened():
        print(f"Cannot open video {root}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total_frames


def extract_frame(video, frame_index,dataset_type):
    root = f"./football_train/{video}/{video}.mp4"
    cap = cv2.VideoCapture(root)

    if not cap.isOpened():
        print(f"Cannot open video {root}")
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_index < 0 or frame_index >= total_frames:
        print(f"Frame {frame_index} out of range")
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()

    if not ret:
        print(f"Cannot read video at frame {frame_index}")
        cap.release()
        return None

    cap.release()

    # save the image into corresponding file
    if dataset_type == "test":
        output_dir = "data/images/test"
    elif dataset_type == "validation":
        output_dir = "data/images/val"
    else:
        output_dir = "../data/images/train"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video}_{frame_index}.jpg")
    cv2.imwrite(output_path, frame)

    return frame


# Finding annotation list of corresponding image with category_id = 4
def get_label_data(label_root):
    with open(label_root, "r") as f:
        data = json.load(f)

    list_images = data["images"]
    list_annotations = data["annotations"]
    width = list_images[0]["width"]
    height = list_images[0]["height"]

    return list_images, list_annotations, width, height


def get_annotations(image_id, list_annotations):
    annotations = []
    for annot in list_annotations:
        if annot["image_id"] == image_id:
            annotations.append(annot)
    return annotations


def extract_labels(category_id, list_annotations, image_id, width, height, video, dataset_type):
    annotations = get_annotations(image_id, list_annotations)
    results = []
    for annot in annotations:
        if annot["category_id"] == category_id:
            x_min, y_min, bbox_width, bbox_height = annot["bbox"]
            x_center = (x_min + bbox_width / 2) / width
            y_center = (y_min + bbox_height / 2) / height
            norm_width = bbox_width / width
            norm_height = bbox_height / height

            results.append((category_id, x_center, y_center, norm_width, norm_height))
            print(
                f"Player detected in YOLO format: "
                f"{category_id}, {x_center:.6f}, {y_center:.6f}, {norm_width:.6f}, {norm_height:.6f}")

    # Write results into txt file
    if dataset_type == "test":
        output_dir = "data/labels/test"
    elif dataset_type == "validation":
        output_dir = "../data/labels/val"
    else:
        output_dir = "../data/labels/train"

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video}_{image_id - 1}.txt")
    with open(output_path, "w") as f:
        for result in results:
            category_id, x_center, y_center, norm_width, norm_height = result
            f.write(f"{category_id - 4} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}\n")

    return results

def prepare_dataset(videos, dataset_type):
    for video in videos:
        root = f"./football_train/{video}/{video}.mp4"
        label_root = f"./football_train/{video}/{video}.json"
        (list_images, list_annotations, width, height) = get_label_data(label_root)

        for i in range(len(list_images)):
            extract_frame(video, frame_index=i, dataset_type="train")
            extract_labels(
                category_id=4,
                image_id=i + 1,
                list_annotations=list_annotations,
                width=width,
                height=height,
                video=video,
                dataset_type=dataset_type)

if __name__ == '__main__':

    # Full extraction of data
    video_files_train = ["Match_1824_1_0_subclip_3", "Match_1951_1_0_subclip", "Match_2022_3_0_subclip"]
    video_files_val = ["Match_2023_3_0_subclip"]
    
    datasets = {
        "train": video_files_train,
        "validation": video_files_val
    }
    
    # Prepare the data
    for dataset_type, videos in datasets.items():
        prepare_dataset(videos, dataset_type)

    # video_files_train = ["Match_1824_1_0_subclip_3", "Match_1951_1_0_subclip", "Match_2022_3_0_subclip"]
    # video_files_val = ["Match_2023_3_0_subclip"]

    # # Prepare train/validation data - for testing purpose
    # for video in video_files_train:
    #     root = f"./football_train/{video}/{video}.mp4"
    #     label_root = f"./football_train/{video}/{video}.json"
    #     (list_images, list_annotations, width, height) = get_label_data(label_root)
    #     for i in range(len(list_images)):
    #         extract_frame(video, frame_index=i, dataset_type="train")
    #         extract_labels(category_id=4, image_id=i + 1, list_annotations=list_annotations,
    #                        width=width, height=height, video=video, dataset_type="train")

    # for video in video_files_val:
    #     root = f"./football_train/{video}/{video}.mp4"
    #     label_root = f"./football_train/{video}/{video}.json"
    #     (list_images, list_annotations, width, height) = get_label_data(label_root)
    #     for i in range(len(list_images)):
    #         extract_frame(video, frame_index=i, dataset_type="validation")
    #         extract_labels(category_id=4, image_id=i + 1, list_annotations=list_annotations,
    #                        width=width, height=height, video=video, dataset_type="validation")

