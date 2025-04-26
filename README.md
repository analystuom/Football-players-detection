## 📍 About this Project
This project aims to develop an AI Pipeline that can detect players and their jersey number within a video. Particularly, There are two main componenents within the pipeline: An object detection model for detecting players in the field and an image clasisfication model for classifying the jersey number.

![image](https://github.com/user-attachments/assets/bb6f855c-5cdc-4038-a767-96ddafff64ca)

### Object Detection Model
This is the first model in the pipeline. It takes a single frame within a video as the input and detecting the position of the players

### Image Classfication Model
After detecting the position of each players, the second model, which is a classifier will predict the jersey number of the corresponding players. It takes the crop image of the players as the input.
There are two classification models that were trained in this section

### Project Structure

```sh

├── football_train                  # The football match videos and their annotation files
├── image_classification
│   ├── .idea
│   ├── checkpoints_mycnn
│   │   ├── best.pt                 # Best Image Classification Model, the one being used in the prediction pipeline
│   │   ├── last.pt  
│   │   ├── confusion_matrixes      # The confusion matrix showing the prediction results
│   ├── checkpoints_resnet50
│   │   ├── best.pt                 # Best Image Classification Model, the one being used in the prediction pipeline
│   │   ├── last.pt  
│   │   ├── confusion_matrixes      # The confusion matrix showing the prediction results
│   ├── player_images               # The datasets used for training classification model
│   │   ├── train                 
│   │   ├── val                   
│   ├── tensorboard_mycnn 
│   ├── tensorboard_mycnn                 
│   ├── players_dataset.py          # Instantiate the DataLoader for the model
│   ├── players_model_MyCNN.py      # Model architecture - MyCNN
│   ├── players_model_ResNet50.py   # Model architecture - MyCNN
│   ├── players_train.py            # Training classification model
│   └── prepare_data.py             # Prepare the datasets for image classification model
├── object_detection
│   ├── data                        # The datas in COCO format, used for training object detection model        
│   │   ├── images
│   │   │     ├── train
│   │   │     ├── val
│   │   ├── labels
│   │   │     ├── train
│   │   │     ├── val
│   │   ├── data.yaml
├── output_results               # The videos produced by the prediction pipeline, annotating players with binding boxes and jersey number in text format
└── prediction_pipeline.py       # The main script used for assembling models into a prediction pipeline, running this script will produce the output results

```
