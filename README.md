## 📍 About this Project
This project aims to develop an AI Pipeline that can detect players and their jersey number within a video. Particularly, There are two main componenents within the pipeline: An object detection model for detecting players in the field and an image clasisfication model for classifying the jersey number.

### Object Detection Model
This is the first model in the pipeline. It takes a single frame within a video as the input and detecting the position of the players. Following is the result of the object detector
![results](https://github.com/user-attachments/assets/5f513efb-bfc9-44b1-9f85-ddeba95dc2db)


### Image Classfication Model
After detecting the position of each players, the second model, which is a classifier will predict the jersey number of the corresponding players. It takes the crop image of the players as the input.
There are two classification models that were trained in this section: One using a customed CNN architecture: MyCNN and one model using transfer learning setup with ResNet50.

In general, the first model outperformed the second one with 97% accuracy while the second one only achived 75% accuracy

#### Results of the first model using transfer learning setup with MyCNN
Train Loss:
<img width="1318" alt="image" src="https://github.com/user-attachments/assets/cfe6292b-91d6-4ec0-8b58-0639b3cbd684" />

Val Accuracy and Val Loss:
</br>
![image](https://github.com/user-attachments/assets/696ffe1a-c09b-4269-a0ef-e87be1539681)

Confusion Matrix:
![image](https://github.com/user-attachments/assets/ac6ee849-8e76-4828-b8b0-5e251a1c868f)


#### Results of the first model using transfer learning setup with ResNet50
Train Loss:
![image](https://github.com/user-attachments/assets/99ff06c1-a2e4-42bc-a9fa-aa6e3fa75cd3)

Val Accuracy and Val Loss:
</br>
![image](https://github.com/user-attachments/assets/bbc0f0b3-b9bc-4cbf-9e01-4e6fc65738c1)

Confusion Matrix:
![image](https://github.com/user-attachments/assets/4a37245d-8c2c-4d73-82b3-152d3b9a2b36)


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
│   ├── players_model_ResNet50.py   # Model architecture - ResNet50
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
