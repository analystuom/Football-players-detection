import os
import time
from torch.utils.tensorboard import SummaryWriter
from players_dataset import PlayersDataset

from players_model_ResNet50 import ResNet50
from players_model_MyCNN import MyCNN


from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from tqdm.autonotebook import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from torchvision.transforms import (
    ToTensor, Resize, Normalize, Compose, Lambda
)
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import ImageOps

def custom_resize(image):
    return ImageOps.pad(image, (224, 224), color=(0, 0, 0))

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    root = "player_images"
    batch_size = 64
    num_epochs = 50
    learning_rate = 1e-3
    weight_decay = 5e-4
    num_classes = 12
    patience = 5

    log_path = "tensorboard_resnet50"
    checkpoint_path = "checkpoints_resnet50"

    # log_path = "tensorboard_mycnn"
    # checkpoint_path = "checkpoints_mycnn"

    if not os.path.exists(log_path):
        os.mkdir(log_path)
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    # Would use normalize if using ResNet50 for Transfer Learning
    transform = Compose([
        Lambda(custom_resize),
        ToTensor(),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = PlayersDataset(
        root=root,
        train=True,
        transform=transform
    )

    val_dataset = PlayersDataset(
        root=root,
        train=False,
        transform=transform
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4,
        drop_last=True
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=4,
        drop_last=False
    )

    # Initiate ResNet50 or MyCNN here
    model = ResNet50(num_classes=num_classes).to(device)
    # model = MyCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=patience, verbose=True
    )

    writer = SummaryWriter(log_dir=log_path)

    best_acc = 0
    no_improve_epochs = 0

    # Start training the model
    for epoch in range(num_epochs):
        start_time = time.time()
        # Training stage
        model.train()
        train_losses = []
        train_progress_bar = tqdm(train_loader, desc=f"Epoch: {epoch + 1}/{num_epochs} [Train]")

        # Forward Pass
        for iter, (images, labels) in enumerate(train_progress_bar):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{np.mean(train_losses):.4f}"
            })

            avg_train_loss =np.mean(train_losses)

            # Log training metrics
            global_step = epoch * len(train_loader) + iter
            writer.add_scalar(tag="Train/Loss",
                              scalar_value=loss.item(), global_step=global_step)
            writer.add_scalar(tag="Train/LearningRate",
                              scalar_value=optimizer.param_groups[0]['lr'], global_step=global_step)

        # Validation Stage
        model.eval()
        val_losses = []
        val_predictions = []
        val_labels = []
        val_progress_bar = tqdm(val_loader, desc=f"Epoch: {epoch + 1}/{num_epochs} [Val]", colour="yellow")

        with torch.no_grad():
            for images, labels in val_progress_bar:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                prediction = torch.argmax(outputs, dim=1)

                val_losses.append(loss.item())
                val_predictions.extend(prediction.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())

                val_progress_bar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'avg_loss': f"{np.mean(val_losses):.4f}"
                })

        # Validation Metrics
        avg_val_loss = np.mean(val_losses)
        accuracy = accuracy_score(val_labels, val_predictions)
        print("Average loss: {}. Accuracy: {}".format(avg_val_loss, accuracy))

        # Log Validation metrics
        writer.add_scalar(tag="Val/Loss", scalar_value=avg_val_loss, global_step=epoch)
        writer.add_scalar(tag="Val/Accuracy", scalar_value=accuracy, global_step=epoch)

        scheduler.step(accuracy)

        torch.save(model.state_dict(), os.path.join(checkpoint_path, "last.pt"))
        if accuracy > best_acc:
            best_acc = accuracy
            no_improve_epochs = 0
            torch.save(model.state_dict(), os.path.join(checkpoint_path, "best.pt"))

            # Generate and save confusion matrix
            cm = confusion_matrix(val_labels, val_predictions)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues")
            plt.title(f"Confusion matrix - Epoch {epoch + 1}")
            plt.ylabel("True label")
            plt.xlabel("Predicted label")
            plt.savefig(os.path.join(checkpoint_path, f'confusion_matrix_epoch_{epoch + 1}.png'))
            plt.close()
        else:
            no_improve_epochs += 1

        # Epoch Summary:
        epoch_time = time.time() - start_time
        print("Epoch {}/{} completed in {}".format(epoch + 1, num_epochs, epoch_time))
        print("Train Loss: {:.4f}. Val Loss: {:.4f}".format(avg_train_loss, avg_val_loss))
        print("Accuracy: {:.4f}".format(accuracy))
        print("Best Accuracy: {:.4f}. No Improvement Epoch: {}".format(best_acc, no_improve_epochs))

        # Early stopping trigger:
        if no_improve_epochs >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    writer.close()
    print("Training completed with best accuracy: {:.4f}".format(best_acc))


if __name__ == '__main__':
    train()
