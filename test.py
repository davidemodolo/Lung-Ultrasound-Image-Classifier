import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image

# Define the transforms for image normalization
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ImageDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.image_list[idx])
        image = Image.open(img_name)
        image = transform(image)
        # Extract label from the image filename
        label = int(img_name.split("_")[-1].split(".")[0])
        return image, label

def train_model(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc

def test_model(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = running_corrects.double() / len(dataloader.dataset)
    return epoch_loss, epoch_acc

def kfold_cross_validation(data_dir, k=5, num_epochs=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 4) # 4 possible score classes
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    image_list = os.listdir(data_dir)
    num_images = len(image_list)
    fold_size = num_images // k
    indices = list(range(num_images))
    np.random.shuffle(indices)
    kfold_results = []
    for i in range(k):
        print(f"Fold {i+1}")
        start_idx = i * fold_size
        end_idx = min((i+1) * fold_size, num_images)
        val_indices = indices[start_idx:end_idx]
        train_indices = list(set(indices) - set(val_indices))
        train_dir = [image_list[j] for j in train_indices]
        val_dir = [image_list[j] for j in val_indices]
        train_dataset = ImageDataset(os.path.join(data_dir, train_dir))
        val_dataset = ImageDataset(os.path.join(data_dir, val_dir))
        train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}")
            train_loss, train_acc = train_model(model, train_dataloader, optimizer, criterion, device)
            print(f"Training Loss: {train_loss:.4f} Training Accuracy: {train_acc:.4f}")
            val_loss, val_acc = test_model(model, val_dataloader, criterion, device)
            print(f"Validation Loss: {val_loss:.4f} Validation Accuracy: {val_acc:.4f}")
        kfold_results.append((train_loss, train_acc, val_loss, val_acc))
    return kfold_results

data_dir = "images"
k = 5
num_epochs = 1
learning_rate = 0.001

kfold_results = kfold_cross_validation(data_dir, k, num_epochs, learning_rate)

train_losses = []
train_accs = []
val_losses = []
val_accs = []

for train_loss, train_acc, val_loss, val_acc in kfold_results:
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

print("Training Loss:", sum(train_losses)/len(train_losses))
print("Training Accuracy:", sum(train_accs)/len(train_accs))
print("Validation Loss:", sum(val_losses)/len(val_losses))
print("Validation Accuracy:", sum(val_accs)/len(val_accs))

