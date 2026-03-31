import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
import os
import urllib.request
import tarfile
import time

image_size = 224
batch_size = 32
num_epochs = 10
learning_rate = 0.001
momentum = 0.9


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_dir = "./dog_data"
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

dataset_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
data_file = os.path.join(data_dir, "images.tar")

if not os.path.exists(data_file):
    print("Downloading dataset...")
    urllib.request.urlretrieve(dataset_url, data_file)
    print("Dataset downloaded.")

    print("Extracting dataset...")
    with tarfile.open(data_file, "r") as tar_ref:
        tar_ref.extractall(data_dir)
    print("Dataset extracted.")
else:
    print("Dataset already exists.")

# Image transformations with data augmentation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=15),  # Added rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Added color jitter
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load the datasets with ImageFolder
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, 'Images'), data_transforms[x]) for x in ['train', 'val']}

# Get the class names before splitting
class_names = image_datasets['train'].classes

# Split dataset into train and validation (80% train, 20% validation)
train_size = int(0.6 * len(image_datasets['train']))
val_size = len(image_datasets['train']) - train_size
image_datasets['train'], image_datasets['val'] = random_split(image_datasets['train'], [train_size, val_size])

# Create DataLoaders
dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True if x == 'train' else False, num_workers=0) for x in ['train', 'val']}  # num_workers=0 for CPU
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}


# Model Definition (ResNet152)
model = models.resnet152(pretrained=True)

# Modify the classifier part of the model
num_ftrs = model.fc.in_features  # Get the input features of the last layer
model.fc = nn.Linear(num_ftrs, len(class_names))  # Replace the last layer

model = model.to(device)

# Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

# Training Loop with timing
start_time = time.time()

for epoch in range(num_epochs):
    print(f'Epoch {epoch + 1}/{num_epochs}')
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()   # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

end_time = time.time()
execution_time = end_time - start_time
print(f"Training complete in {execution_time:.2f} seconds")

# Save the trained model
torch.save(model.state_dict(), 'dog_breed_classifier_resnet152.pth')  # Save with a different name
print('Model saved.')
