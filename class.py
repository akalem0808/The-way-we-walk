import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from sklearn.metrics import accuracy_score
import numpy as np

# Constants
dataset_dir = '/Users/amankaleem/Desktop/MDE/Quant-gsd/code/Dataset/organized_data'
img_height, img_width = 256, 256
batch_size = 32
num_epochs = 15

# OpticalFlowDataset class
class OpticalFlowDataset(Dataset):
    def __init__(self, main_dir, split='train', transform=None):
        self.main_dir = main_dir
        self.transform = transform
        self.dataset = []
        self.labels = []

        # Adjust the directory path based on the split
        split_dir = os.path.join(main_dir, split)

        # Iterate through each class directory ('away' and 'towards')
        for label_type in ['away', 'towards']:
            class_dir = os.path.join(split_dir, label_type)
            class_label = 0 if label_type == 'away' else 1
            # List all image files
            files = [os.path.join(class_dir, f) for f in sorted(os.listdir(class_dir)) if f.endswith('.jpg')]
            self.dataset.extend(files)
            self.labels.extend([class_label] * len(files))

        self.prev_frame = None

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path = self.dataset[idx]
        label = self.labels[idx]
        current_frame = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # If current_frame is None, skip this frame
        if current_frame is None:
            print(f"Warning: Skipping image {image_path}")
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)

        # Resize the current frame
        current_frame_resized = cv2.resize(current_frame, (img_height, img_width))

        # Handle the first frame case
        if self.prev_frame is None:
            self.prev_frame = current_frame_resized
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)

        # Calculate optical flow
        prev_frame_resized = cv2.resize(self.prev_frame, (img_height, img_width))
        flow = cv2.calcOpticalFlowFarneback(prev_frame_resized, current_frame_resized, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        if self.transform:
            flow = self.transform(flow)

        self.prev_frame = current_frame_resized

        # Handle last frame case by returning the current flow and label
        if idx == len(self.dataset) - 1:
            return flow, label

        return flow, label

    def reset_prev_frame(self):
        self.prev_frame = None

def collate_fn(batch):
    batch = [data for data in batch if data is not None]
    if len(batch) == 0:
        return torch.Tensor(), torch.Tensor()
    else:
        return torch.utils.data.dataloader.default_collate(batch)

# DataLoader setup
train_dataset = OpticalFlowDataset(dataset_dir, split='train')
val_dataset = OpticalFlowDataset(dataset_dir, split='val')
test_dataset = OpticalFlowDataset(dataset_dir, split='test')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

# OpticalFlowCNN model definition
class OpticalFlowCNN(nn.Module):
    def __init__(self):
        super(OpticalFlowCNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional layer
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Third convolutional layer
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 32 * 32, 128)  # Adjust the size
        self.act4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 1)
        self.act5 = nn.Sigmoid()

   

    def forward(self, x):
        x = self.pool1(self.act1(self.conv1(x)))
        x = self.pool2(self.act2(self.conv2(x)))
        x = self.pool3(self.act3(self.conv3(x)))
        # Use .reshape() instead of .view()
        x = x.reshape(x.size(0), -1)  # This will work even if x is not contiguous
        x = self.act4(self.fc1(x))
        x = self.act5(self.fc2(x))
        return x

# Model, loss function, optimizer
model = OpticalFlowCNN()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)


# Training loop
for epoch in range(num_epochs):
    model.train()
    train_dataset.reset_prev_frame()
    
    for batch in train_loader:
        flows, labels = batch
        if flows.nelement() == 0:
            continue
        # Permute the dimensions of the flow tensor
        flows = flows.permute(0, 3, 1, 2).to(torch.float32)
        labels = labels.to(torch.float32).view(-1, 1)
        flows = flows.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(flows)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    val_predictions = []
    val_labels = []
    with torch.no_grad():
        val_dataset.reset_prev_frame()
        for batch in val_loader:
            flows, labels = batch
            if flows.nelement() == 0:
                continue
            # Permute the dimensions of the flow tensor
            flows = flows.permute(0, 3, 1, 2).to(torch.float32)
            labels = labels.to(torch.float32).view(-1, 1)
            flows = flows.to(device)
            labels = labels.to(device)
            outputs = model(flows)
            val_predictions.extend(outputs.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    
    # After the validation loop, before calculating the accuracy
    val_predictions = np.concatenate(val_predictions)

    # Now val_predictions is a flat array, and you can compare it with 0.5
    val_accuracy = accuracy_score(val_labels, (val_predictions > 0.5).astype(int))
    print(f"Epoch {epoch+1}/{num_epochs}, Validation accuracy: {val_accuracy}")


# Save the model
model_save_path = '/Users/amankaleem/Desktop/MDE/Quant-gsd/code/my_model2.pth'
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)  # Create the directory if it doesn't exist
torch.save(model.state_dict(), model_save_path)
