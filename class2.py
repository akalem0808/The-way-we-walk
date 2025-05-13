import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
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
        
        split_dir = os.path.join(main_dir, split)
        for label_type in ['far', 'medium', 'away']:
            class_dir = os.path.join(split_dir, label_type)
            class_label = ['far', 'medium', 'away'].index(label_type)
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

        if current_frame is None:
            print(f"Warning: Skipping image {image_path}")
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)

        current_frame_resized = cv2.resize(current_frame, (img_height, img_width))

        if self.prev_frame is None:
            self.prev_frame = current_frame_resized
            next_idx = (idx + 1) % len(self)
            return self.__getitem__(next_idx)

        prev_frame_resized = cv2.resize(self.prev_frame, (img_height, img_width))
        flow = cv2.calcOpticalFlowFarneback(prev_frame_resized, current_frame_resized, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        if self.transform:
            flow = self.transform(flow)

        self.prev_frame = current_frame_resized

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
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1)
        self.act1 = nn.ReLU()
