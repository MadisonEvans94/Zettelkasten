#seed 
upstream: [[Deep Learning]]

---

**video links**: 

---

# Brain Dump: 


--- 


main.py 

```python 
# Import necessary modules
from MyDataset import MyDataset
from MyModel import MyModel
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import yaml

# Read configuration file
try:
    with open("config.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
except yaml.YAMLError as exc:
    print(exc)

# Extract hyperparameters from config file
learning_rate = config.get('learning_rate', 0.001)
batch_size = config.get('batch_size', 32)
num_epochs = config.get('num_epochs', 10)
root_dir = config.get('root_dir', './data')
architecture = config.get('architecture', [])

# Initialize transformations and datasets
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

train_dataset = MyDataset(root_dir=root_dir, transform=transform, train=True)
test_dataset = MyDataset(root_dir=root_dir, transform=transform, train=False)

# Initialize DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = MyModel(num_classes=4, input_height=256, input_width=256)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

# Move model to appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training and evaluation code
def main():
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

        # Evaluation code can be added here

if __name__ == "__main__":
    main()

```

config.yaml: 

```yaml
# General Configuration
# model_name: 'CustomModel'
# experiment_name: 'Experiment1'
# timestamp: False  # Whether to append a timestamp to the experiment name

# Data Loader Configuration
# root_dir: './data'
# batch_size: 64
# num_workers: 4  # Number of data loading threads
# shuffle: True

# Image Preprocessing
# img_height: 256
# img_width: 256
# normalize_mean: [0.485, 0.456, 0.406]
# normalize_std: [0.229, 0.224, 0.225]

# Model Architecture Configuration
# num_classes: 33
# depth: 18  # For architectures like ResNet

# Training Configuration
# num_epochs: 50
# learning_rate: 0.001
# lr_decay: 0.95  # Learning rate decay factor
# lr_decay_step: 10  # Decay the learning rate every N epochs
# momentum: 0.9
# weight_decay: 0.0005  # L2 regularization

# Loss Function
# loss_function: 'CrossEntropy'  # Could be 'MSE', 'CrossEntropy', 'Huber', etc.

# Optimizer
# optimizer: 'SGD'  # Could be 'Adam', 'SGD', etc.

# Scheduler (Optional)
# scheduler: 'StepLR'  # Could be 'StepLR', 'MultiStepLR', 'CosineAnnealingLR', etc.
# scheduler_gamma: 0.1  # Decrease lr by a factor of gamma
# scheduler_step_size: 30  # Decrease lr every N epochs

# Early Stopping (Optional)
# early_stopping: True
# patience: 10  # Number of epochs with no improvement to wait before stopping

# Checkpointing (Optional)
# save_best_model: True
# checkpoint_dir: './checkpoints'
# save_frequency: 10  # Save a checkpoint every N epochs

# TensorBoard (Optional)
# use_tensorboard: True
# tensorboard_dir: './tensorboard_logs'

# Miscellaneous
# seed: 42  # for reproducibility
# debug_mode: False  # If true, load less data
```

Dataset Class: 

```python
import os
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

        # List classes (assuming each class has its own folder)
        self.classes = os.listdir(self.root_dir)
        
        self.image_paths = []
        self.labels = []

        # Populate image paths and labels
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            image_files = glob.glob(os.path.join(class_dir, "*.jpg"))
            
            labels = [class_idx] * len(image_files)
            self.image_paths.extend(image_files)
            self.labels.extend(labels)

        # Train-test split
        self.train_image_paths, self.test_image_paths, self.train_labels, self.test_labels = train_test_split(
            self.image_paths, self.labels, test_size=0.2, random_state=42, stratify=self.labels
        )

        self.image_paths = self.train_image_paths if self.train else self.test_image_paths
        self.labels = self.train_labels if self.train else self.test_labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        with Image.open(img_path) as img:
            if self.transform:
                img = self.transform(img)
        
        return img, label

```

required setup: 
```
root_dir/
│
├── class_1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
├── class_2/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
├── class_3/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
│
└── ...
```

Convolutional Model:  

```python 
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):

	def __init__(self, num_classes=None):
	
		super(MyModel, self).__init__()
		# Feature extractor
		self.features = nn.Sequential(
			nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),	
			nn.ReLU(inplace=True),	
			nn.MaxPool2d(kernel_size=2, stride=2),		  	
			nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),	
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2),
			nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2)
		)
		
		with torch.no_grad():
			# [batch, channels, height, width]
			dummy_x = torch.zeros(1, 3, input_height, input_width)
			dummy_x = self.features(dummy_x)
			# Flatten the tensor and get the length
			num_flat_features = dummy_x.view(-1).shape[0]

  

		# Classifier
		
		self.classifier = nn.Sequential(
			nn.Linear(num_flat_features, 128),
			nn.ReLU(inplace=True),
			nn.Linear(128, num_classes)
	
	def forward(self, x):
		x = self.features(x)
		x = torch.flatten(x, 1) # Flatten the tensor along dimension 1
		x = self.classifier(x)
		return F.log_softmax(x, dim=1)
```