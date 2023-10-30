#incubator 
upstream: [[Deep Learning]]

The following walks through a python notebook that uses the [swin_t](https://pytorch.org/vision/main/models/generated/torchvision.models.swin_t.html) transformer model from pytorch and does fine tuning on the classification task [stl-10](https://cs.stanford.edu/~acoates/stl10/)

---
## 1. Import Dependencies 

```python
%%capture

# Install necessary libraries
!pip install torch torchvision matplotlib numpy

# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
```

---
## 2. Data Loading

```python 
from torchvision.datasets import STL10
  
# Update the transforms for STL10 to be compatible with Swin Transformer
transform_train = transforms.Compose([
	transforms.RandomResizedCrop(224), # Resizing to 224x224
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_test = transforms.Compose([
	transforms.Resize(232), # Resize and then crop to match Swin's requirements
	transforms.CenterCrop(224),
	transforms.ToTensor(),
	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
  

# Load STL10 dataset
trainset = torchvision.datasets.STL10(root='./data', split='train', download=True, transform=transform_train)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.STL10(root='./data', split='test', download=True, transform=transform_test)

testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

  
classes = trainset.classes # Updated to get classes from trainset
```

---
## 3. Swin Transformer Model Definition 

> see [documentation](https://pytorch.org/vision/main/models/generated/torchvision.models.swin_t.html) for more info

```python 
import torchvision.models as models

# Load the pre-trained Swin Transformer model
model = models.swin_t(weights=models.Swin_T_Weights.IMAGENET1K_V1, progress=True)

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Update the classifier head to match the number of classes in STL10
model.head = nn.Linear(model.head.in_features, len(classes)).to(device)
```

---
## 4. Optimizer 

```python 
optimizer = optim.Adam(model.parameters(), lr=0.0001) # Smaller learning rate for fine-tuning
```

---
## 5. Loss Function

```python 
criterion = nn.CrossEntropyLoss()
```

---
## 6. Training Loop 

```python 
import torch.nn.functional as F

def accuracy(outputs, labels):
	_, preds = torch.max(outputs, 1)
	return torch.tensor(torch.sum(preds == labels).item() / len(preds))

num_epochs = 8
train_loss_history, val_loss_history, train_acc_history, val_acc_history = [], [], [], []

for epoch in range(num_epochs):
	train_loss, train_acc, val_loss, val_acc = 0.0, 0.0, 0.0, 0.0

	# Training Phase
	model.train()
	for i, data in enumerate(trainloader, 0):
		inputs, labels = data
		inputs, labels = inputs.cuda(), labels.cuda()

		optimizer.zero_grad()
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		train_loss += loss.item()
		train_acc += accuracy(outputs, labels)

	# Validation Phase
	model.eval()
	with torch.no_grad():
		for i, data in enumerate(testloader, 0):
			inputs, labels = data
			inputs, labels = inputs.cuda(), labels.cuda()
			outputs = model(inputs)
			loss = criterion(outputs, labels)

			val_loss += loss.item()
			val_acc += accuracy(outputs, labels)

			train_loss /= len(trainloader)
			train_acc /= len(trainloader)
			val_loss /= len(testloader)
			val_acc /= len(testloader)

  
	train_loss_history.append(train_loss)
	train_acc_history.append(train_acc)
	val_loss_history.append(val_loss)
	val_acc_history.append(val_acc)

	print(f"Epoch {epoch+1}/{num_epochs}")
	print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
	print('-' * 60)

print('Finished Training')
```

---
## 7. Plotting

```python 
# Setting up the figure and axis
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plotting Training and Validation Losses
ax1.plot(train_loss_history, color='blue', label='Training Loss')
ax1.plot(val_loss_history, color='red', linestyle='dashed', label='Validation Loss')
ax1.set_title('Training & Validation Loss over Time')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.legend()
ax1.grid(True)

# Plotting Training and Validation Accuracies
ax2.plot(train_acc_history, color='green', label='Training Accuracy')
ax2.plot(val_acc_history, color='purple', linestyle='dashed', label='Validation Accuracy')
ax2.set_title('Training & Validation Accuracy over Time')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.legend()
ax2.grid(True)
  
# Display the plot
plt.tight_layout()

plt.show()
```

![[Swin Loss Curves.png]]
