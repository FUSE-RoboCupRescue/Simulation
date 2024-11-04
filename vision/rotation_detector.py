import os
import cv2
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from models.mlp import MLP
from trainers.standard import train
from ladolt_generator import rotated_rectangle


IMAGE_RESIZE = 64
device = "mps"

def images_to_np_array(path: str, image_dim: int=256):
    data = []
    for i in range(len(os.listdir(path))-1):
        image = cv2.imread(f"{path}/{i}.png")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (image_dim, image_dim))
        image = image/255.0
        image = image.flatten()
        data.append(image)

    data = np.array(data)
    return data


# images = images_to_np_array("../data/simple_ladolts", IMAGE_RESIZE)
# rotations = json.load(open("../data/simple_ladolts/rotations.json"))
images = images_to_np_array("./datasets/cropped", IMAGE_RESIZE)
rotations = json.load(open("./datasets/cropped/rotations.json"))
rotations = np.array([np.array([np.cos(rot), np.sin(rot)]) for rot in rotations])

X_train, X_test, y_train, y_test = train_test_split(images, rotations, test_size=0.2, random_state=42)

X_train_tensor = torch.tensor(X_train, dtype=torch.float)
y_train_tensor = torch.tensor(y_train, dtype=torch.float)
X_val_tensor = torch.tensor(X_test, dtype=torch.float)
y_val_tensor = torch.tensor(y_test, dtype=torch.float)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=True)

model = MLP(IMAGE_RESIZE**2, 2, hidden_layers=[1024, 1024, 1024])
model = train(model, device, train_loader, val_loader, 100, lr=1e-4)

example_count = 6
rand_indices = np.random.choice(list(range(len(X_test))), size=example_count)
images = X_test[rand_indices]
vector = model(torch.Tensor(images).to(device)).cpu().detach().numpy()

grid_dims = [2, 2]
fig, axes = plt.subplots(grid_dims[1], grid_dims[0], figsize=(7, 7))
for i in range(grid_dims[1]):
    for j in range(grid_dims[0]):

        image = images[i*grid_dims[0]+j]
        image = np.array([[c*255, c*255, c*255] for c in image], dtype=np.uint8)
        image = image.reshape(IMAGE_RESIZE, IMAGE_RESIZE, 3)

        start = np.array([IMAGE_RESIZE*.5, IMAGE_RESIZE*.5])
        end = vector[i*grid_dims[0]+j] * IMAGE_RESIZE*.4 + start
        start = (int(start[0]), int(start[1]))
        end = (int(end[0]), int(end[1]))
        cv2.line(image, start, end, (0, 255, 0), 2)

        axes[i, j].imshow(image)
        axes[i, j].axis('off')
plt.tight_layout()
plt.show()
