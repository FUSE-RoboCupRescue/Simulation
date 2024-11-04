import os
import json
import torch
import torch.nn as nn
from datetime import datetime


class MLP(nn.Module):
    def __init__(self,
                 input_size: int,
                 output_size: int,
                 dropout_prob: float = 0.2,
                 activation: str = 'relu',
                 hidden_layers: list[int] = [16, 16]):
        
        super(MLP, self).__init__()

        # Initialize encoder layers
        layers = []
        last_dim = input_size
        for size in hidden_layers:
            layers.append(nn.Linear(last_dim, size))
            last_dim = size
            
            # Add activation function
            if activation.lower() == 'relu':
                layers.append(nn.ReLU())
            else:
                raise ValueError(f"Error: Activation function '{activation}' not implemented")
            
            # Add dropout layer
            layers.append(nn.Dropout(dropout_prob))

        layers.append(nn.Linear(last_dim, output_size))

        self.layers = nn.Sequential(*layers)

        self.generate_log_data_path()

        model_params = {
            'input_size': input_size,
            'output_size': output_size,
            'hidden_layers': hidden_layers,
            'activation': activation,
            'dropout_prob': dropout_prob
        }

        with open(f'{self.path}/model_params.json', 'w', encoding='utf-8') as f:
            json.dump(model_params, f)

    def generate_log_data_path(self):
        current_time = datetime.now().strftime("%Y-%m-%d_%H:%M")
        self.path = f"./logs/models/mlp_{current_time}"
        if not os.path.exists(f"{self.path}/pth"):
            os.makedirs(f"{self.path}/pth")
    
    def forward(self, x):
        x = self.layers(x)
        return x
    
    def save(self, epoch: int):
        file_name = f"{self.path}/pth/epoch_{epoch}.pth"
        torch.save(self.state_dict(), file_name)
    
