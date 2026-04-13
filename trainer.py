import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class Trainer:
    def __init__(self, model, lr=0.01, loss_type='CrossEntropy'):
        self.model = model
        self.lr = lr
        self.optimizer = optim.SGD(model.parameters(), lr=self.lr)
        self.set_loss(loss_type)
        
        self.loss_history = []
        self.weight_history = [self.model.get_weights()]
        self.delta_history = []
        self.global_max_delta = 0.0
        
    def set_loss(self, loss_type):
        if loss_type == 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_type == 'MSE':
            self.criterion = nn.MSELoss()
        elif loss_type == 'MAE':
            self.criterion = nn.L1Loss()
        else:
            self.criterion = nn.CrossEntropyLoss()
            
    def step(self, X_batch, y_batch):
        self.model.train()
        self.optimizer.zero_grad()
        
        X_batch = torch.tensor(X_batch, dtype=torch.float32)
        
        if isinstance(self.criterion, nn.CrossEntropyLoss):
            y_batch = torch.tensor(y_batch, dtype=torch.long)
        else:
            # Output matching for MSE / MAE
            if len(y_batch.shape) == 1:
                y_batch = torch.tensor(y_batch, dtype=torch.long)
                y_batch = nn.functional.one_hot(y_batch, num_classes=self.model.architecture[-1]).float()
            else:
                y_batch = torch.tensor(y_batch, dtype=torch.float32)

        outputs = self.model(X_batch, record_activations=True)
        loss = self.criterion(outputs, y_batch)
        loss.backward()
        self.optimizer.step()
        
        self.loss_history.append(loss.item())
        
        current_weights = self.model.get_weights()
        prev_weights = self.weight_history[-1]
        
        # Calculate positive deltas (abs difference)
        deltas = [np.abs(cw - pw) for cw, pw in zip(current_weights, prev_weights)]
        self.delta_history.append(deltas)
        
        # Keep last 10 deltas for sliding window
        if len(self.delta_history) > 10:
            self.delta_history.pop(0)

        # average delta across the window
        avg_deltas = []
        if len(self.delta_history) > 0:
            for l_idx in range(len(deltas)):
                layer_avg = np.mean([d[l_idx] for d in self.delta_history], axis=0)
                avg_deltas.append(layer_avg)
                
                # trace maximum historical delta for fading visualization
                self.global_max_delta = max(self.global_max_delta, float(np.max(layer_avg)))
                
        self.weight_history.append(current_weights)
        if len(self.weight_history) > 2:
            self.weight_history.pop(0)
            
        return loss.item(), current_weights, avg_deltas
