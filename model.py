import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, architecture):
        super(MLP, self).__init__()
        self.architecture = architecture
        self.layers = nn.ModuleList()
        # Create linear layers
        for i in range(len(architecture) - 1):
            self.layers.append(nn.Linear(architecture[i], architecture[i+1]))
            
        self.last_activations = []
        
    def forward(self, x, record_activations=False):
        activations = []
        if record_activations:
            activations.append(x.detach().squeeze().numpy())
            
        out = x
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < len(self.layers) - 1:
                out = torch.relu(out)
            if record_activations:
                # We save the output of each layer
                activations.append(out.detach().squeeze().numpy())
                
        if record_activations:
            self.last_activations = activations
            
        return out

    def get_weights(self):
        # returns list of weight matrices as numpy arrays: (out_features, in_features)
        weights = []
        for layer in self.layers:
            weights.append(layer.weight.detach().numpy().copy())
        return weights
