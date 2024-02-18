import torch
import torch.nn as tornn
class PyTorchGRU(tornn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PyTorchGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.gru = tornn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = tornn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :]) 
        return out
