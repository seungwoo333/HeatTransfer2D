import torch
from torch import nn


class BranchNet_RGNN(nn.Module):
    def __init__(self, adj, num_RNN, hidden_size, in_feature, out_feature):
        super().__init__()
        self.num_RNN = num_RNN
        self.hidden_size = hidden_size

        self.adj = nn.Parameter(adj, requires_grad=False)
        self.adj_weight = nn.Parameter(torch.randn_like(adj), requires_grad=True)
        self.RNN = nn.RNN(input_size = in_feature, hidden_size = hidden_size, num_layers = num_RNN, batch_first = True)
        self.fc = nn.Linear(hidden_size, out_feature, bias=True)
        self.relaxation = nn.Parameter(torch.tensor([0.001]), requires_grad=True)

    def forward(self, x, hidden):
        out = x @ (self.adj * self.adj_weight)
        out, hidden = self.RNN(out, hidden)
        out = self.fc(out)
        out = self.relaxation * out + (1-self.relaxation) * x
        return out, hidden, self.relaxation

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_RNN, batch_size, self.hidden_size)