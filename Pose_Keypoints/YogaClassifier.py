import torch.nn as nn

class YogaClassifier(nn.Module):
    def __init__(self, input_length, num_classes, hidden_dim, dropout):
        super().__init__()
        self.layer1 = nn.Linear(input_length, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.act = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.act(self.layer1(x))
        x = self.act(self.norm1(x))
        x = self.dropout(x)
        x = self.act(self.layer2(x))
        x = self.act(self.norm2(x))
        x = self.out(x)
        return x