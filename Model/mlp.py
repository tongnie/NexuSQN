import torch
from torch import nn


class MultiLayerPerceptron(nn.Module):
    """
    Multi-Layer Perceptron with residual connections
    """
    def __init__(self, input_dim, hidden_dim, p=0.15) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=p)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        """Feed forward of MLP.
       Args:
           input_data (torch.Tensor): input data with shape [B, N, C]
       Returns:
           torch.Tensor: latent repr
       """

        hidden = self.fc2(self.drop(self.act(self.fc1(input_data))))      # MLP
        hidden = hidden + input_data                           # residual

        return hidden