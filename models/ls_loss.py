import torch
import torch.nn as nn

class LS_Loss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.3, gamma=0.0):
        super(LS_Loss, self).__init__()
        self.alpha = alpha  # Weight for Short-term Loss
        self.beta = beta    # Weight for Long-term Loss
        self.gamma = gamma  # Weight for Cross-Modality Consistency Loss
        self.mse_loss = nn.MSELoss()

    def forward(self, output_long, target_long, output_short, target_short):
        # Short-term Loss
        loss_short = self.mse_loss(output_short, target_short)

        # Long-term Loss
        loss_long = self.mse_loss(output_long, target_long)

        # Optional: Cross-Modality Consistency Loss
        consistency_loss = 0.0
        if self.gamma > 0:
            long_mean = torch.mean(output_long, dim=1, keepdim=True)
            short_mean = torch.mean(output_short, dim=1, keepdim=True)
            consistency_loss = torch.mean((long_mean - short_mean) ** 2)

        # Combined Loss
        total_loss = self.alpha * loss_short + self.beta * loss_long + self.gamma * consistency_loss
        return total_loss

class SimpleMSELoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.1):
        super(SimpleMSELoss, self).__init__()
        self.alpha = alpha  # Weight for Short-term Loss
        self.beta = beta    # Weight for Long-term Loss
        self.mse_loss = nn.MSELoss()

    def forward(self, output_long, target_long, output_short, target_short):
        # Short-term Loss
        loss_short = self.mse_loss(output_short, target_short)

        # Long-term Loss
        loss_long = self.mse_loss(output_long, target_long)

        # Combined Loss
        total_loss = self.alpha * loss_short + self.beta * loss_long
        return total_loss