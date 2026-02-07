import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

n_outputs = 1
class QNetwork(nn.Module):
    def __init__(self, n_features=4, n_outputs=1):
        super().__init__()
        self.n_outputs = n_outputs
        
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        
    def forward(self, x):
        batch_size = x.size(0)
        return self.net(x).view(batch_size, self.n_outputs)  # Reshape into (batch_size, 1)
    
    def predict(self, state, device="cpu"):
        """Unified prediction interface for single state (no batch).
        
        Handles both flat arrays and tuple (board, scalars) formats.
        Returns scalar value.
        """
        if isinstance(state, (tuple, list)) and len(state) == 2:
            # Tuple format: (board, scalars) - not supported by QNetwork
            raise ValueError("QNetwork expects flat state arrays, not tuples")
        else:
            # Flat array format
            state_tensor = torch.tensor([state], dtype=torch.float32, device=device)
            output = self.forward(state_tensor)
            return output[0].item()

class LargeModel(nn.Module):
    """CNN feature extractor for a board plus scalar inputs.

    Expects:
    - `board` tensor shape (B, C, H, W) e.g. (B, 1, 20, 10)
    - optional `scalars` tensor shape (B, scalar_feature_dim)

    The conv stack processes the board, then the flattened conv features are
    projected and concatenated with the scalar features before final FC layers
    that produce `n_outputs` values.
    """
    def __init__(self, in_channels=1, board_dim=(10, 20), scalar_feature_dim=14, n_outputs=n_outputs, hidden_conv_fc=256):
        super().__init__()

        # board_dim is (width, height) in the repo; we need (H, W) for shapes
        board_w, board_h = board_dim

        self.board_h = board_h
        self.board_w = board_w
        self.scalar_feature_dim = scalar_feature_dim
        self.n_outputs = n_outputs

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # LayerNorm takes normalized_shape (C, H, W)
        self.norm1 = nn.LayerNorm([32, board_h, board_w])
        self.norm2 = nn.LayerNorm([64, board_h, board_w])
        self.norm3 = nn.LayerNorm([64, board_h, board_w])

        conv_flat_dim = 64 * board_h * board_w
        self.fc_conv = nn.Linear(conv_flat_dim, hidden_conv_fc)

        # After conv features are projected, concatenate scalar features
        combined_dim = hidden_conv_fc + scalar_feature_dim

        self.fc_combined = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_outputs)
        )

        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, board, scalars=None):
        # board: (B, C, H, W)
        x = F.relu(self.norm1(self.conv1(board)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_conv(x))

        if scalars is None:
            # If no scalar features provided, return the conv feature vector
            return x

        # scalars: (B, scalar_feature_dim)
        combined = torch.cat([x, scalars], dim=1)
        out = self.fc_combined(combined)
        return out
    
    def predict(self, state, device = "cpu"):
        """Unified prediction interface for single state (no batch).
        
        Expects state as tuple (board, scalars).
        Returns scalar value.
        """
        if isinstance(state, (tuple, list)) and len(state) == 2:
            board, scalars = state
            # Add batch and channel dimensions: [20, 10] -> [1, 1, 20, 10]
            board_tensor = torch.tensor([[board]], dtype=torch.float32, device=device)
            scalars_tensor = torch.tensor([scalars], dtype=torch.float32, device=device)
            output = self.forward(board_tensor, scalars_tensor)
            return output[0].item()
        else:
            raise ValueError("LargeModel expects tuple state (board, scalars)")
    
    def batch_predict(self, states, device="cpu"):
        """Batch prediction for multiple states at once.
        
        Expects states as list of tuples (board, scalars).
        Returns list of scalar values (no gradient).
        """
        if not states:
            return []
        
        with torch.no_grad():
            # Add batch and channel dimensions
            boards = torch.tensor([[s[0] for s in states]], dtype=torch.float32, device=device).squeeze(0)
            scalars = torch.tensor([s[1] for s in states], dtype=torch.float32, device=device)
            
            # Ensure boards have correct shape [batch, 1, 20, 10]
            if boards.dim() == 3:
                boards = boards.unsqueeze(1)
            
            output = self.forward(boards, scalars)
            return output.squeeze(-1).cpu().tolist()


class DuelingDQNModel(nn.Module):
    """CNN feature extractor for a board plus scalar inputs.

    Expects:
    - `board` tensor shape (B, C, H, W) e.g. (B, 1, 20, 10)
    - optional `scalars` tensor shape (B, scalar_feature_dim)
    - optional `action_features` tensor shape (B, action_feature_dim) (represented in x,y,rotation,placed_piece, added_garbage)

    The conv stack processes the board, then the flattened conv features are
    projected and concatenated with the scalar features before final FC layers
    that produce `n_outputs` values.
    """
    def __init__(self, in_channels=1, board_dim=(10, 20), scalar_feature_dim=14, action_feature_dim = 5 ,n_outputs=n_outputs, hidden_conv_fc=256):
        super().__init__()

        # board_dim is (width, height) in the repo; we need (H, W) for shapes
        board_w, board_h = board_dim

        self.board_h = board_h
        self.board_w = board_w
        self.scalar_feature_dim = scalar_feature_dim
        self._action_feature_dim = action_feature_dim
        self.n_outputs = n_outputs

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        # LayerNorm takes normalized_shape (C, H, W)
        self.norm1 = nn.LayerNorm([32, board_h, board_w])
        self.norm2 = nn.LayerNorm([64, board_h, board_w])
        self.norm3 = nn.LayerNorm([64, board_h, board_w])

        conv_flat_dim = 64 * board_h * board_w
        self.fc_conv = nn.Linear(conv_flat_dim, hidden_conv_fc)

        # After conv features are projected, concatenate scalar features
        combined_dim = hidden_conv_fc + scalar_feature_dim

        self.fc_combined = nn.Sequential(
            nn.Linear(combined_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_outputs)
        )

        self.criterion = torch.nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)

    def forward(self, board, scalars=None):
        # board: (B, C, H, W)
        x = F.relu(self.norm1(self.conv1(board)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_conv(x))

        if scalars is None:
            # If no scalar features provided, return the conv feature vector
            return x

        # scalars: (B, scalar_feature_dim)
        combined = torch.cat([x, scalars], dim=1)
        out = self.fc_combined(combined)
        return out
    
    def predict(self, state, device = "cpu"):
        """Unified prediction interface for single state (no batch).
        
        Expects state as tuple (board, scalars).
        Returns scalar value.
        """
        if isinstance(state, (tuple, list)) and len(state) == 2:
            board, scalars = state
            # Add batch and channel dimensions: [20, 10] -> [1, 1, 20, 10]
            board_tensor = torch.tensor([[board]], dtype=torch.float32, device=device)
            scalars_tensor = torch.tensor([scalars], dtype=torch.float32, device=device)
            output = self.forward(board_tensor, scalars_tensor)
            return output[0].item()
        else:
            raise ValueError("LargeModel expects tuple state (board, scalars)")
    
    def batch_predict(self, states, device="cpu"):
        """Batch prediction for multiple states at once.
        
        Expects states as list of tuples (board, scalars).
        Returns list of scalar values (no gradient).
        """
        if not states:
            return []
        
        with torch.no_grad():
            # Add batch and channel dimensions
            boards = torch.tensor([[s[0] for s in states]], dtype=torch.float32, device=device).squeeze(0)
            scalars = torch.tensor([s[1] for s in states], dtype=torch.float32, device=device)
            
            # Ensure boards have correct shape [batch, 1, 20, 10]
            if boards.dim() == 3:
                boards = boards.unsqueeze(1)
            
            output = self.forward(boards, scalars)
            return output.squeeze(-1).cpu().tolist()