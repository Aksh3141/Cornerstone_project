"""
Violence Detection Model: CNN + LSTM Architecture
"""

import torch
import torch.nn as nn
import torchvision.models as models

class ViolenceDetectionModel(nn.Module):
    def __init__(self, num_classes=2, hidden_size=256, num_lstm_layers=2, dropout=0.3):
        """
        CNN + LSTM model for violence detection
        
        Args:
            num_classes: Number of output classes (2 for Fight/NonFight)
            hidden_size: LSTM hidden dimension
            num_lstm_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super(ViolenceDetectionModel, self).__init__()
        
        # Use pretrained ResNet as feature extractor
        resnet = models.resnet50(pretrained=True)
        
        # Remove the final FC layer
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        
        # Feature dimension from ResNet50
        self.feature_dim = 2048
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, 256),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, time_steps, channels, height, width)
        Returns:
            Output logits of shape (batch, num_classes)
        """
        batch_size, time_steps, c, h, w = x.size()
        
        # Reshape to process all frames together
        x = x.view(batch_size * time_steps, c, h, w)
        
        # Extract features using CNN
        features = self.feature_extractor(x)
        features = features.view(batch_size, time_steps, -1)
        
        # Process temporal information with LSTM
        lstm_out, (h_n, c_n) = self.lstm(features)
        
        # Use the last LSTM output
        last_output = lstm_out[:, -1, :]
        
        # Classification
        logits = self.classifier(last_output)
        
        return logits


class EfficientNetLSTM(nn.Module):
    """Alternative model using EfficientNet (more efficient)"""
    def __init__(self, num_classes=2, hidden_size=256, num_lstm_layers=2, dropout=0.3):
        super(EfficientNetLSTM, self).__init__()
        
        # Use EfficientNet-B0 as feature extractor
        from torchvision.models import efficientnet_b0
        efficientnet = efficientnet_b0(pretrained=True)
        
        # Remove classifier
        self.feature_extractor = nn.Sequential(*list(efficientnet.children())[:-1])
        self.feature_dim = 1280  # EfficientNet-B0 feature dimension
        
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, num_classes)
        )
        
    def forward(self, x):
        batch_size, time_steps, c, h, w = x.size()
        x = x.view(batch_size * time_steps, c, h, w)
        
        features = self.feature_extractor(x)
        features = features.view(batch_size, time_steps, -1)
        
        lstm_out, _ = self.lstm(features)
        lstm_out = lstm_out.permute(0, 2, 1)  # (B, hidden*2, T)
        
        logits = self.classifier(lstm_out)
        return logits


# Test the model
if __name__ == "__main__":
    # Create model
    model = ViolenceDetectionModel(num_classes=2)
    
    # Test with dummy input
    batch_size = 4
    time_steps = 16
    dummy_input = torch.randn(batch_size, time_steps, 3, 224, 224)
    
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
