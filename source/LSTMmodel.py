import torch
import torch.nn as nn

class PointFeatureExtractor(nn.Module):
    def __init__(self, face_feature_dim=4, hand_feature_dim=3, output_size=128):
        super(PointFeatureExtractor, self).__init__()
        
        # CNN for face points (33 points, 4 features)
        self.face_cnn = nn.Sequential(
            nn.Conv1d(face_feature_dim, 64, kernel_size=1),  # Input: 4 features → 64 features
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),              # 64 → 128 features
            nn.ReLU()
        )
        
        # CNN for hand points (42 points, 3 features)
        self.hand_cnn = nn.Sequential(
            nn.Conv1d(hand_feature_dim, 64, kernel_size=1),  # Input: 3 features → 64 features
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=1),              # 64 → 128 features
            nn.ReLU()
        )
        
        # Fully connected layer to combine face and hand features
        self.fc = nn.Linear((33 + 42) * 128, output_size)  # Combine features for all points

    def forward(self, face_points, hand_points):
        # Face points: (batch_size, 33, 4)
        face_points = face_points.permute(0, 2, 1)  # Change to (batch_size, 4, 33)
        face_features = self.face_cnn(face_points)  # Output: (batch_size, 128, 33)
        
        # Hand points: (batch_size, 42, 3)
        hand_points = hand_points.permute(0, 2, 1)  # Change to (batch_size, 3, 42)
        hand_features = self.hand_cnn(hand_points)  # Output: (batch_size, 128, 42)
        
        # Flatten and concatenate features
        face_features = face_features.view(face_features.size(0), -1)  # Flatten: (batch_size, 33 * 128)
        hand_features = hand_features.view(hand_features.size(0), -1)  # Flatten: (batch_size, 42 * 128)
        
        combined_features = torch.cat([face_features, hand_features], dim=1)  # Concatenate along feature axis
        combined_output = self.fc(combined_features)  # Output: (batch_size, output_size)
        
        return combined_output
    
class CNNLSTMModel(nn.Module):
    def __init__(self, num_classes, output_size, lstm_hidden_size, lstm_num_layers, dropout):
        super(CNNLSTMModel, self).__init__()
        
        self.feature_extractor = PointFeatureExtractor(output_size=output_size)
        
        # LSTM to process temporal information
        self.lstm = nn.LSTM(input_size=output_size, 
                            hidden_size=lstm_hidden_size, 
                            num_layers=lstm_num_layers, 
                            batch_first=True, 
                            dropout=dropout)
        
        # Fully connected layer for classification
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
    
    def forward(self, face_points, hand_points):
        batch_size, seq_len = face_points.size(0), face_points.size(1)
        
        # Process each frame with the CNN
        cnn_features = []
        for t in range(seq_len):
            face_frame = face_points[:, t, :, :]  # Face points at time t
            hand_frame = hand_points[:, t, :, :]  # Hand points at time t
            frame_features = self.feature_extractor(face_frame, hand_frame)  # Extract features
            features.append(frame_features)
        
        # Stack CNN features for all frames
        features = torch.stack(features, dim=1)  # Shape: (batch_size, seq_len, cnn_output_size)
        
        # Pass CNN features through LSTM
        lstm_out, _ = self.lstm(features)  # Shape: (batch_size, seq_len, lstm_hidden_size)
        lstm_out = lstm_out[:, -1, :]  # Take the last time step's output
        
        # Pass through the fully connected layer
        out = self.fc(lstm_out)  # Shape: (batch_size, num_classes)
        
        return out
