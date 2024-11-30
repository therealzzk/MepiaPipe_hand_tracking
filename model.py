import json
import os
import cv2
import mediapipe as mp
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionLayer, self).__init__()
        self.attention_weights = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        nn.init.xavier_uniform_(self.attention_weights)

    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_length, hidden_size)
        attention_scores = torch.matmul(lstm_output, self.attention_weights)  # (batch_size, seq_length, hidden_size) * (hidden_size, hidden_size)
        attention_scores = torch.matmul(attention_scores, lstm_output.transpose(1, 2))  # Attention scores (batch_size, seq_length, seq_length)
        attention_weights = F.softmax(attention_scores, dim=-1)  # Normalize to get attention weights
        context_vector = torch.matmul(attention_weights, lstm_output)  # Weighted sum of LSTM outputs
        return context_vector, attention_weights

class SignLanguageModel(nn.Model):
    def __init__(self, input_dim=258, num_classes=64, lstm_hidden_size=64, attention_input_size=64):
        super(SignLanguageModel, self).__init__()

        self.lstm1 = nn.LSTM(input_dim, lstm_hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(lstm_hidden_size * 2, lstm_hidden_size, batch_first=True, bidirectional=True)  # bidirectional
        self.lstm3 = nn.LSTM(lstm_hidden_size * 2, lstm_hidden_size, batch_first=True)
        
        # self.attention = AttentionLayer(lstm_hidden_size)

        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )


    def forward(self, x):
        #TODO
        # LSTM layers
        x, _ = self.lstm1(x)  # Output shape: [batch_size, seq_len, lstm_hidden_size*2]
        x = F.relu(x)
        x, _ = self.lstm2(x)  # Output shape: [batch_size, seq_len, lstm_hidden_size*2]
        x = F.relu(x)
        x, _ = self.lstm3(x)  # Output shape: [batch_size, seq_len, lstm_hidden_size]
        x = F.relu(x)
        
        # Attention mechanism to focus on important frames
        # attended_out, attn_weights = self.attention(x)  # Output shape: [batch_size, lstm_hidden_size]
        
        # Pass through Dense layers
        x = self.fc_layers(x)  # Output shape: [batch_size, num_classes]
        # x = nn.Softmax(dim=1)(logits)
 
        return x