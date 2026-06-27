"""
Module containing the Convolutional Recurrent Neural Network (CRNN) architecture
and CTC (Connectionist Temporal Classification) Label Converter.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for Optical Character Recognition.
    
    Attributes:
        hidden_size (int): The number of features in the hidden state of the LSTM.
        cnn (nn.Sequential): The convolutional feature extractor.
        rnn (nn.LSTM): The recurrent network for sequence modeling.
        classifier (nn.Linear): The linear layer for classification into character vocabulary.
    """

    def __init__(self, vocab_size, hidden_size=256):
        """
        Initializes the CRNN model.

        Args:
            vocab_size (int): Total number of distinct characters in the dataset plus one for the CTC blank character.
            hidden_size (int, optional): The hidden size for the LSTM layer. Defaults to 256.
        """
        super(CRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # Ekstraktor fitur berbasis CNN (VGG-style architecture)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), 
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), 
            
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)), 
            
            nn.Conv2d(512, 512, kernel_size=3, padding=1), 
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(4, 1), stride=(4, 1)), 
        )
        
        # Jaringan Rekuren berbasis Bidirectional LSTM
        self.rnn = nn.LSTM(512, hidden_size, num_layers=2, bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(hidden_size * 2, vocab_size)
        
    def forward(self, x):
        """
        Performs the forward pass of the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output tensor representing log probabilities of characters.
                          Shape is (batch_size, sequence_length, vocab_size).
        """
        conv_features = self.cnn(x) 
        b, c, h, w = conv_features.size()
        
        # Reshape fitur CNN untuk diumpankan ke dalam RNN
        # Mengubah bentuk menjadi (batch, width, channels * height)
        conv_features = conv_features.permute(0, 3, 1, 2) 
        conv_features = conv_features.contiguous().view(b, w, c * h) 
        
        rnn_out, _ = self.rnn(conv_features)
        output = self.classifier(rnn_out)
        output = F.log_softmax(output, dim=2)
        return output

class CTCLabelConverter:
    """
    A utility class to convert between text strings and CTC network indices.
    
    Attributes:
        chars (list): List of character string elements.
        blank_idx (int): The index allocated for the CTC blank symbol.
        char_to_idx (dict): Mapping from character to its index.
        idx_to_char (dict): Mapping from index to its character.
    """

    def __init__(self, charset="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ "):
        """
        Initializes the CTCLabelConverter.

        Args:
            charset (str, optional): A string containing all recognizable characters. 
                                     Defaults to standard alphanumeric characters and space.
        """
        self.chars = list(charset)
        self.blank_idx = 0
        
        # Indeks karakter asli dimulai dari 1 karena 0 digunakan untuk CTC [BLANK]
        self.char_to_idx = {char: idx + 1 for idx, char in enumerate(self.chars)}
        self.idx_to_char = {idx + 1: char for idx, char in enumerate(self.chars)}
        self.idx_to_char[0] = '[BLANK]'

    def decode(self, indices):
        """
        Decodes a sequence of indices into a text string by removing blanks and consecutive duplicate characters.

        Args:
            indices (list or numpy.ndarray): A sequence of predicted character indices.

        Returns:
            str: The decoded text string.
        """
        decoded = []
        prev_idx = -1
        for idx in indices:
            if idx != self.blank_idx and idx != prev_idx:
                if idx in self.idx_to_char:
                    decoded.append(self.idx_to_char[idx])
            prev_idx = idx
        return ''.join(decoded)

    def get_vocab_size(self):
        """
        Returns the total vocabulary size.

        Returns:
            int: Number of characters in the charset plus one (for the blank token).
        """
        return len(self.chars) + 1
