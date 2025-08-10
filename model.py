import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, img_height, num_channels, num_classes):
        super(CRNN, self).__init__()
        
        # CNN Part (fixed syntax)
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/2
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # H/4
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)),  # H/8
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),  # H/16
            
            # Additional layer to ensure height=1
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1))  # H/32
        )

        # RNN Part
        self.lstm1 = nn.LSTM(input_size=512, hidden_size=256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(input_size=512, hidden_size=256, bidirectional=True, batch_first=True)
        
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        # CNN Forward
        conv = self.cnn(x)
        batch, channel, height, width = conv.size()
        
        # Ensure height is 1
        if height != 1:
            # Adaptive pooling as fallback if height isn't 1
            conv = nn.functional.adaptive_avg_pool2d(conv, (1, width))
            batch, channel, height, width = conv.size()
        
        # Prepare for RNN
        conv = conv.squeeze(2)  # [batch, channel, width]
        conv = conv.permute(0, 2, 1)  # [batch, width, channel]
        
        # RNN Forward
        recurrent, _ = self.lstm1(conv)
        recurrent, _ = self.lstm2(recurrent)
        
        # Output
        output = self.fc(recurrent)  # [batch, width, num_classes]
        return output.permute(1, 0, 2)  # [width, batch, num_classes] for CTC