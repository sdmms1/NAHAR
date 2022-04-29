"""Models for HAR"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


class FFLSTMEncoder1(nn.Module):
    """FFLSTM encoder model for ADDA."""

    def __init__(self, lstm_input_size, lstm_hidden_size, lstm_num_layers,
                 fc2_size):
        """Init FFLSTM encoder."""
        super(FFLSTMEncoder1, self).__init__()

        self.restored = False

        self.lstm_input_size = lstm_input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.fc2_size = fc2_size

        self.lstm = nn.LSTM(lstm_input_size,
                            lstm_hidden_size,
                            lstm_num_layers,
                            batch_first=True)

        self.fc2 = nn.Linear(lstm_hidden_size, fc2_size)
        # self.h0 = torch.zeros(self.lstm_num_layers, lstm_input_size, self.lstm_hidden_size)
        # self.c0 = torch.zeros(self.lstm_num_layers,
        #                                  lstm_input_size,
        #                                   self.lstm_hidden_size)

    def forward(self, x):
        """Forward the FFLSTM."""
        cuda_check = x.is_cuda
        if cuda_check:
            x_device = "cuda:" + str(x.get_device())
        else:
            x_device = "cpu"
        # print("x_device", x_device)
        h0 = torch.zeros(self.lstm_num_layers,
                         x.size(0),
                         self.lstm_hidden_size).to(x_device)
        c0 = torch.zeros(self.lstm_num_layers,
                         x.size(0),
                         self.lstm_hidden_size).to(x_device)

        r_out, (h_n, h_c) = self.lstm(x, (h0, c0))
        out2 = self.fc2(r_out[:, -1, :])
        return out2


class FFLSTMClassifier(nn.Module):
    """FFLSTM classifier model for ADDA."""

    def __init__(self, fc2_size, num_classes):
        """Init FFLSTM encoder."""
        super(FFLSTMClassifier, self).__init__()
        self.fc = nn.Linear(fc2_size, num_classes)

    def forward(self, feat):
        """Forward the FFLSTM classifier."""
        out = F.dropout(F.relu(feat), training=self.training)
        out = self.fc(out)
        return out

class FsHar_Net(nn.Module):
    def __init__(self, encoder, classifier):
        super(FsHar_Net, self).__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, x):
        return self.classifier(self.encoder(x))



if __name__ == "__main__":
    input = torch.randn((2400, 100, 256)).cuda()
    # print(input)
    encoder = FFLSTMEncoder1(lstm_input_size=256, lstm_hidden_size=100, lstm_num_layers=2, fc2_size=256)
    classifier = FFLSTMClassifier(fc2_size=256, num_classes=5)
    encoder = encoder.cuda()
    classifier = classifier.cuda()
    output = classifier(encoder(input))
    print(output.shape)
