import os
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import params
from models import *

from dataset import RadarTrainSet, RadarTestSet


def train():
    bs = params.batch_sz
    lr = params.lr
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = RadarTrainSet("../code/filelists/radar/train.txt")

    train_set, test_set = torch.utils.data.random_split(dataset, [int(0.8 * len(dataset)), int(0.2 * len(dataset))])
    train_loader = DataLoader(dataset=train_set, batch_size=bs, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=bs)

    encoder = FFLSTMEncoder1(lstm_input_size=256, lstm_hidden_size=128, lstm_num_layers=2, fc2_size=128)
    classifier = FFLSTMClassifier(fc2_size=128, num_classes=5)
    net = FsHar_Net(encoder, classifier)
    net.to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    # for e in net.named_parameters():
    #     print(e)
    # exit()

    best_acc = 0
    print("Start training: ")
    for epoch in range(params.src_epochs):

        train_loss = 0.0
        train_corrects = 0.0
        val_corrects = 0.0

        net.train()
        for inputs, labels in train_loader:
            # print(inputs.shape, labels.shape)
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimizer
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                pred = F.softmax(outputs, dim=1).argmax(dim=1)
                train_corrects += torch.eq(pred, labels).sum().item()  # convert to numpy

            # print statistics
            train_loss += loss.item()
        train_loss = train_loss / len(train_set)

        net.eval()
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)

                pred = F.softmax(outputs, dim=1).argmax(dim=1)
                val_corrects += torch.eq(pred, labels).sum().item()  # convert to numpy

        print('[Epoch: %d] Train Loss: %.4f Train Acc: %4.2f Val Acc: %4.2f' %
              (epoch + 1, train_loss, train_corrects / len(train_set) * 100, val_corrects / len(test_set) * 100))

        if best_acc < val_corrects / len(test_set):
            print("Save best model until now! %.2f --> %.2f" % (best_acc, val_corrects / len(test_set)))
            best_acc = val_corrects / len(test_set)
            torch.save({'epoch': epoch, 'model_state_dict': net.state_dict(), },
                       os.path.join("./output/init", "best.tar"))

    return net


if __name__ == '__main__':
    # sets the random seed to a fixed value
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(222)

    # train model
    model = train()

    # save model
    torch.save({'epoch': params.src_epochs, 'model_state_dict': model.state_dict(), },
               os.path.join("./output/init", "last.tar"))
