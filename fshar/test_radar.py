import os
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from skfeature.function.sparse_learning_based.ls_l21 import proximal_gradient_descent
from dataset import RadarTrainSet, RadarTestSet
from collections import OrderedDict
from torch.utils.data import DataLoader
import params
import h5py
from models import *

def getAProb(src_features, sp_features, sim_type="Cos", combination_type="Hard",
                  SRC_NUM_CLASSES=5, TGT_NUM_CLASSES=8, TGT_NUM_SAMP_PER_CLASS=3):
    if sim_type == "SR":
        A, _, _ = proximal_gradient_descent(np.transpose(src_features),
                                      np.transpose(sp_features),
                                      1e-2)
        APos = abs(A)
        NUM_SAMP_SRC = A.shape[0]
        SRC_NUM_SAMP_PER_CLASS = int(NUM_SAMP_SRC / SRC_NUM_CLASSES)
        AProbPrime = np.zeros((SRC_NUM_CLASSES, TGT_NUM_CLASSES))
        for i in range(SRC_NUM_CLASSES):
            for j in range(TGT_NUM_CLASSES):
                APosij = APos[i * SRC_NUM_SAMP_PER_CLASS:(i + 1) * SRC_NUM_SAMP_PER_CLASS,
                         j * TGT_NUM_SAMP_PER_CLASS:(j + 1) * TGT_NUM_SAMP_PER_CLASS]
                AProbPrime[i, j] = sum(sum(APosij))
    elif sim_type == "Cos":
        feat_src_norm = np.zeros(src_features.shape)
        for i in range(src_features.shape[0]):
            x = np.squeeze(src_features[i, :])
            feat_src_norm[i, :] = x / np.linalg.norm(x)
        feat_train_tgt_norm = np.zeros(sp_features.shape)
        for i in range(sp_features.shape[0]):
            x = np.squeeze(sp_features[i, :])
            feat_train_tgt_norm[i, :] = x / np.linalg.norm(x)
        A = np.exp(np.dot(feat_src_norm, np.transpose(feat_train_tgt_norm)))
        NUM_SAMP_SRC = A.shape[0]
        SRC_NUM_SAMP_PER_CLASS = int(NUM_SAMP_SRC / SRC_NUM_CLASSES)
        AProbPrime = np.zeros((SRC_NUM_CLASSES, TGT_NUM_CLASSES))
        for i in range(SRC_NUM_CLASSES):
            for j in range(TGT_NUM_CLASSES):
                Aij = A[i * SRC_NUM_SAMP_PER_CLASS:(i + 1) * SRC_NUM_SAMP_PER_CLASS,
                      j * TGT_NUM_SAMP_PER_CLASS:(j + 1) * TGT_NUM_SAMP_PER_CLASS]
                AProbPrime[i, j] = sum(sum(Aij))
    else:
        raise NotImplementedError

    AProb = np.zeros((SRC_NUM_CLASSES,TGT_NUM_CLASSES))
    for j in range(TGT_NUM_CLASSES):
        AProbPrimej = np.squeeze(AProbPrime[:,j])
        if combination_type == "Soft":
            AProb[:,j] = AProbPrimej/sum(AProbPrimej)
        elif combination_type == "Hard":
            AProb[np.argmax(AProbPrimej).astype(int),j] = 1.0
        else:
            raise NotImplementedError

    return AProbPrime

def test(support_file, query_file, same_people):
    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = os.path.join("./output/init", 'best.tar')
    state_dict = torch.load(path)['model_state_dict']
    encoder_dict, classifier_dict = OrderedDict(), OrderedDict()
    for key in state_dict:
        if "encoder" in key:
            encoder_dict[key[8:]] = state_dict[key]
        elif "classifier" in key:
            classifier_dict[key[11:]] = state_dict[key]

    src_encoder = FFLSTMEncoder1(lstm_input_size=256, lstm_hidden_size=128, lstm_num_layers=2, fc2_size=128)
    src_encoder.load_state_dict(encoder_dict)
    src_classifier = FFLSTMClassifier(fc2_size=128, num_classes=5)
    src_classifier.load_state_dict(classifier_dict)
    src_encoder.to(device)
    src_encoder.eval()

    src_dataset = RadarTrainSet("../code/filelists/radar/train.txt")
    src_dataloader = DataLoader(dataset=src_dataset, batch_size=len(src_dataset), num_workers=4)
    src_features = None
    with torch.no_grad():
        for x, y in src_dataloader:
            src_features = src_encoder(x.to(device))
    # print(src_features.shape)

    # src_net = FsHar_Net(src_encoder, src_classifier)
    # src_net.to(device)
    # corrects = 0
    # src_net.eval()
    # with torch.no_grad():
    #     for inputs, labels in src_dataloader:
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         outputs = src_net(inputs)
    #
    #         pred = F.softmax(outputs, dim=1).argmax(dim=1)
    #         corrects += torch.eq(pred, labels).sum().item()  # convert to numpy
    # print(corrects / len(src_dataset))
    # exit()

    dataset = RadarTestSet(support_file, query_file, length=params.task_num, same_people=same_people)
    dataloader = DataLoader(dataset, num_workers=4)

    corrects, total = 0, 0
    for i, (sp_x, sp_y, qr_x, qr_y) in enumerate(dataloader):
        sp_x, sp_y, qr_x, qr_y = sp_x.squeeze(0), sp_y.squeeze(0), qr_x.squeeze(0), qr_y.squeeze(0)
        # print(sp_x.shape, sp_y.shape, qr_x.shape, qr_y.shape)
        with torch.no_grad():
            sp_features = src_encoder(sp_x.to(device))
        # print(sp_features.shape)

        A = getAProb(src_features.detach().cpu().numpy(), sp_features.detach().cpu().numpy())
        # print(A.shape)
        src_classifier_weight = src_classifier.fc.weight.detach()
        tgt_classifier_weight = np.matmul(np.transpose(A), src_classifier_weight)

        encoder = FFLSTMEncoder1(lstm_input_size=256, lstm_hidden_size=128, lstm_num_layers=2, fc2_size=128)
        encoder.load_state_dict(encoder_dict)

        classifier = FFLSTMClassifier(fc2_size=128, num_classes=8)
        classifier.fc.weight.data.copy_(torch.as_tensor(tgt_classifier_weight))

        net = FsHar_Net(encoder, classifier)
        net.to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=0.001)
        criterion = nn.CrossEntropyLoss()

        # for name, param in net.named_parameters():
        #     if param.requires_grad == True:
        #         print(name)
        # print()
        # exit()

        sp_x, sp_y, qr_x, qr_y = sp_x.to(device), sp_y.to(device), qr_x.to(device), qr_y.to(device)

        for epoch in range(500):
            net.train()
            logits = net(sp_x)
            loss = criterion(logits, sp_y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # pred = F.softmax(logits, dim=1).argmax(dim=1)
            # train_correct = torch.eq(pred, sp_y).sum().item()

        net.eval()
        with torch.no_grad():
            logits = net(qr_x)
            pred = F.softmax(logits, dim=1).argmax(dim=1)
            test_correct = torch.eq(pred, qr_y).sum().item()
            # print(pred, qr_y)
            # exit()
        # if not (epoch + 1) % 100:
        #     print('[Epoch: %d] Train Loss: %.4f Train Acc: %4.2f Val Acc: %4.2f' %
        #           (epoch + 1, loss.item() / sp_x.shape[0],
        #            train_correct / sp_x.shape[0] * 100, test_correct / qr_x.shape[0] * 100))

        corrects += test_correct # convert to numpy
        total += len(qr_y)

        if not (i + 1) % (params.task_num // 10):
            print("[Task %d] Cumulative Acc %4.2f" % (i + 1, corrects / total * 100))

    return corrects / total


if __name__ == '__main__':
    # sets the random seed to a fixed value
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(220)

    for same_people in [True, False]:
        for env in ["env%d_eval" % i for i in range(1, 6)]:
            print('--- Eval in %s with %s people ---' % (env, "same" if same_people else "different"))
            acc = test("../code/filelists/eval/%s.txt" % env, None, same_people)
            print('--- Eval in %s with %s people | Acc = %4.2f%% ---' %
                  (env, "same" if same_people else "different", acc * 100))

    for same_people in [True, False]:
        for i in range(2, 6):
            accs = []
            for j in range(2, 6):
                if i == j:
                    continue

                print("-------Eval in env%d using env%d with %s people-------" %
                      (i, j, "same" if same_people else "different"))

                acc = test("../code/filelists/eval/env%d_eval.txt" % j,
                           "../code/filelists/eval/env%d_eval.txt" % i, same_people)
                accs.append(acc)

                print("----- Eval in env%d using env%d with %s people | Acc = %4.2f%% -----" %
                      (i, j, "same" if same_people else "different", acc * 100))

            print("--- Eval in env%d using different environments with %s people | Acc = %4.2f%% -----" %
                  (i, "same" if same_people else "different", np.mean(accs) * 100))
