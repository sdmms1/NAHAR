import os
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

import io_utils, config
from backbone import TransformerModel
from dataset import TestSet, RadarTestSet
from torch.utils.data import DataLoader


def test(params, support_file, query_file, same_people):
    # load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    checkpoint_dir = io_utils.obtain_checkpoint_dir(config.save_dir, params.model, params.train_aug)
    path = os.path.join(checkpoint_dir, 'best.tar')
    state_dict = torch.load(path)['model_state_dict']
    state_dict.pop('classifier.L.weight_g', None)
    state_dict.pop('classifier.L.weight_v', None)

    dataset = RadarTestSet(support_file, query_file, length=params.task_num, same_people=same_people)
    dataloader = DataLoader(dataset, num_workers=4)

    corrects, total = 0, 0
    for i, (sp_x, sp_y, qr_x, qr_y) in enumerate(dataloader):
        sp_x, sp_y, qr_x, qr_y = sp_x.squeeze(0), sp_y.squeeze(0), qr_x.squeeze(0), qr_y.squeeze(0)
        net = TransformerModel(n_way=params.n_way, n_feature=200, n_head=8,
                               n_encoder_layers=12, dim_projection=128, dim_feedforward=512).to(device)
        net.load_state_dict(state_dict, strict=False)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, dampening=0.9, weight_decay=0.001)
        criterion = nn.CrossEntropyLoss()

        for name, param in net.named_parameters():
            if name not in ['classifier.L.weight_g', 'classifier.L.weight_v']:
                param.requires_grad = False

        # for name, param in net.named_parameters():
        #     if param.requires_grad == True:
        #         print(name, end=" ")
        # print()

        sp_x, sp_y, qr_x, qr_y = sp_x.to(device), sp_y.to(device), qr_x.to(device), qr_y.to(device)

        for epoch in range(300):
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
        # print('[Epoch: %d] Train Loss: %.4f Train Acc: %4.2f Val Acc: %4.2f' %
        #       (epoch + 1, loss.item() / sp_x.shape[1],
        #        train_correct / sp_x.shape[1] * 100, test_correct / qr_x.shape[1] * 100))

        corrects += test_correct # convert to numpy
        total += len(qr_y)

        if not (i + 1) % (params.task_num // 5):
            print("[Task %d] Cumulative Acc %4.2f" % (i + 1, corrects / total * 100))

    return corrects / total


if __name__ == '__main__':
    # sets the random seed to a fixed value
    torch.manual_seed(222)
    torch.cuda.manual_seed_all(222)
    np.random.seed(220)

    # obtain parameters
    params = io_utils.parse_args('test')

    for same_people in [True, False]:
        for env in ["env%d_eval" % i for i in range(1, 6)]:
            print('--- Eval in %s with %s people ---' % (env, "same" if same_people else "different"))
            acc = test(params, "../code/filelists/eval/%s.txt" % env, None, same_people)
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

                acc = test(params, "../code/filelists/eval/env%d_eval.txt" % j,
                           "../code/filelists/eval/env%d_eval.txt" % i, same_people)
                accs.append(acc)

                print("----- Eval in env%d using env%d with %s people | Acc = %4.2f%% -----" %
                      (i, j, "same" if same_people else "different", acc * 100))

            print("--- Eval in env%d using different environments with %s people | Acc = %4.2f%% -----" %
                  (i, "same" if same_people else "different", np.mean(accs) * 100))
