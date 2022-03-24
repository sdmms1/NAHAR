import torch
import torch.nn as nn
import numpy as np
from .meta_template import MetaTemplate
from .gnn import GNN_nl
from . import backbone
from itertools import combinations


class GnnNet(nn.Module):
    maml=False
    def __init__(self, model_func,  n_way, n_support, n_query, flatten=True, leakyrelu=False):
        super(GnnNet, self).__init__()
        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = n_query #(change depends on input)
        self.feature    = model_func(flatten=flatten, leakyrelu=leakyrelu)
        self.feat_dim   = self.feature.final_feat_dim
        # loss function
        self.loss_fn = nn.CrossEntropyLoss()

        # metric function
        self.fc = nn.Sequential(nn.Linear(self.feat_dim, 128), nn.BatchNorm1d(128, track_running_stats=False)) if not self.maml \
            else nn.Sequential(backbone.Linear_fw(self.feat_dim, 128), backbone.BatchNorm1d_fw(128, track_running_stats=False))
        self.gnn = GNN_nl(128 + self.n_way, 96, self.n_way)
        self.method = 'GnnNet'

        # fix label for training the metric function   1*nw(1 + ns)*nw
        support_label = torch.from_numpy(np.repeat(range(self.n_way), self.n_support)).unsqueeze(1)
        support_label = torch.zeros(self.n_way*self.n_support, self.n_way).scatter(1, support_label, 1).view(self.n_way, self.n_support, self.n_way)
        support_label = torch.cat([support_label, torch.zeros(self.n_way, 1, n_way)], dim=1)
        self.support_label = support_label.view(1, -1, self.n_way)

    def cuda(self):
        self.feature.cuda()
        self.fc.cuda()
        self.gnn.cuda()
        self.support_label = self.support_label.cuda()
        return self

    def forward(self, x):
        # input x: n_way * (n_support + n_query) * features
        # feature extraction
        x = x.view(-1, *x.size()[2:])
        temp = self.feature(x)
        z = self.fc(temp)
        z = z.view(self.n_way, -1, z.size(1))

        # stack the feature for metric function: n_way * n_s + n_q * f -> n_q * [1 * n_way(n_s + 1) * f]
        z_stack = [torch.cat([z[:, :self.n_support], z[:, self.n_support + i:self.n_support + i + 1]], dim=1).view(1, -1, z.size(2)) for i in range(self.n_query)]
        assert(z_stack[0].size(1) == self.n_way*(self.n_support + 1))

        scores = self.forward_gnn(z_stack)
        return scores

    def forward_gnn(self, zs):
        # label embedding
        nodes = torch.cat([torch.cat([z, self.support_label], dim=2) for z in zs], dim=0)
        scores = self.gnn(nodes)

        # n_q * n_way(n_s + 1) * n_way -> (n_way * n_q) * n_way
        scores = scores.view(self.n_query, self.n_way, self.n_support + 1, self.n_way)[:, :, -1].permute(1, 0, 2).contiguous().view(-1, self.n_way)
        return scores

    def get_fs_label(self, y):
        assert y.shape[0] == self.n_way, y.shape[1] == self.n_support + self.n_query
        y_query = torch.zeros(y.shape, dtype=torch.int64)
        for i in range(y.shape[0]):
            y_query += (y == y[i, 0]) * (i + 1)
        y_query -= 1
        y_query = y_query[:, 3:]
        # print(y.shape, y_query.shape)
        return y_query.contiguous().view(-1)

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = len(train_loader) // 5
        avg_loss=0
        for i, (x, y) in enumerate(train_loader):
            x , y_query = x[0].cuda(), self.get_fs_label(y[0]).cuda()

            optimizer.zero_grad()
            scores = self.forward(x)
            loss = self.loss_fn(scores, y_query)
            loss.backward()
            optimizer.step()

            avg_loss = avg_loss + loss.item()

            if (i + 1) % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i + 1, len(train_loader), avg_loss/float(i+1)))

        return avg_loss / len(train_loader)

    def test_loop(self, test_loader):
        avg_loss = 0.
        acc_all = []

        for x, y in test_loader:
            x, y = x[0], y[0]
            y_query = self.get_fs_label(y)

            scores = self.forward(x.cuda())
            loss = self.loss_fn(scores, y_query.cuda())

            topk_scores, topk_labels = torch.topk(scores, 1, 1)
            topk_inx = topk_labels.cpu().squeeze(1)
            top1_correct = np.sum((topk_inx == y_query).numpy())

            acc_all.append(top1_correct / len(y_query) * 100)
            avg_loss += loss.item()

        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        avg_loss /= len(test_loader)
        print('--- Val | Loss = %.6f | Acc = %4.2f%% +- %4.2f%% ---' %(avg_loss, acc_mean, 1.96 * acc_std))

        return avg_loss, acc_mean

    def system_evaluation(self, dataloader):

        correct_num = 0
        for i, (X, y) in enumerate(dataloader):
            X = X.squeeze(0)
            # print(X.shape, y)
            votes = np.zeros(len(X))
            for combination in combinations(range(len(X)), self.n_way):
                x = X[combination,]
                # print(combination, x.shape)
                scores = self.forward(x.cuda())

                topk_scores, topk_labels = torch.topk(scores, 1, 1)
                topk_inx = topk_labels.cpu().squeeze(1).numpy().tolist()
                # print(topk_inx)
                choice = list(set(topk_inx))
                assert len(choice) == 1

                votes[combination[int(choice[0])]] += 1

            pred = np.argmax(votes)
            if pred == y:
                correct_num += 1
            else:
                print(votes)
                print("pred: %d, truth: %d, (%d/%d)" % (pred, int(y[0]), correct_num, i + 1))

        return correct_num / len(dataloader)