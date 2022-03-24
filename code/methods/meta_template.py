import time

import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
from tensorboardX import SummaryWriter

class MetaTemplate(nn.Module):
    def __init__(self, model_func, n_way, n_support, flatten=True, leakyrelu=False, change_way=True):
        super(MetaTemplate, self).__init__()
        self.n_way      = n_way
        self.n_support  = n_support
        self.n_query    = -1 #(change depends on input)
        self.feature    = model_func(flatten=flatten, leakyrelu=leakyrelu)
        self.feat_dim   = self.feature.final_feat_dim
        self.change_way = change_way  #some methods allow different_way classification during training and test

    @abstractmethod
    def set_forward(self,x,is_feature):
        pass

    @abstractmethod
    def set_forward_loss(self, x):
        pass

    def forward(self,x):
        out  = self.feature.forward(x)
        return out

    def parse_feature(self,x,is_feature):
        x = x.cuda()
        if is_feature:
            z_all = x
        else:
            x           = x.contiguous().view( self.n_way * (self.n_support + self.n_query), *x.size()[2:])
            z_all       = self.feature.forward(x)
            z_all       = z_all.view( self.n_way, self.n_support + self.n_query, -1)
        z_support   = z_all[:, :self.n_support]
        z_query     = z_all[:, self.n_support:]

        return z_support, z_query

    def correct(self, x):
        scores, loss = self.set_forward_loss(x)
        y_query = np.repeat(range( self.n_way ), self.n_query )

        topk_scores, topk_labels = torch.topk(scores, 1, 1)
        topk_ind = topk_labels.cpu().numpy()
        top1_correct = np.sum(topk_ind[:,0] == y_query)
        return float(top1_correct), len(y_query), loss.item()*len(y_query)

    def train_loop(self, epoch, train_loader, optimizer):
        print_freq = len(train_loader) // 5
        avg_loss=0
        for i, (x,_ ) in enumerate(train_loader):
            x = x[0] if x.shape[0] == 1 else x
            self.n_query = x.size(1) - self.n_support

            optimizer.zero_grad()
            _, loss = self.set_forward_loss(x.cuda())
            loss.backward()
            optimizer.step()
            avg_loss = avg_loss + loss.item()

            if (i + 1) % print_freq==0:
                print('Epoch {:d} | Batch {:d}/{:d} | Loss {:f}'.format(epoch, i + 1, len(train_loader), avg_loss/float(i+1)))

        return avg_loss / len(train_loader)

    def test_loop(self, test_loader, epoch=None):
        loss = 0.
        count = 0
        acc_all = []

        for i, (x,_) in enumerate(test_loader):
            x = x[0] if x.shape[0] == 1 else x
            self.n_query = x.size(1) - self.n_support
            correct_this, count_this, loss_this = self.correct(x.cuda())
            acc_all.append(correct_this / count_this * 100)
            loss += loss_this
            count += count_this

        print(["%.1f" % acc for acc in acc_all])
        acc_mean = np.mean(acc_all)
        acc_std  = np.std(acc_all)
        print('--- Val | Loss = %.6f | Acc = %4.2f%% +- %4.2f%% ---' %(loss/count, acc_mean, 1.96 * acc_std))

        return loss/count, acc_mean
