import os, torch
import random

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms


def parse_data_file(file):
    files = {}
    with open(file, 'r') as f:
        for e in f.readlines():
            e = e.strip().split(" ")
            label = int(e[-1])
            if label in files:
                files[label].append(e[0])
            else:
                files[label] = [e[0]]
    return files


class RadarTrainSet(Dataset):
    def __init__(self, file):
        with open(file, 'r') as f:
            self.files = [e.strip() for e in f]
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])

    def __getitem__(self, i):
        file, y = self.files[i].split(" ")
        x, y = np.load(file), int(y)
        x = torch.from_numpy(x).float()
        x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.files)


class RadarTestSet(Dataset):
    def __init__(self, support_file, query_file=None, length=100, same_people=False,
                 n_way=8, k_shot=3, k_query=12, tep=15):
        self.support_files = parse_data_file(support_file)
        self.query_files = parse_data_file(query_file) if query_file else None
        if query_file:
            assert len(self.support_files.keys()) == len(self.query_files.keys())
        assert len(self.support_files.keys()) == n_way
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        self.length = length
        self.same_people = same_people
        self.k_shot, self.k_query, self.tep = k_shot, k_query, tep
        if same_people:
            assert k_shot + k_query <= tep

    def __getitem__(self, item):
        sp_x, sp_y, qr_x, qr_y = [], [], [], []
        for key in self.support_files.keys():
            support_files = self.support_files[key]
            pidx = random.randint(0, len(support_files) // self.tep - 1)
            if self.same_people:
                if self.query_files:
                    query_files = self.query_files[key][pidx * self.tep: (pidx + 1) * self.tep]  # 同人不同环境
                else:
                    query_files = None  # 同人同环境
            else:
                if self.query_files:
                    query_files = self.query_files[key]
                    query_files = query_files[:pidx * self.tep] + query_files[(pidx + 1) * self.tep:]  # 不同人不同环境
                else:
                    query_files = support_files[:pidx * self.tep] + support_files[(pidx + 1) * self.tep:]  # 不同人同环境

            support_files = support_files[pidx * self.tep: (pidx + 1) * self.tep]

            if not query_files:
                files_idx = torch.randperm(len(support_files))
                for e in files_idx[:self.k_shot]:
                    sp_x.append(np.load(support_files[e]))
                    sp_y.append(key)
                for e in files_idx[self.k_shot:]:
                    qr_x.append(np.load(support_files[e]))
                    qr_y.append(key)
            else:
                for e in torch.randperm(len(support_files))[:self.k_shot]:
                    sp_x.append(np.load(support_files[e]))
                    sp_y.append(key)
                for e in torch.randperm(len(query_files))[:self.k_query]:
                    qr_x.append(np.load(query_files[e]))
                    qr_y.append(key)

        sp_x = torch.from_numpy(np.array(sp_x)).float()
        qr_x = torch.from_numpy(np.array(qr_x)).float()
        sp_y, qr_y = torch.from_numpy(np.array(sp_y)).float(), torch.from_numpy(np.array(qr_y)).float()

        return sp_x, sp_y, qr_x, qr_y

    def __len__(self):
        return self.length
