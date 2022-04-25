# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Sampler, Dataset
import os
import random

def get_img(file, transform=None):
    arr = np.load(file)  # read img
    if arr.shape[0] != arr.shape[1]:
        idx = np.random.randint(low=0, high=arr.shape[1] - arr.shape[0])
        arr = arr[:, idx:idx + arr.shape[0]]
    arr = np.concatenate([np.expand_dims(arr, axis=0)] * 3)  # expand to 3 channels
    arr = torch.from_numpy(arr).float()  # convert to tensor
    if transform:
        arr = transform(arr).numpy()
    return arr

class SimpleFewShotDataset(Dataset):
    """
    Sample out data with size n_way * [n_support + n_query], shuffle means whether the position of
    query sample corresponding to the position of support sample
    """

    def __init__(self, files, n_way, n_support, n_query, tep, transform=None):
        self.files, self.classes = files, list(files.keys())
        self.n_way, self.n_support, self.n_query, self.tep = n_way, n_support, n_query, tep
        assert self.n_way <= len(self.files)
        self.transform = transform

    def __getitem__(self, i):
        support_set, query_set = [], []
        for class_idx in torch.randperm(len(self.classes))[:self.n_way]:
            cls = self.classes[class_idx]
            files = self.files[cls]
            files_idx = torch.randperm(len(files))[:self.n_support + self.n_query]
            support_set += [(files[idx], cls) for idx in files_idx[:self.n_support]]
            query_set += [(files[idx], cls) for idx in files_idx[self.n_support:]]

        random.shuffle(query_set)

        x, y = [], []
        for i in range(self.n_way):
            x.append([])
            y.append([])
            for file, cls in support_set[i * self.n_support: (i + 1) * self.n_support]:
                x[-1].append(get_img(file, self.transform))
                y[-1].append(cls)
            for file, cls in query_set[i * self.n_query: (i + 1) * self.n_query]:
                x[-1].append(get_img(file, self.transform))
                y[-1].append(cls)

        x, y = np.array(x), np.array(y)
        return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        return self.tep

class FewShotDatasetByPeople(Dataset):
    """
    Sample out data with size n_way * [n_support + n_query], ensuring that support set and
    query set come from the same people
    """

    def __init__(self, files, n_way, n_support, n_query, tep, transform=None, sample_each_people=15,
                 single=True, same_people=True):
        self.files, self.classes = files, list(files.keys())
        self.n_way, self.n_support, self.n_query, self.tep = n_way, n_support, n_query, tep
        assert self.n_way <= len(self.files)
        self.transform = transform
        self.sample_each_people = sample_each_people
        assert self.n_support + self.n_query <= self.sample_each_people
        self.single = single
        self.sp = same_people

    def __getitem__(self, i):
        support_set, query_set = [], []
        for class_idx in torch.randperm(len(self.classes))[:self.n_way]:
            cls = self.classes[class_idx]
            files = self.files[cls]
            pidx = random.randint(0, len(files) // self.sample_each_people - 1)
            support_files = files[:pidx * self.sample_each_people] + files[(pidx + 1) * self.sample_each_people:]
            query_files = files[pidx * self.sample_each_people : (pidx + 1) * self.sample_each_people]
            if self.sp:
                # support and query from same people
                assert self.n_support + self.n_query <= len(query_files)
                file_idx = torch.randperm(len(query_files))
                support_set += [(query_files[idx], cls) for idx in file_idx[:self.n_support]]
                query_set += [(query_files[idx], cls) for idx in file_idx[-self.n_query:]]
            else:
                # support and query from different people
                support_set += [(support_files[idx], cls) for idx in torch.randperm(len(support_files))[:self.n_support]]
                query_set += [(query_files[idx], cls) for idx in torch.randperm(len(query_files))[:self.n_query]]

        x, y = [], []
        for i in range(self.n_way):
            x.append([])
            y.append([])
            for file, cls in support_set[i * self.n_support: (i + 1) * self.n_support]:
                x[-1].append(get_img(file, self.transform))
                y[-1].append(cls)
            if self.single:
                # if single, fulfill the query set with unique query sample
                file, cls = query_set[0]
                x[-1].append(get_img(file, self.transform))
                y[-1].append(cls)
            else:
                for file, cls in query_set[i * self.n_query: (i + 1) * self.n_query]:
                    x[-1].append(get_img(file, self.transform))
                    y[-1].append(cls)

        x, y = np.array(x), np.array(y)
        return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        return self.tep

class CrossFewShotDatasetByPeople(Dataset):
    """
    Sample out data with size n_way * [n_support + n_query], ensuring that support set and
    query set come from the same people
    """

    def __init__(self, support_files, query_files, n_way, n_support, n_query, tep,
                 transform=None, sample_each_people=15,single=True, same_people=True):
        self.support_files, self.query_files, self.classes = support_files, query_files, list(support_files.keys())
        self.n_way, self.n_support, self.n_query, self.tep = n_way, n_support, n_query, tep
        assert self.n_way <= len(self.classes)
        self.transform = transform
        self.sample_each_people = sample_each_people
        assert self.n_support + self.n_query <= self.sample_each_people
        self.single = single
        self.sp = same_people

    def __getitem__(self, i):
        support_set, query_set = [], []
        for class_idx in torch.randperm(len(self.classes))[:self.n_way]:
            cls = self.classes[class_idx]
            files = self.support_files[cls]
            pidx = random.randint(0, len(files) // self.sample_each_people - 1)
            if self.sp:
                # support and query from same people
                support_files = files[pidx * self.sample_each_people: (pidx + 1) * self.sample_each_people]
            else:
                # support and query from different people
                support_files = files[:pidx * self.sample_each_people] + files[(pidx + 1) * self.sample_each_people:]
            query_files = self.query_files[cls][pidx * self.sample_each_people : (pidx + 1) * self.sample_each_people]
            support_set += [(support_files[idx], cls) for idx in torch.randperm(len(support_files))[:self.n_support]]
            query_set += [(query_files[idx], cls) for idx in torch.randperm(len(query_files))[:self.n_query]]

        x, y = [], []
        for i in range(self.n_way):
            x.append([])
            y.append([])
            for file, cls in support_set[i * self.n_support: (i + 1) * self.n_support]:
                x[-1].append(get_img(file, self.transform))
                y[-1].append(cls)
            if self.single:
                # if single, fulfill the query set with unique query sample
                file, cls = query_set[0]
                x[-1].append(get_img(file, self.transform))
                y[-1].append(cls)
            else:
                for file, cls in query_set[i * self.n_query: (i + 1) * self.n_query]:
                    x[-1].append(get_img(file, self.transform))
                    y[-1].append(cls)

        x, y = np.array(x), np.array(y)
        return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        return self.tep

class FineTuneDataset(Dataset):
    """
    Sample out data with size n_way * [support + query](where support and query is similar in each way)
    """
    def __init__(self, support_files, query_files, n_way, n_support, n_query, tep, transform = None):
        self.support_files, self.query_files, self.classes = support_files, query_files, list(support_files.keys())
        self.n_way, self.n_support, self.n_query, self.tep = n_way, n_support, n_query, tep
        assert self.n_way <= len(self.support_files)
        self.transform = transform

    def __getitem__(self, i):
        support_set, query_set = [], []
        for class_idx in torch.randperm(len(self.classes))[:self.n_way]:
            cls = self.classes[class_idx]

            files = self.support_files[cls]
            files_idx = torch.randperm(len(files))[:self.n_support]
            support_set += [(files[idx], cls) for idx in files_idx]

            files = self.query_files[cls]
            files_idx = torch.randperm(len(files))[:self.n_query]
            query_set += [(files[idx], cls) for idx in files_idx]

        random.shuffle(query_set)

        x, y = [], []
        for i in range(self.n_way):
            x.append([])
            y.append([])
            for file, cls in support_set[i * self.n_support: (i + 1) * self.n_support]:
                x[-1].append(get_img(file, self.transform))
                y[-1].append(cls)
            for file, cls in query_set[i * self.n_query: (i + 1) * self.n_query]:
                x[-1].append(get_img(file, self.transform))
                y[-1].append(cls)

        x, y = np.array(x), np.array(y)
        return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        return self.tep

class SystemEvaluationDataset(Dataset):
    """
     Sample out data for system evaluation
     total_class * (n_support + 1), query_sample is the sample to be query
    """

    def __init__(self, support_files, query_files, n_way, n_support, n_query, tep, transform=None, same_people=True):
        self.support_files, self.query_files, self.classes = support_files, query_files, list(support_files.keys())
        self.n_support, self.tep = n_support, tep
        self.transform = transform
        self.same_people = same_people

    def __getitem__(self, i):
        x, y = [], random.choice(self.classes)
        fidx = random.randint(0, len(self.query_files[y]) - 1)
        pidx = fidx // 15
        query_file = self.query_files[y][fidx]
        # print(query_file)
        query_sample = get_img(query_file, self.transform)

        for cls in self.classes:
            if cls == y:
                if self.same_people:
                    support_files = self.support_files[y][pidx * 15:(pidx + 1) * 15]
                    support_files.pop(fidx % 15) # avoid the query sample appear in the support set
                else:
                    support_files = self.support_files[y][:pidx * 15] + self.support_files[y][(pidx + 1) * 15:]
            else:
                support_files = self.support_files[cls]
            files_idx = torch.randperm(len(support_files))[:self.n_support]
            x.append([get_img(support_files[idx], self.transform) for idx in files_idx])

        for e in x:
            e.append(query_sample)

        return np.array(x), y

    def __len__(self):
        return self.tep

class TransferMethodDataset(Dataset):
    """
    Sample out data with size n_way * [support + query](where support and query is similar in each way)
    """
    def __init__(self, support_files, query_files, n_way, n_support, n_query, tep, transform = None):
        self.support_files, self.query_files, self.classes = support_files, query_files, list(support_files.keys())
        self.n_way, self.n_support, self.n_query, self.tep = n_way, n_support, n_query, tep
        assert self.n_way <= len(self.support_files)
        self.transform = transform

    def __getitem__(self, i):
        support_set, query_set = [], []
        p = random.random()
        for class_idx in torch.randperm(len(self.classes))[:self.n_way]:
            cls = self.classes[class_idx]

            files = self.support_files[cls]
            files_idx = torch.randperm(len(files))[:self.n_support]
            support_set += [(files[idx], cls) for idx in files_idx]

            # if p < 0.1:
            files = self.query_files[cls]
            files_idx = torch.randperm(len(files))[:self.n_query]
            query_set += [(files[idx], cls) for idx in files_idx]
            # else:
            #     files = self.support_files[cls]
            #     files_idx = torch.randperm(len(files))[:self.n_query]
            #     query_set += [(files[idx], cls) for idx in files_idx]

        random.shuffle(query_set)

        x, y = [], []
        for i in range(self.n_way):
            x.append([])
            y.append([])
            for file, cls in support_set[i * self.n_support: (i + 1) * self.n_support]:
                x[-1].append(get_img(file, self.transform))
                y[-1].append(cls)
            for file, cls in query_set[i * self.n_query: (i + 1) * self.n_query]:
                x[-1].append(get_img(file, self.transform))
                y[-1].append(cls)

        x, y = np.array(x), np.array(y)
        return torch.from_numpy(x), torch.from_numpy(y)

    def __len__(self):
        return self.tep