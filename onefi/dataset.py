import os, torch
import random

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import config


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


class AugmentSet(Dataset):
    # DATA SHAPE: [n_action, n_trial, n_orientation, n_receiver, n_feature, n_timestamp]
    path = os.path.join(config.data_dir, "train_data.npy")

    def __init__(self):
        # load data from npy file
        self.data = torch.from_numpy(np.load(self.path)).float()

        # obtain dataset params
        self.n_action = 5
        self.n_orientation = 1
        self.n_trial = 300
        self.n_receiver = 1
        self.n_feature = 200
        self.n_timestamp = 128

        self.n_class = self.n_action * self.n_orientation
        self.len = self.n_class * self.n_trial

    def __getitem__(self, index):
        class_id = index // self.n_trial
        image_id = index % self.n_trial

        action_id = class_id // self.n_orientation
        orient_id = class_id % self.n_orientation
        item = self.data[action_id, image_id, orient_id]
        label = class_id

        # [n_receiver, n_feature, n_timestamp] --> [n_receiver*n_feature, n_timestamp]
        item = np.reshape(item, (self.n_receiver * self.n_feature, self.n_timestamp))

        return item, label

    def __len__(self):
        return self.len


class TestSet():
    # DATA SHAPE: [n_class, n_trial, n_receiver, n_feature, n_timestamp]

    def __init__(self, n_way, k_shot, k_query, test_file):
        self.n_way = n_way
        self.k_shot = k_shot
        self.k_query = k_query

        data = np.load(os.path.join(config.data_dir, test_file))
        self.data = data[:]
        self.n_img = self.data.shape[1]
        print(self.data.shape)
        self.resize = (1, 200, 128)

    def load_test_set(self):
        # take 5 way 1 shot as example: 5 * 1
        setsz = self.k_shot * self.n_way
        qrysz = self.k_query * self.n_way

        # selected_cls = np.random.choice(self.data.shape[0], self.n_way, False)
        selected_cls = np.arange(self.n_way)

        x_spt, y_spt, x_qry, y_qry = [], [], [], []
        for i, cur_class in enumerate(selected_cls):
            # select k_shot+k_query images from all images in the given class randomly
            selected_img = np.random.choice(self.n_img, self.k_shot + self.k_query, False)

            x_spt.append(self.data[cur_class][selected_img[:self.k_shot]])
            x_qry.append(self.data[cur_class][selected_img[self.k_shot:]])
            y_spt.append([i for _ in range(self.k_shot)])
            y_qry.append([i for _ in range(self.k_query)])

        # shuffle inside a batch
        perm = np.random.permutation(self.n_way * self.k_shot)

        x_spt = np.array(x_spt).reshape(self.n_way * self.k_shot, self.resize[0], self.resize[1], self.resize[2])[perm]
        y_spt = np.array(y_spt).reshape(self.n_way * self.k_shot)[perm]
        perm = np.random.permutation(self.n_way * self.k_query)
        x_qry = np.array(x_qry).reshape(self.n_way * self.k_query, self.resize[0], self.resize[1], self.resize[2])[perm]
        y_qry = np.array(y_qry).reshape(self.n_way * self.k_query)[perm]

        # Transformer needs a different input format: [n_timestamps, setsz, n_features] (i.e. [w, setsz, c*h])
        # [setsz, c, h, w] ==> [setsz, c*h, w], where w is n_timestamps
        x_spt = x_spt.astype(np.float32).reshape(setsz, self.resize[0] * self.resize[1], self.resize[2])
        x_qry = x_qry.astype(np.float32).reshape(qrysz, self.resize[0] * self.resize[1], self.resize[2])

        # [setsz, c*h, n_time] ==> [n_time, setsz, c*h]
        x_spt = np.transpose(x_spt, (2, 0, 1))
        x_qry = np.transpose(x_qry, (2, 0, 1))

        x_spt, y_spt, x_qry, y_qry = torch.from_numpy(x_spt), torch.from_numpy(y_spt), \
                                     torch.from_numpy(x_qry), torch.from_numpy(y_qry)

        return x_spt, y_spt, x_qry, y_qry


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

        sp_x = torch.from_numpy(np.transpose(np.array(sp_x), (2, 0, 1))).float()
        qr_x = torch.from_numpy(np.transpose(np.array(qr_x), (2, 0, 1))).float()
        sp_y, qr_y = torch.from_numpy(np.array(sp_y)).float(), torch.from_numpy(np.array(qr_y)).float()

        return sp_x, sp_y, qr_x, qr_y

    def __len__(self):
        return self.length
