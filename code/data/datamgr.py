# This code is modified from https://github.com/facebookresearch/low-shot-shrink-hallucinate

from abc import abstractmethod

from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms

from .dataset import *

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

stft_transform = transforms.Compose([
                                    transforms.Resize((200, 200)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomVerticalFlip(),
                                    ])

class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass

class SimpleFewShotDataManager:
    def __init__(self, data_file):
        self.files = parse_data_file(data_file)
        self.transform = stft_transform
        print(len(self.files))

    def get_data_loader(self, fs_param, tep, group_by_people=True, single=True, same_people=False):
        if group_by_people:
            dataset = FewShotDatasetByPeople(self.files, **fs_param, tep=tep, transform=self.transform,
                                             single=single, same_people=same_people)
        else:
            dataset = SimpleFewShotDataset(self.files, **fs_param, tep=tep, transform=self.transform)
        return DataLoader(dataset, num_workers=4)

class CrossFewShotDataManager:
    def __init__(self, support_file, query_file):
        self.support_files = parse_data_file(support_file)
        self.query_files = parse_data_file(query_file)
        self.transform = stft_transform
        assert sorted(list(self.support_files.keys())) == sorted(list(self.query_files.keys()))
        print(len(self.support_files))

    def get_data_loader(self, fs_param, tep, single=True, same_people=False):
        dataset = CrossFewShotDatasetByPeople(self.support_files, self.query_files, **fs_param, tep=tep,
                                              transform=self.transform, single=single, same_people=same_people)
        return DataLoader(dataset, num_workers=4)

class FineTuneDataManager:
    def __init__(self, support_file, query_file):
        self.support_files = parse_data_file(support_file)
        self.query_files = parse_data_file(query_file)
        self.transform = stft_transform
        assert sorted(list(self.support_files.keys())) == sorted(list(self.query_files.keys()))
        print(len(self.support_files))

    def get_data_loader(self, fs_param, tep):
        dataset = FineTuneDataset(self.support_files, self.query_files, **fs_param, tep=tep, transform=self.transform)
        return DataLoader(dataset, num_workers=4)

class SystemDataManager:
    def __init__(self, support_file, query_file):
        self.support_files = parse_data_file(support_file)
        self.query_files = parse_data_file(query_file)
        self.transform = stft_transform
        assert sorted(list(self.support_files.keys())) == sorted(list(self.query_files.keys()))
        print(len(self.support_files))

    def get_data_loader(self, fs_param, tep, same_people=True):
        dataset = SystemEvaluationDataset(self.support_files, self.query_files, **fs_param, tep=tep,
                                          transform=self.transform, same_people=same_people)
        return DataLoader(dataset, num_workers=4)