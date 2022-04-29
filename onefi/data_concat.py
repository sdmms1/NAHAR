import random

import numpy as np
import os
import cv2


def numf(i, l=3, end='.jpg'):
    fname = '%d' % i
    return '0' * (l - len(fname)) + fname + end


if __name__ == '__main__':
    src = "E:/ymz/DataProc/output/111_200x100_dataset_std_halfed/%s/"
    seen_cls = ["rip", "run", "sit", "squat", "walk"]
    unseen_cls = ["jm", "wip", "throw"]

    # print(np.load(src % "env1" + "jm" + "/%s" % numf(5, end=".npy")).shape)
    # exit()

    if not os.path.exists("./data"):
        os.mkdir("data")

    # Env1 for training
    n_cls = 5
    n_sample = 600
    result = np.zeros((n_cls, int(n_sample * 0.8), 1, 1, 200, 128))
    for c, cls in enumerate(seen_cls):
        print(c, cls)
        for i in range(int(n_sample * 0.8)):
            result[c, i, 0, 0] = np.load(src % "env1" + cls + "/%s" % numf(i + int(n_sample * 0.2) + 1, end=".npy"))
    np.save("./data/train_data.npy", result)

    # Env 1 Testing
    n_cls = 3
    result = np.zeros((n_cls, n_sample, 1, 1, 200, 128))
    for c, cls in enumerate(unseen_cls):
        print(c, cls)
        for i in range(n_sample):
            result[c, i, 0, 0] = np.load(src % "env1" + cls + "/%s" % numf(i + 1, end=".npy"))
    np.save("./data/env1_testing.npy", result)

    # Env 2, 3, 4 Testing
    n_cls = 8
    n_sample = 120
    for env in ["env%d" % e for e in range(2, 5)]:
        result = np.zeros((n_cls, n_sample, 1, 1, 200, 128))
        for c, cls in enumerate(unseen_cls + seen_cls):
            print(c, cls)
            for i in range(n_sample):
                result[c, i, 0, 0] = np.load(src % env + cls + "/%s" % numf(i + 1, end=".npy"))
        np.save("./data/%s_testing.npy" % env, result)

    for file in os.listdir("./data"):
        print(np.load("./data/" + file).shape)
