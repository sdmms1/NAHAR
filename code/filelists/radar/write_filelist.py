import glob
import os
import random

data_dir = "/home/yangmuzhen/NAHAR/data/radar_10s/env1/"

actions = [action for action in os.listdir(data_dir) if action[0] != "p"]
seen_actions = ['jm', 'run', 'sit', 'squat', 'walk']
unseen_actions = ['rip', 'throw', 'wip']

"""
    Train: p5-p20, seen
    Val: p5-p20, unseen
    Test: p1-p4, seen, unseen
"""

if __name__ == '__main__':
    random.seed(99)

    train_data, val_data, seen_test_data, unseen_test_data = [], [], [], []
    for i, action in enumerate(seen_actions):
        files = sorted(glob.glob(data_dir + action + "/*.npy"))
        length = len(files)
        train_data = train_data + ["%s %d" % (file, i) for file in files[int(length * 0.2):]]
        seen_test_data = seen_test_data + ["%s %d" % (file, i) for file in files[:int(length * 0.2)]]

    for i, action in enumerate(unseen_actions):
        files = sorted(glob.glob(data_dir + action + "/*.npy"))
        length = len(files)
        val_data = val_data + ["%s %d" % (file, 5 + i) for file in files[int(length * 0.2):]]
        unseen_test_data = unseen_test_data + ["%s %d" % (file, 5 + i) for file in files[:int(length * 0.2)]]

    print(len(train_data), len(val_data), len(seen_test_data), len(unseen_test_data))
    with open("./train.txt", "w") as file:
        file.write("\n".join(train_data))
    with open("./val.txt", "w") as file:
        file.write("\n".join(val_data))
    with open("./test.txt", "w") as file:
        file.write("\n".join(seen_test_data + unseen_test_data))
    with open("./seen_test.txt", "w") as file:
        file.write("\n".join(seen_test_data))
    with open("./unseen_test.txt", "w") as file:
        file.write("\n".join(unseen_test_data))

