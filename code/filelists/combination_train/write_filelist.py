import glob
import os
import random

data_dir = "/home/yangmuzhen/NAHAR/data/"

actions = [action for action in os.listdir(data_dir) if action[0] != "p"]
seen_actions = ['jm', 'run', 'sit', 'squat', 'walk']
unseen_actions = ['rip', 'throw', 'wip']

"""
    Train: p5-p20, seen, simulation + p5-p6, seen, radar
    Val: p5-p20, unseen, simulation + p5-p6, unseen, radar
"""

if __name__ == '__main__':
    random.seed(99)

    train_data, val_data = [], []
    for i, action in enumerate(seen_actions):
        files = sorted(glob.glob(data_dir + "simulation_10s_by_people/" + action + "/*.npy"))
        train_data = train_data + ["%s %d" % (file, i) for file in files[60:]]
        files = sorted(glob.glob(data_dir + "radar_10s/env1/" + action + "/*.npy"))
        train_data = train_data + ["%s %d" % (file, i) for file in files[60:90]]

    for i, action in enumerate(unseen_actions):
        files = sorted(glob.glob(data_dir + "simulation_10s_by_people/" + action + "/*.npy"))
        val_data = val_data + ["%s %d" % (file, 5 + i) for file in files[:60]]
        files = sorted(glob.glob(data_dir + "radar_10s/env1/" + action + "/*.npy"))
        val_data = val_data + ["%s %d" % (file, 5 + i) for file in files[:60]]

    print(len(train_data), len(val_data))
    with open("./train.txt", "w") as file:
        file.write("\n".join(train_data))
    with open("./val.txt", "w") as file:
        file.write("\n".join(val_data))
