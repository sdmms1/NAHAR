import glob
import os
import random

data_dir = "/home/yangmuzhen/NAHAR/data/radar_10s/env5/"

actions = [action for action in os.listdir(data_dir) if action[0] != "p"]
seen_actions = ['jm', 'run', 'sit', 'squat', 'walk']
unseen_actions = ["rip", "wip", "drink", "lying"]
new_actions_in_test = ["bow", "throw"]

if __name__ == '__main__':

    data = []
    for i, action in enumerate(seen_actions):
        files = sorted(glob.glob(data_dir + action + "/*.npy"))
        data = data + ["%s %d" % (file, i) for file in files]

    print(len(data))
    with open("./5_class_eval.txt", "w") as file:
        file.write("\n".join(data))

    for i, action in enumerate(unseen_actions + new_actions_in_test):
        files = sorted(glob.glob(data_dir + action + "/*.npy"))
        data = data + ["%s %d" % (file, 5 + i) for file in files]

        print(len(data))
        with open("./%d_class_eval.txt" % (i + 6), "w") as file:
            file.write("\n".join(data))
