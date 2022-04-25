import glob
import os
import random

src_dir = "/home/yangmuzhen/NAHAR/data/simulation_10s_by_people/"
ft_dir = "/home/yangmuzhen/NAHAR/data/radar_10s/env1/"

actions = [action for action in os.listdir(ft_dir) if action[0] != "p"]
seen_actions = ['jm', 'run', 'sit', 'squat', 'walk']
unseen_actions = ['rip', 'throw', 'wip']

"""
    Train: p5-p6, seen
    Val: p5-p6, unseen
"""

if __name__ == '__main__':
    random.seed(99)

    ft_support_data, ft_query_data, val_support_data, val_query_data = [], [], [], []
    for i, action in enumerate(seen_actions):
        files = sorted(glob.glob(src_dir + action + "/*.npy"))
        ft_support_data = ft_support_data + ["%s %d" % (file, i) for file in files[60:]]

        files = sorted(glob.glob(ft_dir + action + "/*.npy"))
        length = len(files)
        ft_query_data = ft_query_data + ["%s %d" % (file, i) for file in files[int(length * 0.2):int(length * 0.3)]]

    for i, action in enumerate(unseen_actions):
        files = sorted(glob.glob(src_dir + action + "/*.npy"))
        val_support_data = val_support_data + ["%s %d" % (file, 5 + i) for file in files[60:]]

        files = sorted(glob.glob(ft_dir + action + "/*.npy"))
        length = len(files)
        val_query_data = val_query_data + ["%s %d" % (file, 5 + i) for file in files[int(length * 0.2):int(length * 0.3)]]

    print(len(ft_support_data), len(ft_query_data), len(val_support_data), len(val_query_data))
    with open("./ft_support.txt", "w") as file:
        file.write("\n".join(ft_support_data))
    with open("./ft_query.txt", "w") as file:
        file.write("\n".join(ft_query_data))
    with open("./val_support.txt", "w") as file:
        file.write("\n".join(val_support_data))
    with open("./val_query.txt", "w") as file:
        file.write("\n".join(val_query_data))

    with open("./train.txt", "w") as file:
        file.write("\n".join(ft_query_data))
    with open("./val.txt", "w") as file:
        file.write("\n".join(val_query_data))

