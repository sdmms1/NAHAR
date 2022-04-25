import glob
import os
import random


seen_actions = ['jm', 'run', 'sit', 'squat', 'walk']
unseen_actions = ["rip", "wip", "throw"]

if __name__ == '__main__':

    for env in ["env%d" % e for e in range(1, 6)] + ["simulation"]:
        if env != "simulation":
            data_dir = "/home/yangmuzhen/NAHAR/data/radar_10s/%s/" % env
        else:
            data_dir = "/home/yangmuzhen/NAHAR/data/%s_10s_by_people/" % env
        data = []
        for i, action in enumerate(seen_actions):
            files = sorted(glob.glob(data_dir + action + "/*.npy"))
            if env == "env1" or env == "simulation":
                files = files[:60]
            data = data + ["%s %d" % (file, i) for file in files]

        for i, action in enumerate(unseen_actions):
            files = sorted(glob.glob(data_dir + action + "/*.npy"))
            if env == "env1" or env == "simulation":
                files = files[:60]
            data = data + ["%s %d" % (file, 5 + i) for file in files]

        print(len(data))
        with open("./%s_eval.txt" % env, "w") as file:
            file.write("\n".join(data))
