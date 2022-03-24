import os
import random
import cv2
import matplotlib.pyplot as plt
import glob

import numpy as np


def show_diff_actions(dir_path, actions):
    print(actions)

    rows, cols = (len(actions) - 1) // 4 + 1, 4

    plt.figure(figsize=(15, 10))
    for i, action in enumerate(actions):
        file = random.choice(glob.glob(dir_path + action + "/*.npy"))
        print(file)
        img = np.load(file)
        print(img.shape)

        plt.subplot(rows, cols, i + 1)
        plt.axis(False)
        plt.title(action, fontsize=18)
        plt.imshow(img, cmap="gray")
    plt.tight_layout()
    plt.show()


def compare_simulation(origin, simulation, actions):
    print(actions)

    rows, cols = (len(actions) * 2 - 1) // 4 + 1, 4

    plt.figure(figsize=(15, 10))
    for d in [origin, simulation]:
        for i, action in enumerate(actions):
            file = random.choice(glob.glob(d + action + "/*.npy"))
            print(file)
            img = np.load(file)
            print(img.shape)

            plt.subplot(rows, cols, i + 1 + (0 if d == origin else 4))
            plt.axis(False)
            plt.title(action, fontsize=18)
            plt.imshow(img, cmap="gray")
    plt.tight_layout()
    plt.show()


def check_first_people_in_simulation(path="./simulation_10s_by_people/"):
    rows, cols = 3, 5

    for act in os.listdir(path):
        files = sorted(glob.glob(path + act + "/*"))
        for i in range(4):
            plt.figure(figsize=(15, 10))
            for row in range(rows):
                for col in range(cols):
                    fidx = i * 15 + row * 5 + col
                    print(act, fidx)
                    img = np.load(files[fidx])

                    plt.subplot(rows, cols, row * 5 + col + 1)
                    plt.axis(False)
                    plt.title(fidx, fontsize=18)
                    plt.imshow(img, cmap="gray")

            plt.tight_layout()
            plt.show()




if __name__ == '__main__':
    actions = [action for action in os.listdir("./radar_10s/env1/") if action[0] != "p"]
    compare_simulation("./radar_10s/env1/", "./simulation_10s/", actions[:4])
    compare_simulation("./radar_10s/env1/", "./simulation_10s/", actions[4:])

    # check_first_people_in_simulation()