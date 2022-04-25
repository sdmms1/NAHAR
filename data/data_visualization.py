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

    rows, cols = 2, 8

    act2d = {"rip": "/046.npy", "sit": "/061.npy", "walk": "/032.npy", "throw": "/016.npy",
             "jm": "/004.npy", "squat": "/001.npy", "wip": "/032.npy", "run": "/001.npy"}

    plt.figure(figsize=(14, 4))
    for d in [origin, simulation]:
        for i, action in enumerate(actions):
            file = random.choice(glob.glob(d + action + act2d[action]))
            print(file)
            img = np.load(file)
            img = cv2.resize(img, (200, 200))
            print(img.shape)

            plt.subplot(rows, cols, i + 1 + (0 if d == origin else 8))
            plt.axis(False)
            if d == origin:
                plt.title(action, fontsize=18)
            plt.imshow(img, cmap="gray")
    plt.tight_layout()
    plt.savefig("./stft comparison.png")
    # plt.show()

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

def check_simulation_with_radar():
    rows, cols = 3, 5

    simulation_path = "./simulation_10s_by_people/"
    radar_path = "./radar_10s/env1/"
    for act in os.listdir(simulation_path):
        for path in [radar_path, simulation_path]:
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
                _ = input()

def check_people_in_diff_env(actions):
    for action in actions:
        files = sorted(glob.glob("./radar_10s/env1/%s/*" % action))
        plt.figure(figsize=(8, 4))
        for i in range(4):
            file = files[15 * i]
            print(file)
            plt.subplot(2, 4, 1 + i)
            plt.axis(False)
            plt.imshow(np.load(file), cmap="gray")
            plt.subplot(2, 4, 5 + i)
            plt.axis(False)
            plt.imshow(np.load(file.replace("env1", "env2")), cmap="gray")
        plt.tight_layout()
        plt.show()
        plt.close()

if __name__ == '__main__':
    actions = [action for action in os.listdir("./radar_10s/env1/") if action[0] != "p"]

    # compare_simulation("./radar_10s/env1/", "./simulation_10s/", actions[4:])

    # check_first_people_in_simulation()

    # check_simulation_with_radar()

    check_people_in_diff_env(actions)