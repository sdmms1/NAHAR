import glob
import os
import random

import numpy as np

def nf(i, length=3, end=''):
    result = str(i)
    result = "0" * max(0, length - len(result)) + result
    return result + end

if __name__ == '__main__':
    src = "./stft_simulation/"
    tgt = "./simulation_10s_by_people/"

    # with open("./simulation_data_list.txt", "r") as input_file:
    #     files = [e.strip() for e in input_file.readlines()]
    # print(len(files))
    #
    # unseen_people_files = [e for e in files if "01-17" in e]
    # print(len(unseen_people_files))
    #
    # actual_unseen_people_files = glob.glob("./stft_simulation/*/*01-17*")
    # print(len(actual_unseen_people_files))
    #
    # act_people_files = {}
    # for act in os.listdir(src):
    #     act_people_files[act] = []
    #
    # for i, e in enumerate(unseen_people_files):
    #     act = e.split("/")[-2]
    #     if not i % 5:
    #         act_people_files[act].append([])
    #     if e in actual_unseen_people_files:
    #         act_people_files[act][-1].append(e)
    #
    # for act in act_people_files:
    #     temp_dir = tgt + act
    #     if not os.path.exists(temp_dir):
    #         os.makedirs(temp_dir)
    #
    #     print(act)
    #     for pi, pl in enumerate(act_people_files[act]):
    #         print([e.split(" ")[-2] for e in pl])
    #         fidx = 0
    #         for i in range(len(pl)):
    #             file = pl[i]
    #             npy = np.load(file)
    #             assert npy.shape == (256, 765)
    #
    #             a, b, c = npy[:, :256], npy[:, 256:512], npy[:, -256:]
    #             np.save(os.path.join(temp_dir, nf(15 * pi + fidx + 1, end='.npy')), a)
    #             np.save(os.path.join(temp_dir, nf(15 * pi + fidx + 2, end='.npy')), b)
    #             np.save(os.path.join(temp_dir, nf(15 * pi + fidx + 3, end='.npy')), c)
    #             fidx += 3
    #
    #         while fidx < 15:
    #             file = random.choice(pl)
    #             print(fidx, file.split(" ")[-2])
    #             npy = np.load(file)
    #             assert npy.shape == (256, 765)
    #             start = random.randint(0, 764 - 256)
    #             np.save(os.path.join(temp_dir, nf(15 * pi + fidx + 1, end='.npy')), npy[:, start:start + 256])
    #             fidx += 1

    for act in os.listdir(src):
        temp_dir = tgt + act
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        files = [e for e in sorted(glob.glob(src + act + "/*")) if "01-17" not in e]
        print(act, len(files))
        # for e in files:
        #     print(e)
        # exit()

        for i, file in enumerate(files):
            npy = np.load(file)
            assert npy.shape == (256, 765)

            # print(os.path.join(temp_dir, nf(60 + 3 * i + 1, end='.npy')))
            # print(os.path.join(temp_dir, nf(60 + 3 * i + 2, end='.npy')))
            # print(os.path.join(temp_dir, nf(60 + 3 * i + 3, end='.npy')))

            a, b, c = npy[:, :256], npy[:, 256:512], npy[:, -256:]
            # print(a.shape, b.shape, c.shape)
            np.save(os.path.join(temp_dir, nf(60 + 3 * i + 1, end='.npy')), a)
            np.save(os.path.join(temp_dir, nf(60 + 3 * i + 2, end='.npy')), b)
            np.save(os.path.join(temp_dir, nf(60 + 3 * i + 3, end='.npy')), c)