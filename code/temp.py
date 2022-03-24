import numpy as np
from itertools import combinations

import torch

if __name__ == '__main__':
    arr = torch.from_numpy(np.arange(1, 17).reshape((4,2,2)))
    print(arr.shape)
    print(arr)

    for combination in combinations(range(4), 3):
        print(combination)
        print(arr[combination,])
