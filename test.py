import argparse
import os
import time
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from config import config
from rldock.environments.lactamase import LactamaseDocking
from mpl_toolkits.mplot3d import axes3d, Axes3D  # <-- Note the capitalization!

if __name__ == '__main__':

    env = LactamaseDocking(config)
    env.reset()
    num_search = 10
    scores_arr = np.zeros((num_search, num_search, num_search))

    x_pos = []
    y_pos = []
    z_pos = []
    scores = []
    scores_no_removed = []
    for i, x_trans in enumerate(tqdm(np.linspace(-8, 8, num=num_search))):
        for j, y_trans in enumerate(np.linspace(-8, 8, num=num_search)):
            for k, z_trans in enumerate(np.linspace(-8, 8, num=num_search)):
                env.cur_atom = env.translate_molecule(env.cur_atom, x_trans, y_trans, z_trans)
                s = env.get_raw_oe_score('Chemgauss4')
                if s > 100:
                    s = 100
                    scores_arr[i,j,k] = 100
                else:
                    x_pos.append(x_trans)
                    y_pos.append(y_trans)
                    z_pos.append(z_trans)
                    scores.append(s)
                    scores_arr[i,j,k] = s
                scores_no_removed.append(s)
                env.cur_atom = env.translate_molecule(env.cur_atom, -1 * x_trans, -1 * y_trans, -1 * z_trans)

    plt.hist(scores)
    plt.show()

    plt.hist(scores_no_removed)
    plt.show()

    fig = plt.figure(figsize=(15,5))

    plt.subplot(131)
    z = np.mean(scores_arr, axis=-1)
    plt.contourf(np.linspace(-8, 8, num=num_search), np.linspace(-8, 8, num=num_search), z)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.colorbar()

    plt.subplot(132)
    z = np.mean(scores_arr, axis=1)
    plt.contourf(np.linspace(-8, 8, num=num_search), np.linspace(-8, 8, num=num_search), z)
    plt.xlabel("X-axis")
    plt.ylabel("Z-axis")
    plt.colorbar()

    plt.subplot(133)
    z = np.mean(scores_arr, axis=0)
    plt.contourf(np.linspace(-8, 8, num=num_search), np.linspace(-8, 8, num=num_search), z)
    plt.xlabel("Y-axis")
    plt.ylabel("Z-axis")
    plt.colorbar()
    plt.show()
    env.close()
