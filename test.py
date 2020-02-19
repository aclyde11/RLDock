import argparse
import os
import time
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from config import config
from rldock.environments.lactamase import LactamaseDocking



if __name__ == '__main__':

    env = LactamaseDocking(config)

    scores = []
    for x_trans in tqdm(np.linspace(-8, 8, num=20)):
        for y_trans in np.linspace(-8, 8, num=20):
            for z_trans in np.linspace(-8, 8, num=20):
                env.cur_atom = env.translate_molecule(env.cur_atom, x_trans, y_trans, z_trans)
                s = env.get_oe_score()
                if s > 100:
                    s = 100
                else:
                    scores.append(s)
                env.cur_atom = env.translate_molecule(env.cur_atom, -1 * x_trans, -1 * y_trans, -1 * z_trans)

    plt.hist(scores)
    plt.show()
    env.close()
