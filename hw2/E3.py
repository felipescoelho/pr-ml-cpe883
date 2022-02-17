"""
E3 - Lista 2
Luiz Felipe da Silveira Coelho - luizfelipe.coelho@smt.ufrj.br
"""

import os
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':
    # Verify if Figures folder exists
    print("\nThe current dir is: {:s}".format(os.getcwd()))
    FOLDER = './Figures'
    if os.path.isdir(FOLDER):
        print('Figures folder already exists.')
    else:
        print('Creating folder...')
        os.mkdir(FOLDER)
        print('Figures folder created.')

    # Definitions:
    N = int(1e4)  # Number of variables
    K = int(1e4)

    # Computation
    x_n = np.zeros((K, N))
    for n in range(N):
        x_n[0:, n] = np.random.randn(K)

    # Averaging
    avg_x_10 = x_n[0:, 0:9].mean(axis=1)
    avg_x_100 = x_n[0:, 0:99].mean(axis=1)
    avg_x_250 = x_n[0:, 0:249].mean(axis=1)
    avg_x_1000 = x_n[0:, 0:999].mean(axis=1)
    avg_x_10000 = x_n.mean(axis=1)

    dict = {
        1: x_n[0:, 0],
        2: avg_x_10,
        3: avg_x_100,
        4: avg_x_250,
        5: avg_x_1000,
        6: avg_x_10000,
    }
    lst = [1, 10, 100, 250, 1000, 10000]

    # Plotting
    # Make it nice:
    WIDTH = 8.9
    GOLDEN_RATIO = (1 + np.sqrt(5))/2
    HEIGHT = WIDTH/GOLDEN_RATIO

    fig1 = plt.figure('LLN', figsize=(WIDTH, HEIGHT))
    for idx in range(1, 7):
        ax = fig1.add_subplot(2, 3, idx)
        ax.hist(dict[idx], bins=250, density=True)
        ax.text(ax.get_xlim()[0]*.9, ax.get_ylim()[1],
                '$N={}$'.format(lst[idx-1]))
        ax.set_ylim((0, ax.get_ylim()[1]*1.1))
        # ax.set_xlim((-1, 1))
    fig1.tight_layout()

    fig1.savefig(FOLDER+'/E3_Fig1.eps', format='eps', bbox_inches='tight')
    fig1.savefig(FOLDER+'/E3_Fig1.png', format='png', bbox_inches='tight')

    plt.show()