"""
E2 - Lista 2
Luiz Felipe da Silveira Coelho - luizfelipe.coelho@smt.ufrj.br
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from random import choice



def bernoulli(mu, K):
    space = np.hstack((np.ones((int(1e3*mu),)), np.zeros((int(1e3*(1-mu)),))))
    x = np.zeros((K,))
    for k in range(K):
        x[k] = choice(space)
    return x


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

    # 1st case:
    N = 10  # Max number of iid variables
    K = int(1e5)
    unif2 = np.zeros((K, 2))
    unif10 = np.zeros((K, 10))
    for n in range(N):
        unif10[0:, n] = np.random.uniform(0, 1, K)
        if n == 0:
            unif1 = np.random.uniform(0, 1, K)
        if n < 2:
            unif2[0:, n] = np.random.uniform(0, 1, K)
    
    # Averages
    avg_unif2 = unif2.mean(axis=1)
    avg_unif10 = unif10.mean(axis=1)

    # 2nd case:
    N = 10  # Max number of iid variables
    K = int(1e5)
    mu = 0.3
    bern2 = np.zeros((K, 2))
    bern10 = np.zeros((K, 10))
    for n in range(N):
        bern10[0:, n] = bernoulli(mu, K)
        if n < 2:
            bern2[0:, n] = bernoulli(mu, K)
        if n == 0:
            bern1 = bernoulli(mu, K)

    # Averages
    avg_bern2 = bern2.mean(axis=1)
    avg_bern10 = bern10.mean(axis=1)

    # Plotting
    # Make it nice:
    WIDTH = 8.9
    GOLDEN_RATIO = (1 + np.sqrt(5))/2
    HEIGHT = WIDTH/GOLDEN_RATIO
    dict1 = {
        1:unif1, 
        2:avg_unif2, 
        3:avg_unif10,
    }

    dict2 = {
        1:bern1, 
        2:avg_bern2, 
        3:avg_bern10
    }

    lst = [1, 2, 10, 1, 2, 10]
    fig1 = plt.figure('1º Caso', figsize=(WIDTH, HEIGHT/2))
    for idx in range(1, 4):
        aux = dict1[idx]
        ax = fig1.add_subplot(1, 3, idx)
        ax.hist(aux, range=(0, 1), bins=20, density=True, edgecolor='black')
        ax.text(0.75, ax.get_ylim()[1], '$N = {}$'.format(lst[idx-1]))
        ax.set_ylim((0, ax.get_ylim()[1]*1.1))
        ax.set_xlim((0, 1))
    fig1.tight_layout()

    fig2 = plt.figure('2º Caso', figsize=(WIDTH, HEIGHT/2))
    for idx in range(1, 4):
        aux = dict2[idx]
        ax = fig2.add_subplot(1, 3, idx)
        ax.hist(aux, range=(0, 1), bins=20, density=True, edgecolor='black')
        ax.text(0.75, ax.get_ylim()[1], '$N = {}$'.format(lst[idx-1]))
        ax.text(0.75, ax.get_ylim()[1]*.9, '$\mu=0.3$')
        ax.set_ylim((0, ax.get_ylim()[1]*1.1))
        ax.set_xlim((0, 1))
    fig2.tight_layout()

    fig1.savefig(FOLDER+'/E2_Fig1.eps', format='eps', bbox_inches='tight')
    fig1.savefig(FOLDER+'/E2_Fig1.png', format='png', bbox_inches='tight')
    fig2.savefig(FOLDER+'/E2_Fig2.eps', format='eps', bbox_inches='tight')
    fig2.savefig(FOLDER+'/E2_Fig2.png', format='png', bbox_inches='tight')

    plt.show()