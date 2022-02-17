"""
E5 - Lista 2
Luiz Felipe da Silveira Coelho - luizfelipe.coelho@smt.ufrj.br
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def knn(K, x, classes):
    vect = np.empty((0, 2))
    kk = len(classes)  # No. of classes
    for k in range(kk):
        for n in classes[k]:
            vect = np.vstack((vect, np.array((k, np.linalg.norm(n-x)**2))))
    sorted = vect[vect[0:, 1].argsort()]
    K_k = sorted[0:K, 0]
    classif = np.zeros((kk,))
    for k in range(kk):
        classif[k] = len(K_k[K_k == k])/K

    return classif


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
    sigma_c1 = 1
    sigma_c2 = 1
    mu_c1 = -1
    mu_c2 = 1
    c1 = sigma_c1*np.random.randn(10) + mu_c1
    c2 = sigma_c2*np.random.randn(10) + mu_c2

    # computing
    c1_unknown = sigma_c1*np.random.randn(2) + mu_c1
    c2_unknown = sigma_c2*np.random.randn(2) + mu_c2


    c1_1 = np.zeros((3,))
    c1_2 = np.zeros((3,))
    c2_1 = np.zeros((3,))
    c2_2 = np.zeros((3,))
    K = (1, 9, 15)
    for k in range(3):
        # Unknown c1:
        px = knn(K[k], c1_unknown[0], (c1,c2))
        c1_1[k] = np.argmax(px)
        px = knn(K[k], c1_unknown[1], (c1,c2))
        c1_2[k] = np.argmax(px)
        # Unknown c2:
        px = knn(K[k], c2_unknown[0], (c1,c2))
        c2_1[k] = np.argmax(px)
        px = knn(K[k], c2_unknown[1], (c1,c2))
        c2_2[k] = np.argmax(px)

    # Plotting
    # Make it nice:
    WIDTH = 8.9
    GOLDEN_RATIO = (1 + np.sqrt(5))/2
    HEIGHT = WIDTH/GOLDEN_RATIO
    fig1 = plt.figure('1º Caso', figsize=(WIDTH, HEIGHT))
    for idx in range(1, 4):
        ax = fig1.add_subplot(1, 3, idx)
        ax.plot(np.zeros((10,)), c1, 'o', color='tab:red')
        if c1_1[idx-1] == 0:
            res1_1 = 's'
        else:
            res1_1 = 'x'
        ax.plot(c1_unknown[0], res1_1, color='tab:red', mfc='none')
        if c1_2[idx-1] == 0:
            res1_2 = 's'
        else:
            res1_2 = 'x'
        ax.plot(c1_unknown[1], res1_2, color='tab:red', mfc='none')
        ax.plot(np.zeros((10,)), c2, 'o', color='tab:blue')
        if c2_1[idx-1] == 1:
            res2_1 = 's'
        else:
            res2_1 = 'x'
        ax.plot(c2_unknown[0], res2_1, color='tab:blue', mfc='none')
        if c2_2[idx-1] == 1:
            res2_2 = 's'
        else:
            res2_2 = 'x'
        ax.plot(c2_unknown[1], res2_2, color='tab:blue', mfc='none')
        ax.text(0.025, ax.get_ylim()[1],
                '$K={}$'.format(K[idx-1]))
        ax.set_ylim((ax.get_ylim()[0], ax.get_ylim()[1]*1.15))
        ax.set_xticks([])
        ax.set_ylabel('$x$')
    fig1.tight_layout()
    
    fig1.savefig(FOLDER+'/E5_Fig1.eps', format='eps', bbox_inches='tight')
    fig1.savefig(FOLDER+'/E5_Fig1.png', format='png', bbox_inches='tight')

    plt.show()