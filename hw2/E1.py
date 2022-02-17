"""
E1 - Lista 2
Luiz Felipe da S. Coelho - luizfelipe.coelho@smt.ufrj.br
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from random import choice
from math import gamma, factorial


def beta(mu, a, b):
    """Beta distribution"""
    return (gamma(a+b)/(gamma(a)*gamma(b)))*mu**(a-1)*(1-mu)**(b-1)


def binom(m, N, mu):
    """Binomial distribution"""
    comb = factorial(N)/(factorial(N-m)*factorial(m))
    return comb*mu**m*(1-mu)**(N-m)


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

    # General definitions
    L = 1000  # Length of my plane of search
    mu_axis = np.linspace(0, 1, L)  # mean for the priori (beta dist.)
    mu_obs = 0.7  # mean for the head toss
    N = 5  # Number of repetitions (tosses)
    # Let the heads be defined as 1.
    toss_space = np.hstack((np.ones((int(10*mu_obs),)),
                           np.zeros((10-int(10*mu_obs),))))  # The toss space


    toss = np.zeros((N,))
    for n in range(N):
        toss[n] = choice(toss_space)  # Making random choices in my space

    # Case 1:
    # Hyperparameters setup
    a = 1
    b = 1
    # Computation
    priori_1 = beta(mu_axis, a, b)
    post_1 = np.zeros((L, N))
    hyperparams_1 = np.zeros((2, N))
    for n in range(N):
        m = np.sum(toss[:n+1])  # Total no. of heads
        post_1[0:, n] = beta(mu_axis, a+m, b+(n+1-m))
        hyperparams_1[0:, n] = np.array((a+m, b+(n+1-m)), dtype=int)

    # Case 2:
    # Hyperparameters setup
    a = 2
    b = 2
    # Computation
    priori_2 = beta(mu_axis, a, b)
    post_2 = np.zeros((L, N))
    hyperparams_2 = np.zeros((2, N))
    for n in range(N):
        m = np.sum(toss[:n+1])  # Total no. of heads
        post_2[0:, n] = beta(mu_axis, a+m, b+(n+1-m))
        hyperparams_2[0:, n] = np.array((a+m, b+(n+1-m)), dtype=int)

    # Plotting
    # Make it nice:
    y_max = 1.10*max([np.max(post_2), np.max(post_1)])
    WIDTH = 8.9
    GOLDEN_RATIO = (1 + np.sqrt(5))/2
    HEIGHT = WIDTH/GOLDEN_RATIO

    fig1 = plt.figure('Caso 1', figsize=(WIDTH, HEIGHT))
    ax1 = fig1.add_subplot(231)
    ax1.plot(mu_axis, priori_1)
    ax1.set_xlabel('$\mu$')
    ax1.text(0.1, y_max*.90, '$a=1$')
    ax1.text(0.1, y_max*.80, '$b=1$')
    ax1.set_ylim((0, y_max))
    ax1.set_xlim((0, 1))
    ax2 = fig1.add_subplot(232)
    ax2.plot(mu_axis, post_1[0:, 0])
    ax2.set_xlabel('$\mu$')
    ax2.text(0.1, y_max*.90, '$a={:.0f}$'.format(hyperparams_1[0, 0]))
    ax2.text(0.1, y_max*.80, '$b={:.0f}$'.format(hyperparams_1[1, 0]))
    ax2.text(0.1, y_max*.70, '$N=1$')
    ax2.set_ylim((0, y_max))
    ax2.set_xlim((0, 1))
    ax3 = fig1.add_subplot(233)
    ax3.plot(mu_axis, post_1[0:, 1])
    ax3.set_xlabel('$\mu$')
    ax3.text(0.1, y_max*.90, '$a={:.0f}$'.format(hyperparams_1[0, 1]))
    ax3.text(0.1, y_max*.80, '$b={:.0f}$'.format(hyperparams_1[1, 1]))
    ax3.text(0.1, y_max*.70, '$N=2$')
    ax3.set_ylim((0, y_max))
    ax3.set_xlim((0, 1))
    ax4 = fig1.add_subplot(234)
    ax4.plot(mu_axis, post_1[0:, 2])
    ax4.set_xlabel('$\mu$')
    ax4.text(0.1, y_max*.90, '$a={:.0f}$'.format(hyperparams_1[0, 2]))
    ax4.text(0.1, y_max*.80, '$b={:.0f}$'.format(hyperparams_1[1, 2]))
    ax4.text(0.1, y_max*.70, '$N=3$')
    ax4.set_ylim((0, y_max))
    ax4.set_xlim((0, 1))
    ax5 = fig1.add_subplot(235)
    ax5.plot(mu_axis, post_1[0:, 3])
    ax5.set_xlabel('$\mu$')
    ax5.text(0.1, y_max*.90, '$a={:.0f}$'.format(hyperparams_1[0, 3]))
    ax5.text(0.1, y_max*.80, '$b={:.0f}$'.format(hyperparams_1[1, 3]))
    ax5.text(0.1, y_max*.70, '$N=4$')
    ax5.set_ylim((0, y_max))
    ax5.set_xlim((0, 1))
    ax6 = fig1.add_subplot(236)
    ax6.plot(mu_axis, post_1[0:, 4])
    ax6.set_xlabel('$\mu$')
    ax6.text(0.1, y_max*.90, '$a={:.0f}$'.format(hyperparams_1[0, 4]))
    ax6.text(0.1, y_max*.80, '$b={:.0f}$'.format(hyperparams_1[1, 4]))
    ax6.text(0.1, y_max*.70, '$N=5$')
    ax6.set_ylim((0, y_max))
    ax6.set_xlim((0, 1))
    fig1.tight_layout()

    fig2 = plt.figure('Caso 2', figsize=(WIDTH, HEIGHT))
    ax1 = fig2.add_subplot(231)
    ax1.plot(mu_axis, priori_2)
    ax1.set_xlabel('$\mu$')
    ax1.text(0.1, y_max*.90, '$a=2$')
    ax1.text(0.1, y_max*.80, '$b=2$')
    ax1.set_ylim((0, y_max))
    ax1.set_xlim((0, 1))
    ax2 = fig2.add_subplot(232)
    ax2.plot(mu_axis, post_2[0:, 0])
    ax2.set_xlabel('$\mu$')
    ax2.text(0.1, y_max*.90, '$a={:.0f}$'.format(hyperparams_2[0, 0]))
    ax2.text(0.1, y_max*.80, '$b={:.0f}$'.format(hyperparams_2[1, 0]))
    ax2.text(0.1, y_max*.70, '$N=1$')
    ax2.set_ylim((0, y_max))
    ax2.set_xlim((0, 1))
    ax3 = fig2.add_subplot(233)
    ax3.plot(mu_axis, post_2[0:, 1])
    ax3.set_xlabel('$\mu$')
    ax3.text(0.1, y_max*.90, '$a={:.0f}$'.format(hyperparams_2[0, 1]))
    ax3.text(0.1, y_max*.80, '$b={:.0f}$'.format(hyperparams_2[1, 1]))
    ax3.text(0.1, y_max*.70, '$N=2$')
    ax3.set_ylim((0, y_max))
    ax3.set_xlim((0, 1))
    ax4 = fig2.add_subplot(234)
    ax4.plot(mu_axis, post_2[0:, 2])
    ax4.set_xlabel('$\mu$')
    ax4.text(0.1, y_max*.90, '$a={:.0f}$'.format(hyperparams_2[0, 2]))
    ax4.text(0.1, y_max*.80, '$b={:.0f}$'.format(hyperparams_2[1, 2]))
    ax4.text(0.1, y_max*.70, '$N=3$')
    ax4.set_ylim((0, y_max))
    ax4.set_xlim((0, 1))
    ax5 = fig2.add_subplot(235)
    ax5.plot(mu_axis, post_2[0:, 3])
    ax5.set_xlabel('$\mu$')
    ax5.text(0.1, y_max*.90, '$a={:.0f}$'.format(hyperparams_2[0, 3]))
    ax5.text(0.1, y_max*.80, '$b={:.0f}$'.format(hyperparams_2[1, 3]))
    ax5.text(0.1, y_max*.70, '$N=4$')
    ax5.set_ylim((0, y_max))
    ax5.set_xlim((0, 1))
    ax6 = fig2.add_subplot(236)
    ax6.plot(mu_axis, post_2[0:, 4])
    ax6.set_xlabel('$\mu$')
    ax6.text(0.1, y_max*.90, '$a={:.0f}$'.format(hyperparams_2[0, 4]))
    ax6.text(0.1, y_max*.80, '$b={:.0f}$'.format(hyperparams_2[1, 4]))
    ax6.text(0.1, y_max*.70, '$N=5$')
    ax6.set_ylim((0, y_max))
    ax6.set_xlim((0, 1))
    fig2.tight_layout()

    fig1.savefig(FOLDER+'/E1_Fig1.eps', format='eps', bbox_inches='tight')
    fig1.savefig(FOLDER+'/E1_Fig1.png', format='png', bbox_inches='tight')
    fig2.savefig(FOLDER+'/E1_Fig2.eps', format='eps', bbox_inches='tight')
    fig2.savefig(FOLDER+'/E1_Fig2.png', format='png', bbox_inches='tight')


    plt.show()