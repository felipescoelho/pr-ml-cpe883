"""
E4 - Lista 2
Luiz Felipe da Silveira Coelho - luizfelipe.coelho@smt.ufrj.br
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def hist_density(x, delta):

    N = x.shape[-1]
    n_bins = int(1/delta)
    bin_count = np.zeros((n_bins,))

    for i in range(n_bins):
        inf_lim = i*delta
        sup_lim = (i+1)*delta
        for n in x:
            if inf_lim <= n <= sup_lim:
                bin_count[i] += 1

    bar_height = bin_count/(N*delta)
    bar_position = (np.arange(0, 1, delta)+np.arange(delta, 1+delta, delta))/2

    return bar_position, bar_height


def gaussian_kernel(x, h):

    N = x.shape[-1]
    k_n = np.zeros((N, N))
    ii = np.linspace(0, 1, N)

    for i in range(N):
        for n in range(N):
            k_n[i, n] = (1/np.sqrt(2*np.pi*h**2))*np.exp(
                -(np.linalg.norm(ii[i] - x[n])**2/(2*h**2)))
    return k_n.mean(axis=1)


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
    N = 50
    pi_1 = .3
    pi_2 = 1-pi_1
    mu_1 = .25
    mu_2 = .75
    sigma_1 = .125
    sigma_2 = .125

    # Generating Gaussians:
    n1 = sigma_1*np.random.randn(int(pi_1*N)) + mu_1
    n2 = sigma_2*np.random.randn(int(pi_2*N)) + mu_2

    x = np.hstack((n1, n2))

    # Histogram
    delta1 = 1/25
    bp1, bh1 = hist_density(x, delta1)
    delta2 = 1/12
    bp2, bh2 = hist_density(x, delta2)
    delta3 = 1/4
    bp3, bh3 = hist_density(x, delta3)

    # Gaussian Kernel
    h1 = .005
    k1 = gaussian_kernel(x, h1)
    h2 = .07
    k2 = gaussian_kernel(x, h2)
    h3 = .2
    k3 = gaussian_kernel(x, h3)

    # Plotting
    # Make it nice:
    WIDTH = 8.9
    GOLDEN_RATIO = (1 + np.sqrt(5))/2
    HEIGHT = WIDTH/GOLDEN_RATIO

    # Base curve
    xx = np.linspace(0, 1, 150)
    p = ((pi_1/(sigma_1*np.sqrt(2*np.pi)))*np.exp(-(xx-mu_1)**2/(2*sigma_1**2)) + 
         (pi_2/(sigma_2*np.sqrt(2*np.pi)))*np.exp(-(xx-mu_2)**2/(2*sigma_2**2)))
    
    fig1 = plt.figure('Histograma', figsize=(WIDTH, HEIGHT))
    ax0 = fig1.add_subplot(311)
    ax0.bar(x=bp1, height=bh1, width=delta1, edgecolor='black')
    ax0.plot(xx, p, color='tab:green')
    ax0.text(0.025, 4.3, '$\Delta={:.2f}$'.format(delta1))
    ax0.set_ylim((0, 5))
    ax0.set_xlim((0, 1))
    ax1 = fig1.add_subplot(312)
    ax1.bar(x=bp2, height=bh2, width=delta2, edgecolor='black')
    ax1.plot(xx, p, color='tab:green')
    ax1.text(0.025, 4.3, '$\Delta={:.2f}$'.format(delta2))
    ax1.set_ylim((0, 5))
    ax1.set_xlim((0, 1))
    ax2 = fig1.add_subplot(313)
    ax2.bar(x=bp3, height=bh3, width=delta3, edgecolor='black')
    ax2.plot(xx, p, color='tab:green')
    ax2.text(0.025, 4.3, '$\Delta={:.2f}$'.format(delta3))
    ax2.set_ylim((0, 5))
    ax2.set_xlim((0, 1))
    fig1.tight_layout()

    fig2 = plt.figure('Kernel Gaussiano', figsize=(WIDTH, HEIGHT))
    ax0 = fig2.add_subplot(311)
    ax0.plot(np.linspace(0, 1, N), k1)
    ax0.plot(xx, p, color='tab:green')
    ax0.text(0.025, 4.3, '$h={:.3f}$'.format(h1))
    ax0.set_ylim((0, 5))
    ax0.set_xlim((0, 1))
    ax1 = fig2.add_subplot(312)
    ax1.plot(np.linspace(0, 1, N), k2)
    ax1.plot(xx, p, color='tab:green')
    ax1.text(0.025, 4.3, '$h={:.2f}$'.format(h2))
    ax1.set_ylim((0, 5))
    ax1.set_xlim((0, 1))
    ax2 = fig2.add_subplot(313)
    ax2.plot(np.linspace(0, 1, N), k3)
    ax2.plot(xx, p, color='tab:green')
    ax2.text(0.025, 4.3, '$h={:.1f}$'.format(h3))
    ax2.set_ylim((0, 5))
    ax2.set_xlim((0, 1))
    fig2.tight_layout()
    
    fig1.savefig(FOLDER+'/E4_Fig1.eps', format='eps', bbox_inches='tight')
    fig1.savefig(FOLDER+'/E4_Fig1.png', format='png', bbox_inches='tight')
    fig2.savefig(FOLDER+'/E4_Fig2.eps', format='eps', bbox_inches='tight')
    fig2.savefig(FOLDER+'/E4_Fig2.png', format='png', bbox_inches='tight')

    plt.show()