# E3) (Exercício Computacional)
# Replique o experimento computacional denominado "Polynomial Curve Fitting"
# usado diversas vezes no livro texto (veja páginas 4 e 5 do livro, bem como
# Apêndice A).
# Faça:
#
# (a) Replique os resultados da Figura 1.4 e da Figura 1.6 para validar seu
# código (i.e., ter certeza de que ele está funcionando adequadamente);
#
# (b) Simule uma base de dados que não tenha relevância estatística, isto é, 
# que seja uma amostra que NÃO apresenta bem o todo (a população). Verifique
# alguns resultados experimentais para compreender a importância de ter uma 
# amostra relevante. Explique qual a relação entre o caso simulado e casos
# práticos envolvendo vetores de dimensão elevada.
# Dica: Para a simulação, ao invés de pegar dados igualmente espaçados no
# intervalo [0, 1], você pode forçar com que seus dados sejam amostrados apenas
# apenas no semiciclo positivo (ou apenas do negativo) do modelo gerador.
#
# (c) Simule uma base de dados em que 1 dos dados seja outlier. O que ocorre 
# com a curva vermelha, estimativa da curva verde (modelo gerador), neste caso?
# Dica: Para a simulação, você pode gerar seus dados de treinamento
# normalmente, igual foi feito no item (a), e ao final do processo escolher 1
# desses dados para atribuir um novo valor de target que seja completamente
# "maluco" (por exemplo, target = 10).
#
# Luiz Felipe da S. Coelho - luizfelipe.coelho@smt.ufrj.br
# July 2021
#

import os
import numpy as np
import matplotlib.pyplot as plt

# I am using ruindows, but if you want nice LaTeX charcters, uncomment this:
# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})

# Verify if Figures folder exists
print("\nThe current dir is: {:s}".format(os.getcwd()))
FOLDER = './Figures'
if os.path.isdir(FOLDER):
    print('Figures folder already exists.')
else:
    print('Creating folder...')
    os.mkdir(FOLDER)
    print('Figures folder created.')

# -----------------------------------------------------------------------------
#                                     (a)
# -----------------------------------------------------------------------------
print('\nStarting part (a)')

np.random.seed(4242)  # Set a random seed to reproduce Figures can be commented

# Figure 1.4:
# Definitions:
N = 10  # Number of samples
M = (0, 1, 3, 9)  # Different Orders
snr = 10  # Signal to noise ratio, dB

x = np.linspace(0, 1, N)  # Samples
y = np.sin(2*np.pi*x)  # Signal
n = np.random.randn(N)  # Noise
P_y = np.sum(y**2)/N  # Signal power
P_n = P_y*10**(-.1*snr)  # Noise power
t = y + np.sqrt(N*P_n/np.sum(n**2))*n  # Target

# Computing:
# Numpy's polyfit minimizes the squared error
p0 = np.poly1d(np.polyfit(x, t, M[0]))
p1 = np.poly1d(np.polyfit(x, t, M[1]))
p3 = np.poly1d(np.polyfit(x, t, M[2]))
p9 = np.poly1d(np.polyfit(x, t, M[3]))

# Plotting
xx = np.linspace(0, 1, 100)
tt = np.sin(2*np.pi*xx)

# Make it nice:
y_lim = (1.5*min((np.min(t), np.min(tt))), 1.5*max((np.max(t), np.max(tt))))
WIDTH = 8.9
GOLDEN_RATIO = (1 + np.sqrt(5))/2
HEIGHT = WIDTH/GOLDEN_RATIO

fig1 = plt.figure(figsize=(WIDTH, HEIGHT))
ax0 = fig1.add_subplot(221)
ax0.plot(x, t, 'o', mfc='none')
ax0.plot(xx, p0(xx))
ax0.plot(xx, tt)
ax0.text(0.75, 1, '$M=0$')
ax0.set_xlabel('$x$')
ax0.set_ylabel('$t$')
ax0.set_ylim(y_lim)
ax1 = fig1.add_subplot(222)
ax1.plot(x, t, 'o', mfc='none')
ax1.plot(xx, p1(xx))
ax1.plot(xx, tt)
ax1.text(0.75, 1, '$M=1$')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$t$')
ax1.set_ylim(y_lim)
ax2 = fig1.add_subplot(223)
ax2.plot(x, t, 'o', mfc='none')
ax2.plot(xx, p3(xx))
ax2.plot(xx, tt)
ax2.text(0.75, 1, '$M=3$')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$t$')
ax2.set_ylim(y_lim)
ax3 = fig1.add_subplot(224)
ax3.plot(x, t, 'o', mfc='none')
ax3.plot(xx, p9(xx))
ax3.plot(xx, tt)
ax3.text(0.75, 1, '$M=9$')
ax3.set_xlabel('$x$')
ax3.set_ylabel('$t$')
ax3.set_ylim(y_lim)

fig1.tight_layout()
print('Saving the equivalent of Figure 1.4...')
fig1.savefig(FOLDER+'/Figure_1.eps', format='eps', bbox_inches='tight')
print('Successfully saved in EPS format.')
fig1.savefig(FOLDER+'/Figure_1.png', format='png', bbox_inches='tight')
print('Successfully saved in PNG format.')

# Figure 1.6
# Definitions
N = (15, 100)  # Numer of samples
M = 9  # Order of the polynomial
snr = 10  # Signal to noise ratio, dB

# N = 15
x15 = np.linspace(0, 1, N[0])
y15 = np.sin(2*np.pi*x15)  # Signal
n = np.random.randn(N[0])  # Noise
P_y = np.sum(y15**2)/N[0]  # Signal power
P_n = P_y*10**(-.1*snr)  # Noise power
t15 = y15 + np.sqrt(N[0]*P_n/np.sum(n**2))*n  # Target

# N = 100
x100 = np.linspace(0, 1, N[1])
y100 = np.sin(2*np.pi*x100)  # Signal
n = np.random.randn(N[1])  # Noise
P_y = np.sum(y100**2)/N[1]  # Signal power
P_n = P_y*10**(-.1*snr)  # Noise power
t100 = y100 + np.sqrt(N[1]*P_n/np.sum(n**2))*n  # Target

# Computing
p15 = np.poly1d(np.polyfit(x15, t15, M))
p100 = np.poly1d(np.polyfit(x100, t100, M))

# Plotting

# Make it nice:
y_lim = (1.5*min((np.min(t15), np.min(t100), np.min(tt))),
         1.5*max((np.max(t15), np.max(t100), np.max(tt))))

fig2 = plt.figure(figsize=(WIDTH, HEIGHT/2))
ax0 = fig2.add_subplot(121)
ax0.plot(x15, t15, 'o', mfc='none')
ax0.plot(xx, p15(xx))
ax0.plot(xx, tt)
ax0.text(0.75, 1, '$N=15$')
ax0.set_xlabel('$x$')
ax0.set_ylabel('$t$')
ax0.set_ylim(y_lim)
ax1 = fig2.add_subplot(122)
ax1.plot(x100, t100, 'o', mfc='none')
ax1.plot(xx, p100(xx))
ax1.plot(xx, tt)
ax1.text(0.75, 1, '$N=100$')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$t$')
ax1.set_ylim(y_lim)

fig2.tight_layout()
print('Saving the equivalent of Figure 1.6...')
fig2.savefig(FOLDER+'/Figure_2.eps', format='eps', bbox_inches='tight')
print('Successfully saved in EPS format.')
fig2.savefig(FOLDER+'/Figure_2.png', format='png', bbox_inches='tight')
print('Successfully saved in PNG format.')

# -----------------------------------------------------------------------------
#                                   (b)
# -----------------------------------------------------------------------------
print('\nStarting part (b)')

# Figure 1.4:
# Definitions:
N = 10  # Number of samples
M = (0, 1, 3, 9)  # Different Orders
snr = 10  # Signal to noise ratio, dB

x = np.linspace(0, .5, N)  # Samples
y = np.sin(2*np.pi*x)  # Signal
n = np.random.randn(N)  # Noise
P_y = np.sum(y**2)/N  # Signal power
P_n = P_y*10**(-.1*snr)  # Noise power
t = y + np.sqrt(N*P_n/np.sum(n**2))*n  # Target

# Computing:
# Numpy's polyfit minimizes the squared error
p0 = np.poly1d(np.polyfit(x, t, M[0]))
p1 = np.poly1d(np.polyfit(x, t, M[1]))
p3 = np.poly1d(np.polyfit(x, t, M[2]))
p9 = np.poly1d(np.polyfit(x, t, M[3]))

# Plotting
xx = np.linspace(0, 1, 100)
tt = np.sin(2*np.pi*xx)

# Make it nice:
y_lim = (1.5*min((np.min(t), np.min(tt))), 1.5*max((np.max(t), np.max(tt))))

fig3 = plt.figure(figsize=(WIDTH, HEIGHT))
ax0 = fig3.add_subplot(221)
ax0.plot(x, t, 'o', mfc='none')
ax0.plot(xx, p0(xx))
ax0.plot(xx, tt)
ax0.text(0.75, 1, '$M=0$')
ax0.set_xlabel('$x$')
ax0.set_ylabel('$t$')
ax0.set_ylim(y_lim)
ax1 = fig3.add_subplot(222)
ax1.plot(x, t, 'o', mfc='none')
ax1.plot(xx, p1(xx))
ax1.plot(xx, tt)
ax1.text(0.75, 1, '$M=1$')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$t$')
ax1.set_ylim(y_lim)
ax2 = fig3.add_subplot(223)
ax2.plot(x, t, 'o', mfc='none')
ax2.plot(xx, p3(xx))
ax2.plot(xx, tt)
ax2.text(0.75, 1, '$M=3$')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$t$')
ax2.set_ylim(y_lim)
ax3 = fig3.add_subplot(224)
ax3.plot(x, t, 'o', mfc='none')
ax3.plot(xx, p9(xx))
ax3.plot(xx, tt)
ax3.text(0.75, 1, '$M=9$')
ax3.set_xlabel('$x$')
ax3.set_ylabel('$t$')
ax3.set_ylim(y_lim)

fig3.tight_layout()
print('Saving the equivalent of Figure 1.4...')
fig3.savefig(FOLDER+'/Figure_3.eps', format='eps', bbox_inches='tight')
print('Successfully saved in EPS format.')
fig3.savefig(FOLDER+'/Figure_3.png', format='png', bbox_inches='tight')
print('Successfully saved in PNG format.')

# Figure 1.6
# Definitions
N = (15, 100)  # Numer of samples
M = 9  # Order of the polynomial
snr = 10  # Signal to noise ratio, dB

# N = 15
x15 = np.linspace(0, .5, N[0])
y15 = np.sin(2*np.pi*x15)  # Signal
n = np.random.randn(N[0])  # Noise
P_y = np.sum(y15**2)/N[0]  # Signal power
P_n = P_y*10**(-.1*snr)  # Noise power
t15 = y15 + np.sqrt(N[0]*P_n/np.sum(n**2))*n  # Target

# N = 100
x100 = np.linspace(0, .5, N[1])
y100 = np.sin(2*np.pi*x100)  # Signal
n = np.random.randn(N[1])  # Noise
P_y = np.sum(y100**2)/N[1]  # Signal power
P_n = P_y*10**(-.1*snr)  # Noise power
t100 = y100 + np.sqrt(N[1]*P_n/np.sum(n**2))*n  # Target

# Computing
p15 = np.poly1d(np.polyfit(x15, t15, M))
p100 = np.poly1d(np.polyfit(x100, t100, M))

# Plotting

# Make it nice:
y_lim = (1.5*min((np.min(t15), np.min(t100), np.min(tt))),
         1.5*max((np.max(t15), np.max(t100), np.max(tt))))

fig4 = plt.figure(figsize=(WIDTH, HEIGHT/2))
ax0 = fig4.add_subplot(121)
ax0.plot(x15, t15, 'o', mfc='none')
ax0.plot(xx, p15(xx))
ax0.plot(xx, tt)
ax0.text(0.75, 1, '$N=15$')
ax0.set_xlabel('$x$')
ax0.set_ylabel('$t$')
ax0.set_ylim(y_lim)
ax1 = fig4.add_subplot(122)
ax1.plot(x100, t100, 'o', mfc='none')
ax1.plot(xx, p100(xx))
ax1.plot(xx, tt)
ax1.text(0.75, 1, '$N=100$')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$t$')
ax1.set_ylim(y_lim)

fig4.tight_layout()
print('Saving the equivalent of Figure 1.6...')
fig4.savefig(FOLDER+'/Figure_4.eps', format='eps', bbox_inches='tight')
print('Successfully saved in EPS format.')
fig4.savefig(FOLDER+'/Figure_4.png', format='png', bbox_inches='tight')
print('Successfully saved in PNG format.')


# -----------------------------------------------------------------------------
#                                   (c)
# -----------------------------------------------------------------------------
print('\nStarting part (c)')

# Figure 1.4:
# Definitions:
N = 10  # Number of samples
M = (0, 1, 3, 9)  # Different Orders
snr = 10  # Signal to noise ratio, dB

x = np.linspace(0, 1, N)  # Samples
y = np.sin(2*np.pi*x)  # Signal
n = np.random.randn(N)  # Noise
P_y = np.sum(y**2)/N  # Signal power
P_n = P_y*10**(-.1*snr)  # Noise power
t = y + np.sqrt(N*P_n/np.sum(n**2))*n  # Target
# Random choice of target to be changes
k = np.random.randint(N)
t[k] = 10  # Setting the k-th element to 10
print('The sample {:d} of the target vector is set to 10.'.format(k+1))

# Computing:
# Numpy's polyfit minimizes the squared error
p0 = np.poly1d(np.polyfit(x, t, M[0]))
p1 = np.poly1d(np.polyfit(x, t, M[1]))
p3 = np.poly1d(np.polyfit(x, t, M[2]))
p9 = np.poly1d(np.polyfit(x, t, M[3]))

# Plotting
xx = np.linspace(0, 1, 100)
tt = np.sin(2*np.pi*xx)

# Make it nice:
y_lim = (1.5*min((np.min(t), np.min(tt))), 1.5*max((np.max(t), np.max(tt))))

fig5 = plt.figure(figsize=(WIDTH, HEIGHT))
ax0 = fig5.add_subplot(221)
ax0.plot(x, t, 'o', mfc='none')
ax0.plot(xx, p0(xx))
ax0.plot(xx, tt)
ax0.text(0.75, 1, '$M=0$')
ax0.set_xlabel('$x$')
ax0.set_ylabel('$t$')
ax0.set_ylim(y_lim)
ax1 = fig5.add_subplot(222)
ax1.plot(x, t, 'o', mfc='none')
ax1.plot(xx, p1(xx))
ax1.plot(xx, tt)
ax1.text(0.75, 1, '$M=1$')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$t$')
ax1.set_ylim(y_lim)
ax2 = fig5.add_subplot(223)
ax2.plot(x, t, 'o', mfc='none')
ax2.plot(xx, p3(xx))
ax2.plot(xx, tt)
ax2.text(0.75, 1, '$M=3$')
ax2.set_xlabel('$x$')
ax2.set_ylabel('$t$')
ax2.set_ylim(y_lim)
ax3 = fig5.add_subplot(224)
ax3.plot(x, t, 'o', mfc='none')
ax3.plot(xx, p9(xx))
ax3.plot(xx, tt)
ax3.text(0.75, 1, '$M=9$')
ax3.set_xlabel('$x$')
ax3.set_ylabel('$t$')
ax3.set_ylim(y_lim)

fig5.tight_layout()
print('Saving the equivalent of Figure 1.4...')
fig5.savefig(FOLDER+'/Figure_5.eps', format='eps', bbox_inches='tight')
print('Successfully saved in EPS format.')
fig5.savefig(FOLDER+'/Figure_5.png', format='png', bbox_inches='tight')
print('Successfully saved in PNG format.')

# Figure 1.6
# Definitions
N = (15, 100)  # Numer of samples
M = 9  # Order of the polynomial
snr = 10  # Signal to noise ratio, dB

# N = 15
x15 = np.linspace(0, 1, N[0])
y15 = np.sin(2*np.pi*x15)  # Signal
n = np.random.randn(N[0])  # Noise
P_y = np.sum(y15**2)/N[0]  # Signal power
P_n = P_y*10**(-.1*snr)  # Noise power
t15 = y15 + np.sqrt(N[0]*P_n/np.sum(n**2))*n  # Target
# Random choice of target to be changed
k = np.random.randint(N[0])
t15[k] = 10  # Setting the k-th element to 10
print('The sample {:d} of the target vector is set to 10, for N=15.'.format(
    k+1))

# N = 100
x100 = np.linspace(0, 1, N[1])
y100 = np.sin(2*np.pi*x100)  # Signal
n = np.random.randn(N[1])  # Noise
P_y = np.sum(y100**2)/N[1]  # Signal power
P_n = P_y*10**(-.1*snr)  # Noise power
t100 = y100 + np.sqrt(N[1]*P_n/np.sum(n**2))*n  # Target
# Random choice of target to be changed
k = np.random.randint(N[1])
t100[k] = 10  # Setting the k-th element to 10
print('The sample {:d} of the target vector is set to 10, for N=100.'.format(
    k+1))

# Computing
p15 = np.poly1d(np.polyfit(x15, t15, M))
p100 = np.poly1d(np.polyfit(x100, t100, M))

# Plotting

# Make it nice:
y_lim = (1.5*min((np.min(t15), np.min(t100), np.min(tt))),
         1.5*max((np.max(t15), np.max(t100), np.max(tt))))

fig6 = plt.figure(figsize=(WIDTH, HEIGHT/2))
ax0 = fig6.add_subplot(121)
ax0.plot(x15, t15, 'o', mfc='none')
ax0.plot(xx, p15(xx))
ax0.plot(xx, tt)
ax0.text(0.75, 1, '$N=15$')
ax0.set_xlabel('$x$')
ax0.set_ylabel('$t$')
ax0.set_ylim(y_lim)
ax1 = fig6.add_subplot(122)
ax1.plot(x100, t100, 'o', mfc='none')
ax1.plot(xx, p100(xx))
ax1.plot(xx, tt)
ax1.text(0.75, 1, '$N=100$')
ax1.set_xlabel('$x$')
ax1.set_ylabel('$t$')
ax1.set_ylim(y_lim)

fig6.tight_layout()
print('Saving the equivalent of Figure 1.6...')
fig6.savefig(FOLDER+'/Figure_6.eps', format='eps', bbox_inches='tight')
print('Successfully saved in EPS format.')
fig6.savefig(FOLDER+'/Figure_6.png', format='png', bbox_inches='tight')
print('Successfully saved in PNG format.')

print('\nFigures are saved in the following folder:')
print('{}/Figures/'.format(os.getcwd()))

# plt.show()

