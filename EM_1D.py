import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import copy
import math
from scipy.stats import multivariate_normal
import random
from mpl_toolkits.mplot3d import Axes3D

N = 99     # Number of training samples
K = 3       # Number of groups
D = 1       # Number of dimensions


x1 =np.random.normal(50, 10,size=(N//3, D))
x2 =np.random.normal(100, 10,size=(N//3, D))
x3 =np.random.normal(200, 10,size=(N//3, D))

x = np.vstack([x1,x2,x3])

#%%

fig = plt.figure()
ax = fig.add_subplot(111)#, projection='3d')
for c,m,x_ in zip(['r','g','y'],['o','^','*'],[x1,x2,x3]):
    plt.plot(x_[:,0], np.zeros(len(x_)), c=c, marker=m,alpha=0.5)
ax.set_xlabel('xAxis')
ax.set_ylabel('yAxis')
#ax.set_zlabel('zAxis')
plt.show()

#%%
# initilizing gaussians
seedMu = np.random.randint(1, high = 250, size = (K,D)).astype(np.float64) 
seedCov =  [(np.random.randint(1,20)*np.eye(D)) for i in range(K)]
regCov = 1e-6 * np.eye(D)
seedGauss = [multivariate_normal(mu, cov) for mu,cov in zip(seedMu,seedCov)]
seedPi = np.asarray([1/K for i in range(K)])

alpha = np.zeros((N,K))
pdf = np.zeros((N,K))

log_likelihoods = []

#%%

iterations = 100
for itera in range(iterations):  
    #%%
    # E Step
    
    stop = False
    for i,gauss in enumerate(seedGauss):
       pdf[:,i] = gauss.pdf(x)
    
    pixPdf = np.dot(pdf, np.diag(seedPi))
    pixPdfSum = np.sum(pixPdf , axis = -1)
    
    alpha = np.divide(pixPdf.T,
                      pixPdfSum+1e-8).T 
                      
    #%%
    # M step
    print(itera)
    alphaSum = np.sum(alpha.T , axis = -1)
    
    mu = np.divide((np.dot(alpha.T,x)).T, alphaSum+1e-8).T
    
    muTemp = []
    cov = []
    for i in range(K):

        xSubMu = x-seedMu[i, :]
        alphaTemp = np.multiply(alpha[:,i],np.ones((N,D)).T)
        cov.append(np.divide(np.dot(xSubMu.T, np.multiply(alphaTemp.T, xSubMu)), np.sum(alpha[:, i])+1e-8))
        cov[-1] += regCov
        
    pi = alphaSum/N
    
    if (np.linalg.norm(mu - seedMu) < 1e-4):
        stop = True
    
    seedMu = copy.deepcopy(mu)
    seedCov = copy.deepcopy(cov)
    seedPi = copy.deepcopy(pi)
    seedGauss =  [multivariate_normal(mu_, cov_) for mu_,cov_ in zip(seedMu,seedCov)]
    
#%%

gmm = [multivariate_normal(mu, cov) for mu,cov in zip(seedMu,seedCov)]

surface = []
for i in range(0,255,3):
        surface.append(i)

surface = np.asarray(surface)

cpdf = np.zeros(surface.shape[0])

fig = plt.figure()
ax = fig.add_subplot(111)#, projection='3d')
for gauss,pi,c,m,x_ in zip(gmm,seedPi,['r','g','y'],['o','^','*'],[x1,x2,x3]):
    cpdf = np.add(cpdf, pi*gauss.pdf(surface))
    ax.scatter(surface, gauss.pdf(surface), c=c, marker=m,alpha=0.1)
    plt.plot(x_[:,0], np.zeros(len(x_)), c=c, marker=m,alpha=0.5)

ax.set_xlabel('xAxis')
ax.set_ylabel('yAxis')
#ax.set_zlabel('zAxis')
plt.show()

#%%
fig = plt.figure()
ax = fig.add_subplot(111)#, projection='3d')
for c,m,x_ in zip(['r','g','y'],['o','^','*'],[x1,x2,x3]):
#    ax.scatter(x[:,0], x[:,1], x[:,2], c=c, marker=m,alpha=0.5)
    plt.plot(x_[:,0], np.zeros(len(x_)), c=c, marker=m,alpha=0.5)
ax.scatter(surface, cpdf, c='b', marker='.',alpha=0.2)
ax.set_xlabel('xAxis')
ax.set_ylabel('yAxis')
#ax.set_zlabel('zAxis')
plt.show()