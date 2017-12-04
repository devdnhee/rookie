import numpy as np
import math
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1./(1+np.exp(-x))

def relu(x):
    return np.maximum(x,np.zeros(x.shape[0]))

x=np.arange(-6,6,0.01)

plt.plot(x,relu(x),'k-')
axes=plt.gca()
axes.set_xlim([-1.2,1.2])
axes.set_ylim([-0.2,1.2])
plt.axhline(y=0, color='k', linestyle='-',lw=0.6)
plt.axvline(x=0, color='k', linestyle='-',lw=0.6)
plt.xlabel(r'$a$')
plt.ylabel(r'$f(a)$')
plt.show()
