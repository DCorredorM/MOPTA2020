import numpy as np
import matplotlib.pyplot as plt


def normal(x):
	return np.array([-x[1],x[0]])

n=1000
x,y=12,12
o=np.array([12,12])

X=np.random.normal(x,10,n)
Y=np.random.normal(y,10,n)
XY=np.array(list(zip(X,Y)))


a=normal(o-np.array([np.mean(X),np.mean(Y)]))

A=list(filter(lambda x: a.dot(x)>=a.dot(o),XY))
B=list(filter(lambda x: a.dot(x)<=a.dot(o),XY))

plt.scatter([i[0] for i in A],[i[1] for i in A], c='blue')
plt.scatter([i[0] for i in B],[i[1] for i in B], c='green')
plt.show()











