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
b = normal(a)

A=list(filter(lambda x: a.dot(x)>=a.dot(o) and b.dot(x)>=b.dot(o) ,XY))
B=list(filter(lambda x: a.dot(x)>=a.dot(o) and b.dot(x)<=b.dot(o) ,XY))
C=list(filter(lambda x: a.dot(x)<=a.dot(o) and b.dot(x)<=b.dot(o) ,XY))
D=list(filter(lambda x: a.dot(x)<=a.dot(o) and b.dot(x)>=b.dot(o) ,XY))

plt.scatter([i[0] for i in A],[i[1] for i in A], c='blue')
plt.scatter([i[0] for i in B],[i[1] for i in B], c='green')
plt.scatter([i[0] for i in C],[i[1] for i in C], c='r')
plt.scatter([i[0] for i in D],[i[1] for i in D], c='orange')
plt.show()









