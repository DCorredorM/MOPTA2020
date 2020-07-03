import numpy as np
import matplotlib.pyplot as plt

def pointGen(n):
	m=0
	return np.array(list(np.random.normal(loc=m, size=(int(n/2),2)))+list(np.random.normal(loc=m, size=(int(n/2),2))))

def covAuto(x):
	C=np.cov(x.transpose())
	lam,vecs=np.linalg.eig(C)
	return vecs.transpose()

def plot(x):
	X= [i for i,j in x]
	Y= [j for i,j in x]
	plt.scatter(X,Y)
	v1,v2=covAuto(x)

	X1=[i*v1[0] for i in range(-10,10)]
	Y1=[i*v1[1] for i in range(-10,10)]

	plt.plot(X1,Y1)
	X2=[i*v2[0] for i in range(-10,10)]
	Y2=[i*v2[1] for i in range(-10,10)]
	plt.plot(X2,Y2)





if __name__=='__main__':
	x=pointGen(100)
	
	plot(x)
	plt.show()
	
	






