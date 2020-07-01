from main import *
import pickle
import time


p=master()
p.possibleDepots={17728, 16130, 15427, 16650, 19372, 17582, 15824, 18014, 18463}
p.h=4
p.posDisplays=p.calcPosDisp()
p.posDisplays=700
print(p.posDisplays)

SPS=[]
for id in range(5):		
	pickle_in = open(f'sp{id}.sp','rb')
	try:
		sp = pickle.load(pickle_in)
		SPS.append(sp)
	except:
		pass

p.SPS=SPS[:]
p.time_limit=10000000

'''
t=time.time()
p.Benders_algoMix(epsilon=0.01,read=False)
print(f'El timepo fue {time.time()-t}')

'''
sp=p.SPS[0]

#print(sp.Route_cost)

route=[15824,15711,16234,16212,16059,15716,15713,15920,15765,15824]#[15427,15360,15468,15610,15563,15920,15713,15716,15427]#[15427,15534,16695,15563,15958,15952,15920,15713,15716,15136,15427]
pat=zip(route[:-1],route[1:])

cost=0
time=0
for (i,j) in pat:
	cost+=p.c_mile*p.dist_m[i][j]
	time+=p.t_mile*p.dist_m[i][j]
	if time<2:
		time=2
	print(f'Tiempo a {j}: '+str(time*60))
print(cost)

