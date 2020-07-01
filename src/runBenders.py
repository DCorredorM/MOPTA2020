from main import *
import pickle

p=master()
p.possibleDepots={17728, 16130, 15427, 16650, 19372, 17582, 15824, 18014, 18463}
p.h=5
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
p.Benders_algo(read=False)