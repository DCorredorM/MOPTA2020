from main1 import *
import time
import os
import datetime



def Upload_Scenarios(file,trained=False):
	#p.Demands=p.import_data('mopta2020_q2018.csv',h=0,names=['id']+list(range(1,366))).set_index('id')
	p.Demands=pd.read_csv(file,header=0,index_col=0)	
	id=0
	p.SPS=[]
	pik=''
	Npik=''
	for i in p.Demands:
		di=p.Demands[i][p.Demands[i]>0]
		if trained:
			try:
				pickle_in = open(f'sp{id}.sp','rb')
				sp = pickle.load(pickle_in)
				p.SPS.append(sp)
				sp.master=p		
				#print(f'pikled {sp.id}')
				pik+=f'{sp.id}, '
			except:		
				p.SPS.append(sub_problem(id=f'{len(p.SPS)}',H=[],D=di,dm=p.dist_m,pos=p.pos))
				#print(f'No pikle for {id}')
				Npik+=f'{id}, '
				pass
		else:
			p.SPS.append(sub_problem(id=f'{len(p.SPS)}',H=[],D=di,dm=p.dist_m,pos=p.pos,master=p))
			#print(f'No pikle for {id}')
			Npik+=f'{id}, '

		id+=1
	print(f'Pickled: {pik[:-2]}')
	print(f'Not Pickled: {Npik[:-2]}')

def solveForYHat(y_hat):
	'''
	Solves all Subproblems for the y_hat given
	'''
	
	for sp in p.SPS:
		#Impose y_hat
		sp.H=[i for i,j in y_hat.items() if j>0]
		sp.y_hat=y_hat
		#Solve
		p.solveSpJava(sp,tw=tw)
		file_sp=open(f'sp{sp.id}.sp','wb')
		pickle.dump(sp, file_sp)

def TrainSPs():
	'''
	For each depot solve a problem with only that depot oppended
	'''
	#Train pairs
	TrainPairs()

	#Train all opened

def TrainPairs():
	'''
	Trains by pairs. Select a depot an trains it with the farthest possible depoto
	'''	
	Cand=p.possibleDepots.copy()
	pairs=[]
	while Cand:
		i=Cand.pop()
		try:
			j=max(Cand,key=lambda x: p.dist_m[i][x])
			Cand.remove(j)
			pairs.append([i,j])
		except:
			pairs.append([i])

	for l in pairs:
		y_hat={i:30 if i in l else 0 for i in p.possibleDepots}
		solveForYHat(y_hat)

def TrainOpened():
	'''
	Solves with all depots opened
	'''
	y_hat={i:30 for i in p.possibleDepots}
	solveForYHat(y_hat)

def write(file:str,s:str):
	f=open(file,'a')
	f.write(s+'\n')
	f.close()

def runBenders(h):
	'''
	Runs benders algo for h number of depots
	'''
	p.time_limit=1000000000
	p.h=h
	p.createLog(h)
	UB,LB,x_hat,y_hat,ColGenCalls=p.BendersAlgoMix(epsilon=0.01,read=False)
	return UB,LB,x_hat,y_hat,ColGenCalls
	

if __name__=='__main__':
	#1. Create master problem
	p=master()	

	'''
	Save important paths
	'''
	os.chdir('../Data')
	dPath=os.getcwd()							#Data folder
	os.chdir('../Results/ResultsFSRobust')		#Folder for results.
	rPath= os.getcwd()							#Results folder

	#Constants:	
	################################################################
	tw=True		#TimeWindow strategy?
	scenFile='Scenarios_robust_new.csv'		#Set of scenarios to solve.

	trained=True
	
	p.maxNumRoutes=100				#Number of routes in each sp
	p.nItemptyRoutes=2				#Period for cleaning set of routes

	Nota=f'Just trained, Tw strategy and cleaning routes with {p.maxNumRoutes} for the max number of routes and, {p.nItemptyRoutes} for the number of its for cleaning'										#Note 

	compTimes=rPath+'/compTimes.txt'			#Computational times file 	
	write(compTimes,f'--------------------\nCorrida {datetime.datetime.now()}\n{Nota}')

	resultsFo=rPath+'/resultsFo.txt'			#Computational times file 
	write(resultsFo,f'--------------------\nCorrida {datetime.datetime.now()}\n{Nota}')
	
	os.chdir(dPath)					#Change dir to Data

	################################################################

	
	#2. Load Scenarios
	Upload_Scenarios(scenFile,trained=trained)

	
	#3. Train Scenarios
	if not trained:
		trainingTime=time.time()
		TrainSPs()
		trainingTime=time.time()-trainingTime
		write(compTimes,f'Training time: {trainingTime}')
		write(compTimes,f'\t--------------\n\tSolving Times:')
		write(resultsFo,f'\t--------------\n\th\tValue\tNumber of calls Route Gen')
		for s in p.SPS:
			s.save(f'Tsp{s.id}')
	
	#4. Solve firstStage with SPs trained	
	for h in range(7,len(p.possibleDepots)+1):
		for s in p.SPS:
			s.restartScores()
		SolvingTime=time.time()
		UB,LB,x_hat,y_hat,ColGenCalls=runBenders(h)
		SolvingTime=time.time()-SolvingTime
		write(compTimes,f'\t{h}: {SolvingTime}')
		write(resultsFo,f'\t{h}\t{UB[-1]}\t{ColGenCalls}')






	







