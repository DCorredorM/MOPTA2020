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
				pickle_in = open(f'Tsp{id}.sp','rb')
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

def solveSp(sp):

	print_log('\t\t',sp.id,'\n\t\t',sp.R,'\n\t\t',sp.Route_cost,'\n\t\t',sp.OptRi)
	m=sp.master_problem(relaxed=False)
	m.optimize()
	
	sp.export_results({i:m._z[i].x for i in m._z.keys()})
	pet=[i for i,p in m._z.items() if m._z[i].x>.5]
	routes=[sp.R[i] for i in pet]
	#self.plot_dem(sp.D)
	#self.plot_petals(routes,sp.H,pet=True)
	#plt.show()
	'''
	os.chdir('../Plots/Subproblems')
	if sp.id==0:delete_all_files(os.getcwd())
	p.plot_dem(sp.D)
	p.plot_petals(routes,sp.H,pet=False)
	plt.savefig(f'spNoCG_{sp.id}.png')
	#plt.show()
	plt.clf()			
	os.chdir('..')
	os.chdir('../Data')
	'''
	return m.objVal

	
	foi=sum(sp.Route_cost[r]*m._z[r].x for r in range(len(sp.R)))

def runBenders(h):
	'''
	Runs benders algo for h number of depots
	'''
	p.time_limit=100000000
	p.h=h
	p.createLog(h)
	UB,LB,x_hat,y_hat,ColGenCalls=p.BendersAlgoMix(epsilon=0.01,read=False,WS=WS)
	

	firstStage=val=sum(p.f['cost'].loc[i]*x_hat[i]+ p.c*y_hat[i] for i in  p.possibleDepots)

	secondStage=sum(solveSp(sp) for sp in p.SPS)/len(p.SPS)



	return UB,LB,ColGenCalls,firstStage,secondStage
	

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
	WS=True 	#WarmStart Strategy?
	#scenFile='Scenarios_robust_new.csv'		#Set of scenarios to solve.
	scenFile='Scenarios_robust_equal.csv'

	trained=False
	
	p.maxNumRoutes=1000				#Number of routes in each sp
	p.nItemptyRoutes=2000				#Period for cleaning set of routes
	p.ColGenTolerance=200
	Nota=f'Neutral Scenarios just trained\nNoCleaning Routes\nColGenTolerance:{p.ColGenTolerance}\nWarm start:{WS}\nTw strategy:{tw} with lunchbreake at beginning and end.'										#Note 

	compTimes=rPath+'/compTimes.txt'			#Computational times file 	
	write(compTimes,f'--------------------\nCorrida {datetime.datetime.now()}\n{Nota}')

	resultsFo=rPath+'/resultsFo.txt'			#Results times file 
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
		for s in p.SPS:
			s.save(f'Tsp{s.id}')
	
	write(compTimes,f'\t--------------\n\tSolving Times:')
	write(compTimes,f'\th\tTotal Time\tBenders Master\tSubproblem Master\tRoute Generator\tWarm Start')
	write(resultsFo,f'\t--------------\n\th\tValue Relaxed\tValue Total\t FS Cost\t SS Expected Cost\tNumber of calls Route Gen')

	#4. Solve firstStage with SPs trained	
	for h in range(3,len(p.possibleDepots)+1):
		#restore times
		p.totalBendersTime=0
		p.bendersMasterTime=0
		p.subProblemMasterTime=0
		p.routeGeneratorsTime=0
		p.warmStartTime=0

		for s in p.SPS:
			s.restartScores()
		SolvingTime=time.time()
		UB,LB,ColGenCalls,firstStage,secondStage=runBenders(h)
		SolvingTime=time.time()-SolvingTime

		write(compTimes,f'\t{h}\t{p.totalBendersTime}\t{p.bendersMasterTime}\t{p.subProblemMasterTime}\t{p.routeGeneratorsTime}\t{p.warmStartTime}')
		write(resultsFo,f'\t{h}\t{UB[-1]}\t{firstStage+secondStage}\t{firstStage}\t{secondStage}\t{ColGenCalls}')





	

	







