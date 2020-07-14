from main import *
import time

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
	TrainPairs()

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



if __name__=='__main__':
	
	#Constants:
	tw=True		#TimeWindow strategy?

	
	#1. Create master problem
	p=master()

	#2. Load Scenarios

	file='Scenarios_robust_new.csv'
	Upload_Scenarios(file)

	#3. Train Scenarios
	trainingTime=time.time()
	TrainSPs()
	trainingTime=time.time()-trainingTime






