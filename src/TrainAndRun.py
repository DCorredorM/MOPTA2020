from main import *
import pickle
import pandas as pd




def Upload_Scenarios(file):
	#p.Demands=p.import_data('mopta2020_q2018.csv',h=0,names=['id']+list(range(1,366))).set_index('id')
	p.Demands=pd.read_csv(file,header=0,index_col=0)	
	id=0
	p.SPS=[]
	for i in p.Demands:
		di=p.Demands[i][p.Demands[i]>0]
		di_mat=p.dist_m.loc[list(di.index),list(di.index)]	
		
		try:
			pickle_in = open(f'sp{id}.sp','rb')
			sp = pickle.load(pickle_in)
			p.SPS.append(sp)		
			print(f'pikled {sp.id}')
		except:		
			p.SPS.append(sub_problem(id=f'{len(p.SPS)}',H=[],D=di,dm=p.dist_m,pos=p.pos))
			print(f'No pikle for {id}')
			pass
		#if id==1:
	#		p.SPS[1]=sub_problem(id=f'NoRob{len(p.SPS)-1}',H=[],D=di,dm=p.dist_m,pos=p.pos)
	#		print(f'No pikle for {id}')
		id+=1

def solve_depot(h):
	for sp in p.SPS:
		sp.H=[h]
		sp.y_hat={hh:0 for hh in p.possibleDepots if hh!=h}
		sp.y_hat[h]=20
		p.solveSpJava(sp)
		file_sp=open(f'sp{sp.id}.sp','wb')
		pickle.dump(sp, file_sp)

def FillEachDepot():
	'''
	For each depot solve a problem with only that depot oppended
	'''
	for h in p.possibleDepots:
		solve_depot(h)

def chech_training (sp,y_hat,H):
	'''
	Checks the training of a scenario
	In s: subproblem
	'''
	sp.y_hat=y_hat#{15427:6.0,15824:6.0,18014:9.0}#y_hat
	sp.H=H

	#FOi,λ,π=p.solveSpJava(sp)
	m=sp.master_problem()
	#print(sp.H)
	m.optimize()

	print('Obj: ',m.objVal)

	sp.z_hat={k:kk.x for k,kk in m._z.items()}
	#print(sp.z_hat)
	petals=[sp.R[i] for i,z in sp.z_hat.items() if z>0.5]

	p.plot_dem(sp.D,'green')
	p.plot_petals(petals=petals,h=sp.H,pet=False)
	plt.show()

def runBenders(h):
	global p
	p=master(h)
	Upload_Scenarios('Scenarios_robust_equal.csv')

	p.posDisplays=p.calcPosDisp()
	print('Este deeria ser el numero de its del relajado?; ',p.posDisplays)
	#p.posDisplays=700
	print(p.posDisplays)

	p.time_limit=10000000

	t=time.time()
	p.Benders_algoMix(epsilon=0.01,read=False)
	print(f'El timepo fue {time.time()-t}')
	f=open('tiempos.txt','a')
	f.write(f'{h}\t{time.time()-t}\n')
	f.close()

def calcCosts():
	mainDir=os.getcwd()
	os.chdir('../Results')
	resDir=os.getcwd()
	t=''
	depo=''
	for i in p.possibleDepots:
		depo+=str(i)+'\t'
		t+=str(p.f['cost'].loc[i])+'\t'
	print(depo)
	print(t)
	for h in range(2,10):
		y_hat={pot:0 for pot in p.possibleDepots}
		f=open(f'First_stage{h}.txt','r')
		for l in f:
			try:
				lista=l.split('\t')
				y_hat[int(lista[0])]=int(float(lista[1]))
			except:
				pass
		f.close()		
		print(''.join(str(y_hat[i])+'\t' for i in y_hat.keys()))

def solveDays(h,days):
	#p=master()
	#p.Demands=p.import_data('mopta2020_q2018.csv',h=0,names=['id']+list(range(1,366))).set_index('id')
	mainDir=os.getcwd()
	os.chdir('../Results')
	resDir=os.getcwd()
	
	y_hat={pot:0 for pot in p.possibleDepots}
	f=open(f'First_stage{h}.txt','r')	
	for l in f:
		try:
			lista=l.split('\t')
			y_hat[int(lista[0])]=int(float(lista[1]))
		except:
			pass
	f.close()
	os.chdir('./SecondStage')
	'''
	p.SPS=[]
	id=0
	for i in p.Demands:
		di=p.Demands[i][p.Demands[i]>0]
		di_mat=p.dist_m.loc[list(di.index),list(di.index)]	
		
		try:
			pickle_in = open(f'sp_day{id}.sp','rb')
			sp = pickle.load(pickle_in)
			p.SPS.append(sp)		
			#print(f'pikled {id}')
		except:		
			p.SPS.append(sub_problem(id=len(p.SPS),H=[],D=di,dm=p.dist_m,pos=p.pos))
			#print(f'No pikle for {id}')
			pass
		id+=1
	'''
	print(len(p.SPS))
	for day in days[::-1]:
		
		sp=p.SPS[day-1]
		print(f'Voy en {day} con demanda de {sum(sp.D)}')
		tspi=time.time()
		sp.y_hat=y_hat#{15427:6.0,15824:6.0,18014:9.0}#
		sp.H=list(sp.y_hat.keys())
		os.chdir(mainDir)
		
		ttt=time.time()
		FOi,λ,π=p.solveSpJava(sp)
		m=sp.master_problem()
		m.optimize()
		ttt=time.time()-ttt
		os.chdir(resDir)
		os.chdir('./SecondStage')

		file_sp=open(f'sp_day{day}.sp','wb')
		pickle.dump(sp, file_sp)


		sp.z_hat={k:kk.x for k,kk in m._z.items()}
		petals=[sp.R[i] for i,z in sp.z_hat.items() if z>0.5]
		p.plot_dem(sp.D,'green')
		p.plot_petals(petals=petals,h=sp.H,pet=False)
		plt.savefig(f'sp_day{day}.png')
		plt.clf()
		#plt.show()


		print(os.getcwd())
		os.chdir(resDir)
		os.chdir('./SecondStage')
		f=open(f'day{day}.txt','w')
		for r in petals:
			f.write(str(r)+'\n')

		f.close()

		NoAtendidos=0
		for d in sp.D.index:
			if m._v[d].x>0.5 :
				NoAtendidos+=sp.D[d]
		

		f=open(f'summary{h}.txt','a')
		f.write(f'{day}\t{m.objVal}\t{NoAtendidos}\t{sp.D.sum()}\t{ttt}'+'\n')
		f.close()

def solveScens(h):
	#p=master()
	#p.Demands=p.import_data('mopta2020_q2018.csv',h=0,names=['id']+list(range(1,366))).set_index('id')
	mainDir=os.getcwd()
	os.chdir('../Results')
	resDir=os.getcwd()
	
	y_hat={pot:0 for pot in p.possibleDepots}
	f=open(f'First_stage{h}.txt','r')	
	for l in f:
		try:
			lista=l.split('\t')
			y_hat[int(lista[0])]=int(float(lista[1]))
		except:
			pass
	f.close()
	os.chdir('./SecondStage')
	'''
	p.SPS=[]
	id=0
	for i in p.Demands:
		di=p.Demands[i][p.Demands[i]>0]
		di_mat=p.dist_m.loc[list(di.index),list(di.index)]	
		
		try:
			pickle_in = open(f'sp_day{id}.sp','rb')
			sp = pickle.load(pickle_in)
			p.SPS.append(sp)		
			#print(f'pikled {id}')
		except:		
			p.SPS.append(sub_problem(id=len(p.SPS),H=[],D=di,dm=p.dist_m,pos=p.pos))
			#print(f'No pikle for {id}')
			pass
		id+=1
	'''
	for sp in p.SPS:		
		print(f'Voy en {sp.id} con demanda de {sum(sp.D)}')
		tspi=time.time()
		sp.y_hat=y_hat#{15427:6.0,15824:6.0,18014:9.0}#
		sp.H=list(sp.y_hat.keys())
		os.chdir(mainDir)
		
		ttt=time.time()
		FOi,λ,π=p.solveSpJava(sp)
		m=sp.master_problem()
		m.optimize()
		ttt=time.time()-ttt
		os.chdir(resDir)
		os.chdir('./SecondStage')

		file_sp=open(f'spScen{sp.id}.sp','wb')
		pickle.dump(sp, file_sp)


		sp.z_hat={k:kk.x for k,kk in m._z.items()}
		petals=[sp.R[i] for i,z in sp.z_hat.items() if z>0.5]
		p.plot_dem(sp.D,'green')
		p.plot_petals(petals=petals,h=sp.H,pet=False)
		plt.savefig(f'spScen{sp.id}.png')
		plt.clf()
		#plt.show()


		print(os.getcwd())
		os.chdir(resDir)
		os.chdir('./SecondStage')
		f=open(f'spScen{sp.id}.txt','w')
		for r in petals:
			f.write(str(r)+'\n')

		f.close()

		NoAtendidos=0
		for d in sp.D.index:
			if m._v[d].x>0.5 :
				NoAtendidos+=sp.D[d]
		

		f=open(f'summary{h}.txt','a')
		f.write(f'{sp.id}\t{m.objVal}\t{NoAtendidos}\t{sp.D.sum()}\t{ttt}'+'\n')
		f.close()

def daysSorted():
	print(os.getcwd())
	os.chdir('/Users/davidcorredor/Universidad de los Andes/MOPTA - MOPTA/Implementation/Data')
	p.Demands=p.import_data('mopta2020_q2018.csv',h=0,names=['id']+list(range(1,366))).set_index('id')
	aux=p.import_data('mopta2020_q2019.csv',h=0,names=['id']+list(range(366+1,367+365))).set_index('id')
	p.Demands=pd.concat([p.Demands,aux],axis=1)
	id=100
	p.SPS=[]
	for i in p.Demands:
		di=p.Demands[i][p.Demands[i]>0]
		di_mat=p.dist_m.loc[list(di.index),list(di.index)]	
		#print(i)
		try:
			pickle_in = open(f'sp{id}.sp','rb')
			sp = pickle.load(pickle_in)
			p.SPS.append(sp)		
			print(f'pikled {id}')
		except:		
			p.SPS.append(sub_problem(id=id,H=[],D=di,dm=p.dist_m,pos=p.pos))
			#print(f'No pikle for {id}')
			pass
		id+=1

	
	sorted=p.Demands.sum(axis=0).sort_values()
	for s in p.Demands.columns:
		print(s,'\t',sum(p.Demands[s]))
	
	return list(sorted.index)
	
def solveSolOPt(h):
	mainDir=os.getcwd()
	os.chdir('../Results')
	resDir=os.getcwd()
	
	y_hat={pot:0 for pot in p.possibleDepots}
	f=open(f'First_stage{h}.txt','r')	
	for l in f:
		try:
			lista=l.split('\t')
			y_hat[int(lista[0])]=int(float(lista[1]))
		except:
			pass
	f.close()
	os.chdir('./SecondStage')
	print(len(p.SPS))

	for s,sp in enumerate(p.SPS):		
		tspi=time.time()
		sp.y_hat=y_hat#{15427:6.0,15824:6.0,18014:9.0}#
		sp.H=list(sp.y_hat.keys())
		os.chdir(mainDir)
		
		ttt=time.time()
		FOi,λ,π=p.solveSpJava(sp,2)
		m=sp.master_problem()
		m.optimize()
		ttt=time.time()-ttt
		os.chdir(resDir)
		os.chdir('./SecondStage')

		file_sp=open(f'sp{s}.sp','wb')
		pickle.dump(sp, file_sp)


		sp.z_hat={k:kk.x for k,kk in m._z.items()}
		petals=[sp.R[i] for i,z in sp.z_hat.items() if z>0.5]
		p.plot_dem(sp.D,'green')
		p.plot_petals(petals=petals,h=sp.H,pet=False)
		plt.savefig(f'sp{s}.png')
		#plt.show()


		print(os.getcwd())
		os.chdir(resDir)
		os.chdir('./SecondStage')
		f=open(f'SecondStage{s}.txt','w')
		for r in petals:
			f.write(str(r)+'\n')

		f.close()

		NoAtendidos=0
		for d in sp.D.index:
			if m._v[d].x>0.5 :
				NoAtendidos+=sp.D[d]
		

		f=open(f'summarySS{h}.txt','a')
		f.write(f'{s}\t{m.objVal}\t{NoAtendidos}\t{sp.D.sum()}\t{ttt}'+'\n')
		f.close()
	os.chdir(mainDir)

def chechAimms():
	
	R={}
	
	days=daysSorted()

	mainDir=os.getcwd()
	os.chdir('../Results')
	resDir=os.getcwd()
	h=5
	y_hat={pot:0 for pot in p.possibleDepots}
	f=open(f'First_stage{h}.txt','r')	
	for l in f:
		try:
			lista=l.split('\t')
			y_hat[int(lista[0])]=int(float(lista[1]))
		except:
			pass
	f.close()
	os.chdir(mainDir)

	f=open('resultsAIMMS.txt','r')
	day=int(f.readline().replace('#',''))
	R[day]=[]
	H=list(y_hat.keys())
	print(H)
	for l in f:
		if '#' in l:			
			day=int(l.replace('#',''))
			R[day]=[]
			if day==104: 
				print('aca')
		else:			
			num=l.replace('\n','')[:-1].split(',')
			vals={int(i):int(j) for i,j in map(lambda x:x.split(':'),num)}
			r=sorted(vals.keys(), key=lambda x: vals[x])
			r=[r[-1]]+r
			R[day].append(r)
			
			if r[0] not in H:
				print(f'Problema en ###{day}: \n {r}')
	f.close()

	day=days[0]
	day=list(R.keys())[0]
	print(day)
	print(sum(p.SPS[day-1].D))
	print(R[day])
	p.plot_dem(p.SPS[day -1].D)
	p.plot_petals(petals=R[day],h=[i for i in y_hat.keys() if y_hat[i]>0 ],pet=False)
	plt.show()

def consolidateResults():
	os.chdir('../Results/SecondStage')
	f=open('RoutesFina.txt','r')
	R={}

	day=int(f.readline().replace('#',''))
	R[day]=[]
	#H=list(y_hat.keys())
	H=[17728,16130,15427,15824,18014]
	read=True
	for l in f:
		if '#' in l:
			day=int(l.replace('#',''))			
			if day in R.keys():				
				read=False
			else:
				R[day]=[]
				read=True
		else:
			if read:
				num=l.replace('\n','')[:-1].split(',')
				vals={int(j):int(i) for i,j in map(lambda x:x.split(':'),num)}
				#r=sorted(vals.keys(), key=lambda x: vals[x])

				
				r=[vals[i] for i in  sorted(vals.keys())]
				r=[r[-1]]+r
				R[day].append(r)
				
				if r[0] not in H:
					print(f'Problema en ###{day}: \n {r}')
	f.close()

	f=open('2018-2019RoutingResults.txt','w')
	g=open('2018-2019RoutingResultsAIMMS.txt','w')
	for i in sorted(R.keys()):
		f.write(f'###{i}\n')
		g.write(f'###{i}\n')
		for r in R[day]:
			f.write(str(r)+'\n')			
			cont=1
			for k in r[1:]:
				g.write(f'{k}:{cont},')
				cont+=1
			g.write('\n')
	f.close()
	g.close()
	print(len(R.keys()))
	
def ReadPySolsNPrint(idis):
	print(os.getcwd())
	os.chdir('../Results/SecondStage')
	f=open(f'summaryPy.txt','r')
	results={}
	for l in f:
		lista=list(map(lambda x: round(float(x),2),l.replace('\n','').split('\t')))
		results[int(lista[0])]=lista[1:]
		print(int(lista[0]),results[lista[0]])
	f.close()
	routes={}
	#print(results.keys())
	os.chdir('./ResultsByDay')
	for i in results.keys():
	
		ff=open(f'day{idis[i]}.txt','r')
		routes[i]=[]
		#print(f'day{idis[i]}.txt')
		for j in ff:
			r=list(map(int,j.replace('\n','').replace('[','').replace(']','').split(',')))

			routes[i].append(r)
		#except:
		#	print(f'no {i}')
		#	pass
		ff.close()
	os.chdir('..')


	f=open('consolidateRoutes.txt','w')
	g=open('summaryAimmsFormat.txt','w')

	for d in results.keys():
		#dya\tfo\faltantes\demandaTot\nroutes\tiempo*100*60
		g.write(f'{d}\t{results[d][0]}\t{int(results[d][1])}\t{int(results[d][2])}\t{len(routes[d])}\t{int(results[d][3]*100)}\n')
		f.write(f'###{d}\n')
		for r in routes[d]:
			cont=1
			for k in r[1:]:
				f.write(f'{k}:{cont},')
				cont+=1
			f.write('\n')

	g.close()
	f.close()


	'''
	for l in idis.keys():
		
		pickle_in = open(f'sp{id}.sp','rb')
		sp = pickle.load(pickle_in)
		print(f'pikled {sp.id}')

		sp=p.SPS[day-1]
		print(f'Voy en {day} con demanda de {sum(sp.D)}')
		tspi=time.time()
		sp.y_hat=y_hat#{15427:6.0,15824:6.0,18014:9.0}#
		sp.H=list(sp.y_hat.keys())
		os.chdir(mainDir)
		
		ttt=time.time()
		FOi,λ,π=p.solveSpJava(sp)
		m=sp.master_problem()
		m.optimize()
		ttt=time.time()-ttt
		os.chdir(resDir)
		os.chdir('./SecondStage')

		file_sp=open(f'sp_day{day}.sp','wb')
		pickle.dump(sp, file_sp)
		

		sp.z_hat={k:kk.x for k,kk in m._z.items()}
		petals=[sp.R[i] for i,z in sp.z_hat.items() if z>0.5]
		p.plot_dem(sp.D,'green')
		p.plot_petals(petals=petals,h=sp.H,pet=False)
		plt.savefig(f'sp_day{day}.png')
		plt.clf()
		#plt.show()
		
	
		print(os.getcwd())
		os.chdir(resDir)
		os.chdir('./SecondStage')
		f=open(f'day{day}.txt','r')
		R=[]
		for r in f:
			R.append()
			

		f.close()
		'''

if __name__=='__main__':
	p=master()
	#Upload_Scenarios('Scenarios_robust_new.csv')

	#chechAimms()

	#Uploads the scenarios (trained)

 #solveSolOPt(8)

	#p.solveSpJava(sp)

	'''
	Fill each depot with routes.
	'''
	#FillEachDepot()

	'''
	h=list(p.possibleDepots)[2]
	y_hat={hh:0 for hh in p.possibleDepots if hh!=h}
	y_hat[h]=20
	H=[h]
	chech_training(sp,y_hat,H)
	'''
	#calcCosts()

	#solveDays(h=5,days=[3])
	#for sp in p.SPS:
		#print(len(sp.D),'\t',sum(sp.D))

	#for h in [4,5]:
	#	runBenders(h)

	'''
	days=daysSorted()
	#done=[366,104,449,477,119,665,286,91,475,469,126,532,498,90,301,400,293,327,652,725,294,473,463,659,50,252,470,476,729,21,701,298,478,70,474,505,83,462,464,491,271,726,272,125,700,496,76,673,645,490,644,117,442,687,504,670,42,285,295,105,691,480,727,109,676,728,96,658,64,668,113,61,284,300,699,27,309,359,118,103,484,519,322,115,307,308,680,77,122,455,89,112,501,111,529,22,328,698,290,57,456,468,653,664,650,63,671,54,132,371,265,49,336,62,60,101,497,51,709,438,486,81,329,666,420,92,93,287,84,481,48,499,56,500,139,311,489,138,657,461,513,672,97,707,488,654,660,483,690,303,693,78,266,450,503,251,526,602,86,273,495,662,692,121,651,466,472,677,291,722,358,454,708,334,435,209,321,459,350,444,493,278,312,434,696,98,633,656,669,110,69,539,482,663,617,20,349,367,127,448,686,452,357,280,379,427,378,437,46,661,485,502,643,270,306,375,335,369,467,133,315,258,414,509,675,292,624,28,610,611,648,649,714,310,460,302,372,694,731,80,116,471,35,106,155,254,124,506,515,678,402,508,114,453,59,75,131,210,82,176,446,299,354,688,705,360,10,674,74,174,507,94,120,175,445,697,71,479,517,140,305,439,364,428,516,47,68,647,415,401,492,512,533,85,102,129,689,142,249,274,465,605,721,720,289,26,326,410,87,313,451,541,634,361,269,609,724,608,79,458,55,368,637,65,173,348,706,408,527,723,330,679,53,616,635,41,530,638,128,279,374,443,520,525,88,325,413,436,244,569,716,39,441,356,95,667,154,531,681,403,406,447,288,340,695,189,511,107,123,524,314,323,421,715,588,99,353,411,518,629,100,34,603,108,259,380,23,385,655,296,370,130,487,52,268,147,255,342,425,8,399,704,528,393,419,685,684,362,623,440,570,40,188,554,231,347,646,534,457,494,422,710,11,267,365,32,355,426,304,423,58,521,560,256,318,595,162,38,277,136,580,581,536,135,204,389,345,576,391,213,146,159,15,205,243,166,178,225,149,183,2,395,397,599,545,557,226,227,18,167,263,16,200,548,186,5,552,563,240,181,194,192,591,169,564,239,396,218,220,17,4,549,180,565,566]


	#NotDone=[407,25,376,514,682,141,283,13,297,630,72,9,236,382,275,153,417,636,601,712,384,343,416,320,332,234,156,245,316,702,582,373,143,424,631,73,383,538,352,405,144,160,30,386,202,250,537,281,157,44,137,66,542,717,211,398,730,642,182,224,246,338,418,351,632,713,31,596,214,606,237,177,535,45,333,575,37,33,276,158,317,579,212,719,208,510,165,388,344,217,242,394,604,172,377,683,145,203,607,134,346,161,324,14,390,148,621,341,625,235,381,561,29,641,6,223,703,547,551,613,618,164,207,216,36,574,12,553,67,639,339,433,404,577,24,253,363,540,151,392,248,337,568,215,578,718,626,543,199,7,412,19,43,522,523,593,409,152,711,185,3,544,257,195,612,222,230,319,201,594,572,587,232,171,238,150,432,559,615,585,430,583,170,198,219,163,233,241,555,627,628,431,600,184,556,1,190,206,586,260,196,264,187,429,262,331,589,546,619,562,571,614,282,221,640,620,168,229,622,558,584,247,387,550,193,567,191,598,197,592,597,228,573,179,590,261]

	n=int(round(len(NotDone)/3,0))
	David=NotDone[:n]
	Alfa=NotDone[n:2*n]
	VM=NotDone[2*n:]

	#print(David)
	#print(Alfa)
	#print(VM)
	'''

	#solveDays(h=5,days=David)


	'''
	l=[685,393,623,362,440,570,188,554,40,231,646,347,534,494,457,422,710,365,11,267,355,32,426,304,423,58,560,521,318,595,256,407,25,376,682,514,283,141,297,630,13,72,382,236,9,162,275,417,153,636,38,601,712,277,384,416,343,320,332,234,156,245,316,136,702,581,582,373,424,631,143,73,538,684,383,352,405,144,160,386,30,536,202]
	vals=[851,961,1187,653,677,1228,1118,981,992,1332,682,865,784,522,653,817,635,753,1030,1342,914,692,836,565,616,599,1326,798,794,1237,499,1001,509,877,761,788,770,905,480,1104,859,737,676,1072,1368,933,524,689,1257,661,802,1234,612,645,976,775,997,872,590,1057,1039,1103,657,600,568,1018,934,807,796,1104,628,663,924,784,803,759,709,726,869,880,751,740,1099]

	for i in range(len(l)):
		if sum(p.Demands[l[i]-3])==vals[i]:
			print(l[i]-3)
		elif sum(p.Demands[l[i]-4])==vals[i]:
			print(l[i]-4)
	'''




	'''
	consolidate routes
	'''

	#Python->Aiims
	#idis={345:344,166:165,213:212,159:158,577:575,178:177,276:276,33:33,37:37,333:333,45:45,177:177,237:237,607:606,214:214,543:542,703:702,713:712,683:682,377:376,152:152,522:522,43:43,19:19,199:199,579:578,215:215,337:337,248:248,151:151,363:363,253:253,24:24,578:577,339:339,67:67,12:12,36:36,207:207,164:164,223:223,6:6,235:235,626:625,14:14,14:14}
	#ReadPySolsNPrint(idis)
	#Merge documents
	#consolidateResults()


	'''
	Neurtal risk scenarios
	'''

	print(os.getcwd())
	p=master()
	#Unload scenarios
	Upload_Scenarios('Scenarios_robust_equal.csv')

	#Chech basic traning traninig
	h=list(p.possibleDepots)[2]
	y_hat={hh:20 for hh in p.possibleDepots if hh!=h}
	y_hat[h]=20
	H=[h]

	for s in p.SPS:
		print(len(s.R))

	calcCosts()

	#for h in [2,3,4,5,6,7,8,9]:
	#	solveScens(h)

	'''
	calcCosts()

	for h in range(3,10):
		solveSolOPt(h)
	'''




