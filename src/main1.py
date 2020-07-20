from gurobipy import *
import pandas as pd
import os 

import networkx as nx
import random
import numpy as np
import math
from math import exp
import time
import pickle


class master():
	"""docstring fox master"""
	def __init__(self,h=3,load=True):
		#Number of depots to open |H|.
		self.h=h

		self.createLog(h)

		#Maximum number of vehicles per depot.
		self.M=20

		#Cost of positioninng a vehicle
		self.c=30

		#Route time limit
		self.route_time_limit=9

		#Capacity of each truck
		self.cap=60

		#p(master, self).__init__()
		self.data_path = os.chdir("../Data") #Data path

		self.t_mile=1/40
		self.c_mile=0.7
		#Time limit for the Benders Algo
		self.time_limit=1000

		#Clusters
		self.Clusters=[]
		#self.warmstart=self.warm_start()
		
		#Optimality gap for Benders algorithm
		self.epsilon=0.01


		#Rate for the random variable modeling the number of units in an order.
		self.demRate=10

		self.possibleDepots=[17728, 16130, 15427, 16650, 19372, 17582, 15824, 18014, 18463]#list(self.possDepots(N=[9]))
		n=len(self.possibleDepots)
		k=self.h

		self.posDisplays=self.calcPosDisp()
		
		if load:
			#Vertexes

			self.V=self.import_data('mopta2020_vertices.csv',names=['id','loc_name','long','lat']).set_index('id')
			
			#Edges
			self.E=self.import_data('mopta2020_edges.csv',h=0,names=['t','h','d'])

			#Daily demmand
			self.Demands=self.import_data('mopta2020_q2019.csv',h=None,names=['id']+list(range(1,366))).set_index('id')
			#self.Demands=pd.read_csv('Scenarios_robust_new.csv',header=0,index_col=0)
			#pd.read_csv('Scenarios.csv',header=1,sep=',',index_col=0)

			#Vertexes
			self.f=self.import_data('mopta2020_depots.csv',names=['id','cost']).set_index('id')#.to_dict()['cost']


			#Distance matrix
			self.dist_m=self.import_data('distance_matrix.csv',h=0,names=[0]+list(self.V.index)).set_index(0)


			#Graph
			self.G=nx.Graph()
			for i in self.V.iloc:
				self.G.add_node(int(i.name),loc_name=i[0],long=i[1],lat=i[2])
				
			for i in self.E.iloc:
				self.G.add_edge(int(i[0]),int(i[1]),d=i[2],c=0.7*i[2],t=(1/40)*i[2])
			
			#print(self.G.nodes)
			for i in list(self.G.nodes).copy():
				if i not in self.V.index:
					#print(i)
					self.G.remove_node(i)
			
			#Graph position
			self.pos={i:(self.G.nodes[i]['long'],self.G.nodes[i]['lat']) for i in self.G.nodes}

			
			#Subproblems
			self.SPS=[]
			for i in self.Demands:
				di=self.Demands[i][self.Demands[i]>0]
				#print(self.Demands[i])
				#di_mat=self.dist_m.loc[list(di.index),list(di.index)]
				self.SPS.append(sub_problem(id=len(self.SPS),H=[],D=di,dm=self.dist_m,pos=self.pos,master=self))
				#print(self.SPS[-1].id)
				#if len(self.SPS)>1:
			#		break		
			self.τ=4
			#Vecindarios de cada nodo
			self.Adj=(self.dist_m*self.t_mile).applymap(lambda x: 1 if x<=self.τ else 0)

			#Probabilitie of placing orders for eac node.
			self.prob=1/len(self.Demands.index)*self.Demands.applymap(lambda x: 1 if x>0 else 0).sum(axis=1)
			
			#Lower bound on the number of vehicles needed in each node
			#self.compMinVeh(α=0.4)

			self.minVeh=self.import_data('num_veh.csv',h=0,names=['id','minVeh']).set_index('id').applymap(lambda x: max(x,1))
			self.minVeh=self.minVeh['minVeh']
			#Cost of not covering a node
			self.ρ=self.prob*self.demRate*100

			#Maximum set of routes for the subproblems
			self.maxNumRoutes=200
			#Period for cleaning set of routes of the subproblems
			self.nItemptyRoutes=5
	
	def createLog(self,h):
		global log_text,log_path
		os.chdir('../Results')
		log_text=open(f'log{h}.txt','w')
		log_path=os.getcwd()+f'/log{h}.txt'		
		log_text.close()
		os.chdir('../Data')	

	def calcPosDisp(self):
		n=len(self.possibleDepots)
		k=self.h
		return math.factorial(n)/(math.factorial(k)*(math.factorial(n-k)))

	def import_data(self,file,names,h=None):
		'''
		Imports data
		'''
		return pd.read_csv(file,header=h,names=names)		

	def readModel(self,name='BMP_sinH.mps'):
		'''
		Reads the model from a .mts file and re-stores the variables in the dcitionaries used after m._x, m._y, m._η, m._w
		
		Input: None

		Output: Guroby model

		'''

		m=read(name)

		#m._x={i:m.addVar(vtype=GRB.BINARY,name='x_'+str(i)) for i in self.G.nodes}
		m._x={i:m.getVarByName(f'x_{i}') for i in self.possibleDepots}

		#m._y={i:m.addVar(vtype=GRB.INTEGER,name='y_'+str(i)) for i in self.G.nodes}
		m._y={i:m.getVarByName(f'y_{i}') for i in  self.possibleDepots}


		#m._w={i:m.addVar(vtype=GRB.BINARY,name='w_'+str(i)) for i in self.G.nodes}			
		m._w={i:m.getVarByName(f'w_{i}') for i in self.G.nodes}

		#m._η={s:m.addVar(vtype=GRB.CONTINUOUS,name='η_'+str(s)) for s,i in enumerate(self.SPS)}
		m._η={s:m.getVarByName(f'η_{s}') for s,i in enumerate(self.SPS)}

		m._first_stage=quicksum(self.f['cost'].loc[i]*m._x[i]+ self.c*m._y[i] for i in self.possibleDepots)
		n=len(self.SPS)
		
		m._E_second_stage=quicksum((1/n)*m._η[s] for s,i in enumerate(self.SPS))
		m._penal=quicksum(self.ρ[i]*(1-m._w[i]) for i in self.G.nodes)
		m.setObjective(m._first_stage+m._E_second_stage+m._penal)

		return m

	def Benders_master_p(self,epsilon,read=True,name='BMP_sinH.mps'):
		'''
		Creates the Benders master problem.
		
		Input:
			epsilon (floatf): tolerance for the optimaloty gap.
			read: Optinal parameter. If true, the model is charged from an .mps file with name 
			name: Name of the .mps file charged.
		Output: 
			m: Guroby model for the BMP (Benders master problem). Basic model, set of optimality and feasibility cuts emplty.
		
		'''	

		if read:
			m=self.readModel(name=name)
			m.setParam('OutputFlag', 0)
			m._epsilon=epsilon			
			m.update()

		else:
			m=Model('BMP')
			m.setParam('OutputFlag', 0)
			m._epsilon=epsilon

			#Variables:

			#x_i:{■(1&if a facility is located at node i∈N@0&o.t.w)┤
			#m._x={i:m.addVar(vtype=GRB.BINARY,name='x_'+str(i)) for i in self.G.nodes}						
			m._x={i:m.addVar(vtype=GRB.BINARY,name='x_'+str(i)) for i in self.possibleDepots}

			#y_i:Number of trucks located at node i∈N
			#m._y={i:m.addVar(vtype=GRB.INTEGER,name='y_'+str(i)) for i in self.G.nodes}
			m._y={i:m.addVar(vtype=GRB.INTEGER,name='y_'+str(i)) for i in self.possibleDepots}

			#w_i:1 if node i\in N is reachable
			m._w={i:m.addVar(vtype=GRB.BINARY,name='w_'+str(i)) for i in self.G.nodes}

			#η_s:Cost of satisfying demand on scenario s∈S
			m._η={s:m.addVar(vtype=GRB.CONTINUOUS,name='η_'+str(s)) for s,i in enumerate(self.SPS)}

			#Restricciones			

			m._N_depots=m.addConstr(quicksum(m._x[i] for i in self.possibleDepots)==self.h)
			

			#1.)If the location i ∈ N has no facility, the number of vehicles in that node has to be zero.

			m._N_vehicles={i:m.addConstr(m._y[i]<=self.M*m._x[i]) for i in self.possibleDepots}
			
			#2.)Proper activation of wi.
			
			m._Reachable1={i:m.addConstr(m._x[i]<=sum(self.Adj[i])*quicksum(m._w[j] for j in self.G.nodes if self.Adj[i][j]==1)) for i in  self.possibleDepots}

			m._Reachable={i:m.addConstr(m._w[i]<=quicksum(m._x[j] for j in self.possibleDepots if self.Adj[i][j]==1)) for i in self.G.nodes}
			
			m._Reachable=m.addConstr(quicksum(m._w[i] for i in self.G.nodes)==len(self.G.nodes)) 


			#3.) 
			m._N_vehiclesMin={i:m.addConstr(self.minVeh[i]*m._x[i]<=m._y[i])for i in  self.possibleDepots}
			#m._N_vehiclesMin={i:m.addConstr(2*m._x[i]<=m._y[i]) for i in self.G.nodes}
			#4. La solucion no debe ser muy distinta a la del warm_start.

			'''
			y_hat, x_hat, η_hat, Cost=self.warm_start()
			##TODO calcular Fo del warmstart y add to LB (funcion aparte) que le entre y,x return fo		
			for i in y_hat.keys():
				m._y[i].start=y_hat[i]
				m._x[i].start=x_hat[i]
			
			ON= quicksum((1-m._x[i]) for i in self.G.nodes if x_hat[i]>0.5)
			OFF = quicksum(m._x[i] for i in self.G.nodes if x_hat[i]>0.5)
			m.addConstr(ON<=5,name='LS')
			'''

			#5. en cada cluster debe haber al menos un depot...	
			#m._clustersC={i:m.addConstr(quicksum(m._x[j] for j in c)>=1,name=f'Cluster_{i}') for i,c in enumerate(self.Clusters)}


			#Objective function
			
			

			m._first_stage=quicksum(self.f['cost'].loc[i]*m._x[i]+ self.c*m._y[i] for i in  self.possibleDepots)
			n=len(self.SPS)
			m._E_second_stage=quicksum((1/n)*m._η[s] for s,i in enumerate(self.SPS))
			m._penal=quicksum(self.ρ[i]*(1-m._w[i]) for i in self.G.nodes)
			m.setObjective(m._first_stage+m._E_second_stage+m._penal)

			m.update()
			m.write('BMP_sinH.mps')
		return m

	def Benders_algo(self,time_limit=None,read=True):
		'''
		Benders algorithm is implemented.		
		
		Input:
			None
				
		Output: 
			Solution for teh problem: Set of depots (H) and the number of vehicles assigned to each vehicles.

		
		'''
		if time_limit!=None:
			self.time_limit=time_limit

		#print('llegue')

		print_log('############################################################################################################################')
		tCBMP=time.time()
		m=self.Benders_master_p(epsilon=0.1,read=read)
		m.update()
		

		print_log('############################################################################################################################')
		print_log(f'Me demoro construyendo el BMP{time.time()-tCBMP}')
		#print(m.getConstrs())
		#print('termine')
		it=0
		cuts = 0
		LB, UB, CUT, SOL= [], [float("inf")], [],[]
		ETAS,SStage=[],[]
		
		Y_HAT=[]

		Start_time=time.time()
		while True:
			it+=1
			
			#Solve the Master problem callback epsilon_optimal:
			print_log('############################################################')
			print_log(f'Iteracion {it}')
			m.optimize(epsilon_optimal)
			print_log(f'Termine benders master de la it {it}')
			print_log('############################################################')
			#Update the Lower Bound
			LB.append(m.objVal)

			'''			if it==1:
				y_hat, x_hat, η_hat, Cost=self.warm_start()
				##TODO calcular Fo del warmstart y add to LB (funcion aparte) que le entre y,x return fo					
				LB.append(Cost)
				for i in y_hat.keys():
					m._y[i].start=y_hat[i]
					m._x[i].start=x_hat[i]
				ON= quicksum((1-m._x[i]) for i in self.G.nodes if x_hat[i]>0.5)
				OFF = quicksum(m._x[i] for i in self.G.nodes if x_hat[i]>0.5)
				m.addConstr(ON<=5,name='LS')
				m.update()
			else:'''
			

			#Recovers solution
			y_hat={i:m._y[i].x for i in self.possibleDepots}
			x_hat={i:m._x[i].x for i in self.possibleDepots}
			η_hat={s:m._η[s].x for s,i in enumerate(self.SPS)}

			ETAS.append(sum(η_hat.values()))

			
			ON= quicksum((1-m._x[i]) for i in self.possibleDepots if x_hat[i]>0.5)
			OFF = quicksum(m._x[i] for i in self.possibleDepots if x_hat[i]<0.5)

			#Local search constraints... remove (old) and add (new).
			#m.remove(m.getConstrByName('LS'))
			#m.addConstr(ON<=5,name='LS')
			m.update()

			#Solve the subproblems
			OF=[]
			for ss,s in enumerate(self.SPS):
				print(f'\tEscenario {ss}')
				
				tspi=time.time()
				s.H=[i for i in x_hat.keys() if x_hat[i]>.1]
				s.y_hat=y_hat								
				#Solve sp s
				print_log('\t############################################################')
				print_log('\t',f'Subproblema {ss}')
				#FOi,λ,π=self.solve_sp(s)				
				FOi,λ,π=self.solveSpJava(s)
				print_log('\t',f'Subproblema {ss} me demoro {time.time()-tspi}')
				print_log('\t############################################################')
				OF.append(FOi)				
				#Checks for optimality cuts
				if η_hat[ss]<FOi:
					#Combinatorial Benders Cuts
					print_log('######################################################################')
					print_log(f'La FOi da {FOi}')
					print_log(f'El corte me dice {sum(π[i] for i in π.keys()) + sum(λ[i]*y_hat[i] for i in self.possibleDepots)}')
					print_log(f'π dan Esto da {sum(π[i] for i in π.keys())}')
					γ=FOi-(sum(π[i] for i in π.keys()) + sum(λ[i]*y_hat[i] for i in self.possibleDepots))
					print_log(f'Las λ me dan\n {λ.values()}')
					print_log('###############################################®#######################')

					m.addConstr(m._η[ss]>=γ+quicksum(π[i] for i in π.keys()) + quicksum(λ[i]*m._y[i] for i in self.possibleDepots) - FOi*(ON+OFF),name=f'CombOC_{it},{ss}')										

					#Optimality Continious Benders Cuts
					#m.addConstr(m._η[ss]>=γ+quicksum(π[i] for i in π.keys()) + quicksum(λ[i]*m._y[i] for i in self.G.nodes) )
					m.update()
					cuts+=1				
				s.save(f'sp{ss}')
				
				

				#TODO: Feasibility cuts?


			CUT.append(cuts)
			UB.append(min(UB[-1],LB[-1]-(1/len(self.SPS))*(sum(η_hat.values())-sum(OF))))
			SOL.append(LB[-1]-sum(η_hat.values())+sum(OF))

			SStage.append(sum(OF))
						
			num_veh={h:y_hat[h] for h in y_hat.keys() if y_hat[h]>0.5}			
			self.AnimNVeh.append(num_veh)
			
			if it%1==0:

				print_log('Hice esto...')
				print_log(UB)
				print_log(LB)

				os.chdir('../Plots')

				plt.figure(num=None, figsize=(16*scal, 10*scal), dpi=80, facecolor='w', edgecolor='k')
				iii=max(len(LB)-10,0)
				os.chdir('Bounds')
				if it==1: delete_all_files(path=os.getcwd())

				#fig, ax1 = plt.subplots()
				plt.plot(range(len(UB[1:])),UB[1:],'go-')
				plt.plot(range(len(LB[:])),LB[:],'ro-')				
				plt.legend(['Upper bound','Lower bound'])			
				plt.savefig(f'bounds_{it}.png')
				plt.clf()
				
				plt.plot(range(len(UB[iii+1:])),UB[iii+1:],'go-')
				plt.plot(range(len(LB[iii:])),LB[iii:],'ro-')				
				plt.legend(['Upper bound','Lower bound'])			
				plt.savefig(f'boundsZoom_{it}.png')
				plt.clf()

				os.chdir('../Map')
				if it==1: delete_all_files(path=os.getcwd())
				numVeh={i:y_hat[i] for i in y_hat.keys() if y_hat[i]>0.5}
				Y_HAT.append(numVeh)
				self.plotDepots(numVeh)
				plt.savefig(f'BenMap_{it}.png')
				plt.clf()
				os.chdir('../SSCost')
				if it==1: delete_all_files(path=os.getcwd())
				plt.plot(range(len(ETAS)),ETAS,'ro-')
				plt.plot(range(len(SStage)),SStage,'go-')
				#plt.ylim(min(ETAS),min(SStage)*1.3)
				plt.legend(['Estimation of SS cost','SS cost'])
				plt.savefig(f'ssCosts_{it}.png')
				plt.clf()

				plt.plot(range(len(ETAS[iii:])),ETAS[iii:],'ro-')
				plt.plot(range(len(SStage[iii:])),SStage[iii:],'go-')
				#plt.ylim(min(ETAS),min(SStage)*1.3)
				plt.legend(['Estimation of SS cost','SS cost'])
				plt.savefig(f'ssCostsZoom_{it}.png')
				plt.clf()
				os.chdir('..')
				os.chdir('../Data')


			#Check optimality gap
			if ((UB[-1]-LB[-1])<self.epsilon*UB[-1]):
				
				break
			elif (time.time()-Start_time>=self.time_limit):
				print_log('############################################################################################################')
				print_log(f'Time limit exceeded ({self.time_limit} seconds)')
				print_log('############################################################################################################')
				break

		self.export_results(depots=x_hat,veh_depot=y_hat)
		

		print_log('############################################################################################################')
		print_log(f'For animation...')
		print_log(Y_HAT)
		print_log('############################################################################################################')

		return UB,LB,x_hat,y_hat

	def Benders_algoNoCG(self,time_limit=None,read=True):
		'''
		Benders algorithm is implemented.		
		
		Input:
			None
				
		Output: 
			Solution for teh problem: Set of depots (H) and the number of vehicles assigned to each vehicles.

		
		'''
		if time_limit!=None:
			self.time_limit=time_limit

		#print('llegue')

		print_log('############################################################################################################################')
		tCBMP=time.time()
		m=self.Benders_master_p(epsilon=0.1,read=read)
		m.update()
		

		print_log('############################################################################################################################')
		print_log(f'Me demoro construyendo el BMP{time.time()-tCBMP}')
		#print(m.getConstrs())
		#print('termine')
		it=0
		cuts = 0
		LB, UB, CUT, SOL= [], [float("inf")], [],[]
		ETAS,SStage=[],[]
		
		Y_HAT=[]

		Start_time=time.time()
		while True:
			it+=1
			
			#Solve the Master problem callback epsilon_optimal:
			print_log('############################################################')
			print_log(f'Iteracion {it}')
			m.optimize(epsilon_optimal)
			print_log(f'Termine benders master de la it {it}')
			print_log('############################################################')
			#Update the Lower Bound
			LB.append(m.objVal)

			'''			if it==1:
				y_hat, x_hat, η_hat, Cost=self.warm_start()
				##TODO calcular Fo del warmstart y add to LB (funcion aparte) que le entre y,x return fo					
				LB.append(Cost)
				for i in y_hat.keys():
					m._y[i].start=y_hat[i]
					m._x[i].start=x_hat[i]
				ON= quicksum((1-m._x[i]) for i in self.G.nodes if x_hat[i]>0.5)
				OFF = quicksum(m._x[i] for i in self.G.nodes if x_hat[i]>0.5)
				m.addConstr(ON<=5,name='LS')
				m.update()
			else:'''
			

			#Recovers solution
			y_hat={i:m._y[i].x for i in self.possibleDepots}
			x_hat={i:m._x[i].x for i in self.possibleDepots}
			η_hat={s:m._η[s].x for s,i in enumerate(self.SPS)}

			ETAS.append(sum(η_hat.values()))

			
			ON= quicksum((1-m._x[i]) for i in self.possibleDepots if x_hat[i]>0.5)
			OFF = quicksum(m._x[i] for i in self.possibleDepots if x_hat[i]<0.5)

			#Local search constraints... remove (old) and add (new).
			#m.remove(m.getConstrByName('LS'))
			#m.addConstr(ON<=5,name='LS')
			m.update()

			#Solve the subproblems
			OF=[]
			for ss,s in enumerate(self.SPS):
				print(f'\tEscenario {ss}')
				
				tspi=time.time()
				s.H=[i for i in x_hat.keys() if x_hat[i]>.1]
				s.y_hat=y_hat								
				#Solve sp s
				print_log('\t############################################################')
				print_log('\t',f'Subproblema {ss}')
				#FOi,λ,π=self.solve_sp(s)				
				FOi,λ,π=self.solveSpNoCG(s)
				print_log('\t',f'Subproblema {ss} me demoro {time.time()-tspi}')
				print_log('\t############################################################')
				OF.append(FOi)				
				#Checks for optimality cuts
				if η_hat[ss]<FOi:
					#Combinatorial Benders Cuts
					print_log('######################################################################')
					print_log(f'La FOi da {FOi}')
					print_log(f'El corte me dice {sum(π[i] for i in π.keys()) + sum(λ[i]*y_hat[i] for i in self.possibleDepots)}')
					print_log(f'π dan Esto da {sum(π[i] for i in π.keys())}')
					γ=FOi-(sum(π[i] for i in π.keys()) + sum(λ[i]*y_hat[i] for i in self.possibleDepots))
					print_log(f'Las λ me dan\n {λ.values()}')
					print_log('###############################################®#######################')

					m.addConstr(m._η[ss]>=γ+quicksum(π[i] for i in π.keys()) + quicksum(λ[i]*m._y[i] for i in self.possibleDepots) - FOi*(ON+OFF),name=f'CombOC_{it},{ss}')										

					#Optimality Continious Benders Cuts
					#m.addConstr(m._η[ss]>=γ+quicksum(π[i] for i in π.keys()) + quicksum(λ[i]*m._y[i] for i in self.G.nodes) )
					m.update()
					cuts+=1	
				s.save(f'sp{ss}')
				
				

				#TODO: Feasibility cuts?


			CUT.append(cuts)
			UB.append(min(UB[-1],LB[-1]-(1/len(self.SPS))*(sum(η_hat.values())-sum(OF))))
			SOL.append(LB[-1]-sum(η_hat.values())+sum(OF))

			SStage.append(sum(OF))
						
			num_veh={h:y_hat[h] for h in y_hat.keys() if y_hat[h]>0.5}			
			self.AnimNVeh.append(num_veh)
			
			if it%1==0:

				print_log('Hice esto...')
				print_log(UB)
				print_log(LB)

				os.chdir('../Plots')

				plt.figure(num=None, figsize=(16*scal, 10*scal), dpi=80, facecolor='w', edgecolor='k')
				iii=max(len(LB)-10,0)
				os.chdir('Bounds')
				if it==1: delete_all_files(path=os.getcwd())

				#fig, ax1 = plt.subplots()
				plt.plot(range(len(UB[1:])),UB[1:],'go-')
				plt.plot(range(len(LB[:])),LB[:],'ro-')				
				plt.legend(['Upper bound','Lower bound'])			
				plt.savefig(f'boundsNoCG_{it}.png')
				plt.clf()
				
				plt.plot(range(len(UB[iii+1:])),UB[iii+1:],'go-')
				plt.plot(range(len(LB[iii:])),LB[iii:],'ro-')				
				plt.legend(['Upper bound','Lower bound'])			
				plt.savefig(f'boundsZoomNoCG_{it}.png')
				plt.clf()

				os.chdir('../Map')
				if it==1: delete_all_files(path=os.getcwd())
				numVeh={i:y_hat[i] for i in y_hat.keys() if y_hat[i]>0.5}
				Y_HAT.append(numVeh)
				self.plotDepots(numVeh)
				plt.savefig(f'BenMapNoCG__{it}.png')
				plt.clf()
				os.chdir('../SSCost')
				if it==1: delete_all_files(path=os.getcwd())
				plt.plot(range(len(ETAS)),ETAS,'ro-')
				plt.plot(range(len(SStage)),SStage,'go-')
				#plt.ylim(min(ETAS),min(SStage)*1.3)
				plt.legend(['Estimation of SS cost','SS cost'])
				plt.savefig(f'ssCostsNoCG__{it}.png')
				plt.clf()

				plt.plot(range(len(ETAS[iii:])),ETAS[iii:],'ro-')
				plt.plot(range(len(SStage[iii:])),SStage[iii:],'go-')
				#plt.ylim(min(ETAS),min(SStage)*1.3)
				plt.legend(['Estimation of SS cost','SS cost'])
				plt.savefig(f'ssCostsZoomNoCG__{it}.png')
				plt.clf()
				os.chdir('..')
				os.chdir('../Data')


			#Check optimality gap
			if ((UB[-1]-LB[-1])<self.epsilon*UB[-1]):
				
				break
			elif (time.time()-Start_time>=self.time_limit):
				print_log('############################################################################################################')
				print_log(f'Time limit exceeded ({self.time_limit} seconds)')
				print_log('############################################################################################################')
				break

		self.export_results(depots=x_hat,veh_depot=y_hat)
		

		print_log('############################################################################################################')
		print_log(f'For animation...')
		print_log(Y_HAT)
		print_log('############################################################################################################')

		return UB,LB,x_hat,y_hat

	def BendersAlgoMix(self,epsilon=0.1,read=True):
		'''
		Benders algorithm is implemented.		
		
		Input:
			None
				
		Output: 
			Solution for teh problem: Set of depots (H) and the number of vehicles assigned to each vehicles.		
		'''

		#Creates empty (no cuts) master problem
		m=self.Benders_master_p(epsilon=epsilon,read=read)
		m.update()

		#Initializes important variables:
		it=0
		cuts = 0
		LB, UB, CUT= [], [float("inf")], []
		ColGenCalls=0
		Start_time=time.time()
		ColGen=False		
		while True:			
			it+=1
			#If ColGen then it is that the previous solution is going to be improved, so no new solution needs to be obtained.
			if not ColGen:
				#Solve master problem and Update the Lower Bound
				m.update()
				m.optimize()
				LB.append(m.objVal)

				#Recovers solution
				y_hat={i:m._y[i].x for i in self.possibleDepots}
				x_hat={i:m._x[i].x for i in self.possibleDepots}
				η_hat={s:m._η[s].x for s,i in enumerate(self.SPS)}

				#On and Off variables
				ON= quicksum((1-m._x[i]) for i in self.possibleDepots if x_hat[i]>0.5)
				OFF = quicksum(m._x[i] for i in self.possibleDepots if x_hat[i]<0.5)
			
			#Solve the subproblems
			OF=[]	#Objective function
			for ss,s in enumerate(self.SPS):
				s.H=[i for i in x_hat.keys() if x_hat[i]>.1]
				s.y_hat=y_hat

				#Strategy for solvin SPs searching for more routes or not
				if ColGen:
					FOi,λ,π=self.solveSpJava(s,nRoutes=10,tw=True)
					#Save sp with new generated routes:
					s.save(f'sp{s.id}')
					ColGenCalls+=1
				else:
					FOi,λ,π=self.solveSpNoCG(s)
				#Save FO
				OF.append(FOi)
				#Checks for optimality cuts
				if η_hat[ss]<FOi:
					m.addConstr(m._η[ss]>=quicksum(π[i] for i in π.keys()) + quicksum(λ[i]*m._y[i] for i in self.possibleDepots) - FOi*(ON+OFF),name=f'CombOC_{it},{ss}')					
					cuts+=1		
			
			#Update number of cuts and Upper bound
			CUT.append(cuts)
			UB.append(min(UB[-1],LB[-1]-(1/len(self.SPS))*(sum(η_hat.values())-sum(OF))))
			
			#Decide weather to gen routes or not
			if not ColGen and it>2:				
				print(f"Improvements {UB[-1]-UB[-2]}")
				if UB[-1]-UB[-2]<0:
					ColGen=True
					for ss,s in enumerate(self.SPS):
						try:
							m.remove(m.getConstrByName(f'CombOC_{it},{ss}'))
						except:
							pass
			else:
				ColGen=False
			
			#Check optimality gap
			try:
				print(f'El gap es de: {(UB[-1]-LB[-1])/UB[-1]}\nCon pb de {UB[-1]}')
			except:
				pass
			if ((UB[-1]-LB[-1])<self.epsilon*UB[-1]):								
				break
			elif (time.time()-Start_time>=self.time_limit):				
				break
			#Export results
			self.export_results(depots=x_hat,veh_depot=y_hat)
			
		return UB,LB,x_hat,y_hat,ColGenCalls

	def Benders_algoMix(self,epsilon=0.1,time_limit=None,read=True):
		'''
		Benders algorithm is implemented.		
		
		Input:
			None
				
		Output: 
			Solution for teh problem: Set of depots (H) and the number of vehicles assigned to each vehicles.

		
		'''
		if time_limit!=None:
			self.time_limit=time_limit

		
		print_log('############################################################################################################################')
		tCBMP=time.time()
		m=self.Benders_master_p(epsilon=epsilon,read=read)
		m.update()
		
		print_log('############################################################################################################################')
		print_log(f'Me demoro construyendo el BMP{time.time()-tCBMP}')
		#print(m.getConstrs())
		#print('termine')
		it=0
		cuts = 0
		LB, UB, CUT, SOL= [], [float("inf")], [],[]
		ETAS,SStage=[],[]
		
		Y_HAT=[]

		Start_time=time.time()
		ColGen=False
		while True:
			it+=1
			if not ColGen:
				#Solve the Master problem callback epsilon_optimal:
				print_log('############################################################')
				print_log(f'Iteracion {it}')
				m.optimize()
				print_log(f'Termine benders master de la it {it}')
				print_log('############################################################')
				#Update the Lower Bound
				LB.append(m.objVal)
				

				'''			if it==1:
					y_hat, x_hat, η_hat, Cost=self.warm_start()
					##TODO calcular Fo del warmstart y add to LB (funcion aparte) que le entre y,x return fo					
					LB.append(Cost)
					for i in y_hat.keys():
						m._y[i].start=y_hat[i]
						m._x[i].start=x_hat[i]
					ON= quicksum((1-m._x[i]) for i in self.G.nodes if x_hat[i]>0.5)
					OFF = quicksum(m._x[i] for i in self.G.nodes if x_hat[i]>0.5)
					m.addConstr(ON<=5,name='LS')
					m.update()
				else:'''
				
				#Recovers solution
				y_hat={i:m._y[i].x for i in self.possibleDepots}
				x_hat={i:m._x[i].x for i in self.possibleDepots}
				η_hat={s:m._η[s].x for s,i in enumerate(self.SPS)}

				ETAS.append(sum(η_hat.values()))


				
				ON= quicksum((1-m._x[i]) for i in self.possibleDepots if x_hat[i]>0.5)
				OFF = quicksum(m._x[i] for i in self.possibleDepots if x_hat[i]<0.5)

				#Local search constraints... remove (old) and add (new).
				#m.remove(m.getConstrByName('LS'))
				#m.addConstr(ON<=5,name='LS')
				m.update()

			#Solve the subproblems
			OF=[]
			for ss,s in enumerate(self.SPS):
				#print(f'\tEscenario {ss}')
				
				tspi=time.time()
				s.H=[i for i in x_hat.keys() if x_hat[i]>.1]
				s.y_hat=y_hat								
				#Solve sp s
				print_log('\t############################################################')
				print_log('\t',f'Subproblema {ss}')
				#print(f'Colgen antes de correr me dice {ColGen}')
				#if it<self.posDisplays:
					#ColGen=False	

				#if random.random()<1:
				#	ColGen=False
				Colgen=True
				if ColGen:
					FOi,λ,π=self.solveSpJava(s,nRoutes=10)
				else:
					FOi,λ,π=self.solveSpNoCG(s)
				print_log('\t',f'Subproblema {ss} me demoro {time.time()-tspi}')
				print_log('\t############################################################')
				OF.append(FOi)				
				#Checks for optimality cuts
				if η_hat[ss]<FOi:
					#Combinatorial Benders Cuts
					print_log('######################################################################')
					print_log(f'La FOi da {FOi}')
					print_log(f'El corte me dice {sum(π[i] for i in π.keys()) + sum(λ[i]*y_hat[i] for i in self.possibleDepots)}')
					print_log(f'π dan Esto da {sum(π[i] for i in π.keys())}')
					#γ=FOi-(sum(π[i] for i in π.keys()) + sum(λ[i]*y_hat[i] for i in self.possibleDepots))
					print_log(f'Las λ me dan\n {λ.values()}')
					print_log('###############################################®#######################')

					m.addConstr(m._η[ss]>=quicksum(π[i] for i in π.keys()) + quicksum(λ[i]*m._y[i] for i in self.possibleDepots) - FOi*(ON+OFF),name=f'CombOC_{it},{ss}')										

					#Optimality Continious Benders Cuts
					#m.addConstr(m._η[ss]>=γ+quicksum(π[i] for i in π.keys()) + quicksum(λ[i]*m._y[i] for i in self.G.nodes) )
					m.update()
					cuts+=1				
				s.save(f'sp{s.id}')				
				

				#TODO: Feasibility cuts?


			CUT.append(cuts)
			UB.append(min(UB[-1],LB[-1]-(1/len(self.SPS))*(sum(η_hat.values())-sum(OF))))
			SOL.append(LB[-1]-sum(η_hat.values())+sum(OF))			
			SStage.append(sum(OF))
						
			num_veh={h:y_hat[h] for h in y_hat.keys() if y_hat[h]>0.5}			
			self.AnimNVeh.append(num_veh)

			if not ColGen:
				if it>2:
					print(f"Improvements {UB[-1]-UB[-2]}")
					if UB[-1]-UB[-2]<0:
						ColGen=True
						for ss,s in enumerate(self.SPS):
							try:
								m.remove(m.getConstrByName(f'CombOC_{it},{ss}'))
							except:
								pass
					#print('Colge =true')
			else:
				ColGen=False

			print('estoy imprimiendo aca...')
			print_log('Hice esto...')
			print_log(UB)
			print_log(LB)
			if it%10==0:
				os.chdir('../Plots')

				plt.figure(num=None, figsize=(16*scal, 10*scal), dpi=80, facecolor='w', edgecolor='k')
				iii=max(len(LB)-10,0)
				os.chdir('Bounds')
				if it==1: delete_all_files(path=os.getcwd())

				#fig, ax1 = plt.subplots()
				plt.plot(range(len(UB[1:])),UB[1:],'go-')
				plt.plot(range(len(LB[:])),LB[:],'ro-')				
				plt.legend(['Upper bound','Lower bound'])			
				plt.savefig(f'boundsNoCG_{it}.png')
				plt.clf()
				
				plt.plot(range(len(UB[iii+1:])),UB[iii+1:],'go-')
				plt.plot(range(len(LB[iii:])),LB[iii:],'ro-')				
				plt.legend(['Upper bound','Lower bound'])			
				plt.savefig(f'boundsZoomNoCG_{it}.png')
				plt.clf()

				os.chdir('../Map')
				if it==1: delete_all_files(path=os.getcwd())
				numVeh={i:y_hat[i] for i in y_hat.keys() if y_hat[i]>0.5}
				Y_HAT.append(numVeh)
				self.plotDepots(numVeh)
				plt.savefig(f'BenMapNoCG__{it}.png')
				plt.clf()
				os.chdir('../SSCost')
				if it==1: delete_all_files(path=os.getcwd())
				plt.plot(range(len(ETAS)),ETAS,'ro-')
				plt.plot(range(len(SStage)),SStage,'go-')
				#plt.ylim(min(ETAS),min(SStage)*1.3)
				plt.legend(['Estimation of SS cost','SS cost'])
				plt.savefig(f'ssCostsNoCG__{it}.png')
				plt.clf()

				plt.plot(range(len(ETAS[iii:])),ETAS[iii:],'ro-')
				plt.plot(range(len(SStage[iii:])),SStage[iii:],'go-')
				#plt.ylim(min(ETAS),min(SStage)*1.3)
				plt.legend(['Estimation of SS cost','SS cost'])
				plt.savefig(f'ssCostsZoomNoCG__{it}.png')
				plt.clf()
				os.chdir('..')
				os.chdir('../Data')


			#Check optimality gap
			print(f'Voy en la it {it} con gap de {(UB[-1]-LB[-1])/UB[-1]}')
			if ((UB[-1]-LB[-1])<self.epsilon*UB[-1]):				
				break
			elif (time.time()-Start_time>=self.time_limit):
				print_log('############################################################################################################')
				print_log(f'Time limit exceeded ({self.time_limit} seconds)')
				print_log('############################################################################################################')
				break

		self.export_results(depots=x_hat,veh_depot=y_hat)
		

		print_log('############################################################################################################')
		print_log(f'For animation...')
		print_log(Y_HAT)
		print_log('############################################################################################################')

		return UB,LB,x_hat,y_hat

	def Benders_algoMod(self,time_limit=None,read=True):
		'''
		Benders algorithm is implemented.		
		
		Input:
			None
				
		Output: 
			Solution for teh problem: Set of depots (H) and the number of vehicles assigned to each vehicles.

		
		'''
		if time_limit!=None:
			self.time_limit=time_limit

		#print('llegue')

		print_log('############################################################################################################################')
		tCBMP=time.time()
		m=self.Benders_master_p(epsilon=0.05,read=read)
		m.update()
		

		print_log('############################################################################################################################')
		print_log(f'Me demoro construyendo el BMP{time.time()-tCBMP}')
		#print(m.getConstrs())
		#print('termine')
		it=0
		cuts = 0
		LB, UB, CUT, SOL= [], [float("inf")], [],[]
		ETAS,SStage=[],[]
		
		Y_HAT=[]

		Start_time=time.time()
		while True:
			it+=1
			
			#Solve the Master problem callback epsilon_optimal:
			print_log('############################################################')
			print_log(f'Iteracion {it}')
			m.optimize()
			print_log(f'Termine benders master de la it {it}')
			print_log('############################################################')
			#Update the Lower Bound
			LB.append(m.objVal)

			'''			if it==1:
				y_hat, x_hat, η_hat, Cost=self.warm_start()
				##TODO calcular Fo del warmstart y add to LB (funcion aparte) que le entre y,x return fo					
				LB.append(Cost)
				for i in y_hat.keys():
					m._y[i].start=y_hat[i]
					m._x[i].start=x_hat[i]
				ON= quicksum((1-m._x[i]) for i in self.G.nodes if x_hat[i]>0.5)
				OFF = quicksum(m._x[i] for i in self.G.nodes if x_hat[i]>0.5)
				m.addConstr(ON<=5,name='LS')
				m.update()
			else:'''
			

			#Recovers solution
			y_hat={i:m._y[i].x for i in self.G.nodes}
			x_hat={i:m._x[i].x for i in self.G.nodes}
			η_hat={s:m._η[s].x for s,i in enumerate(self.SPS)}

			ETAS.append(sum(η_hat.values()))

			
			ON= quicksum((1-m._x[i]) for i in self.G.nodes if x_hat[i]>0.5)
			OFF = quicksum(m._x[i] for i in self.G.nodes if x_hat[i]<0.5)
			
			try:
				for i, const in closeCstr.items():
					m.remove(Constr)
			except:
				pass
			closeCstr={i:m.addConstr(quicksum(m._x[j] for j in self.G.nodes if (self.dist_m[i][j]*self.t_mile)<0.5)>=1,name=f'close_{i},{it}')  for i in self.G.nodes if x_hat[i]>0.5}
			#Local search constraints... remove (old) and add (new).
			#m.remove(m.getConstrByName('LS'))
			#m.addConstr(ON<=5,name='LS')
			m.update()

			#Solve the subproblems
			OF=[]
			extra_tot=0
			for ss,s in enumerate(self.SPS):
				
				tspi=time.time()
				s.H=[i for i in x_hat.keys() if x_hat[i]>.1]
				s.y_hat=y_hat								
				#Solve sp s
				print_log('\t############################################################')
				print_log('\t',f'Subproblema {ss}')
				
				msp,FOi,λ,π=self.solve_sp(s,Alternative=True)
				
				print_log('\t',f'Subproblema {ss} me demoro {time.time()-tspi}')
				print_log('\t############################################################')

				OF.append(FOi)
				extra_sp=0
				#Checks for optimality cuts
				if η_hat[ss]<FOi:
					m.addConstr(m._η[ss]>=FOi - FOi*(ON+OFF),name=f'CombOC_{it},{ss}')
					for i in s.H:
						extra=msp._v[i].x						
						extra_sp+=extra
						if extra>0:
							print_log('\t',f'metí corte para y con v {extra}')
							m.addConstr(m._y[i]>=y_hat[i]+extra - (y_hat[i]+extra)*(ON+OFF),name=f'yhat_{it},{ss},{i}')

					#Optimality Continious Benders Cuts
					#print_log(f'\tLas duales son \n\t{λ.values()}\n\t{[λ[d] for d in s.H]}')
					#print_log(f'\tLas duales son \n\t{π.values()}')
					#m.addConstr(m._η[ss]>=quicksum(π[i] for i in π.keys()) + quicksum(λ[i]*m._y[i] for i in self.G.nodes) )
					
					m.update()
					cuts+=1				
				if extra_sp>extra_tot: extra_tot=extra_sp


				#TODO: Feasibility cuts?

			print_log(f'\tFSCosts:{LB[-1]}, extra vehicles: {extra_tot}')
			CUT.append(cuts)
			UB.append(min(UB[-1],LB[-1]-sum(η_hat.values())+sum(OF))+self.c*extra_tot)
			SOL.append(LB[-1]-sum(η_hat.values())+sum(OF))

			SStage.append(sum(OF))
						
			num_veh={h:y_hat[h] for h in y_hat.keys() if y_hat[h]>0.5}			
			self.AnimNVeh.append(num_veh)
			
			if it%1==0:

				print_log('Hice esto...')
				print_log(UB)
				print_log(LB)
				m.write(f'BMP_sinH_{it}.mps')
				os.chdir('../Plots')

				plt.figure(num=None, figsize=(16*scal, 10*scal), dpi=80, facecolor='w', edgecolor='k')
				iii=max(len(LB)-10,0)
				os.chdir('Bounds')
				if it==1: delete_all_files(path=os.getcwd())

				#fig, ax1 = plt.subplots()
				plt.plot(range(len(UB[1:])),UB[1:],'go-')
				plt.plot(range(len(LB[:])),LB[:],'ro-')				
				plt.legend(['Upper bound','Lower bound'])			
				plt.savefig(f'bounds_{it}.png')
				plt.clf()
				
				plt.plot(range(len(UB[iii+1:])),UB[iii+1:],'go-')
				plt.plot(range(len(LB[iii:])),LB[iii:],'ro-')				
				plt.legend(['Upper bound','Lower bound'])			
				plt.savefig(f'boundsZoom_{it}.png')
				plt.clf()

				os.chdir('../Map')
				if it==1: delete_all_files(path=os.getcwd())
				numVeh={i:y_hat[i] for i in y_hat.keys() if y_hat[i]>0.5}
				Y_HAT.append(numVeh)
				self.plotDepots(numVeh)
				plt.savefig(f'BenMap_{it}.png')
				plt.clf()

				os.chdir('../SSCost')
				if it==1: delete_all_files(path=os.getcwd())
				plt.plot(range(len(ETAS)),ETAS,'ro-')
				plt.plot(range(len(SStage)),SStage,'go-')
				#plt.ylim(min(ETAS),min(SStage)*1.3)
				plt.legend(['Estimation of SS cost','SS cost'])
				plt.savefig(f'ssCosts_{it}.png')
				plt.clf()

				plt.plot(range(len(ETAS[iii:])),ETAS[iii:],'ro-')
				plt.plot(range(len(SStage[iii:])),SStage[iii:],'go-')
				#plt.ylim(min(ETAS),min(SStage)*1.3)
				plt.legend(['Estimation of SS cost','SS cost'])
				plt.savefig(f'ssCostsZoom_{it}.png')
				plt.clf()
				os.chdir('..')
				os.chdir('../Data')


			#Check optimality gap
			if ((UB[-1]-LB[-1])<self.epsilon*UB[-1]):
				
				break
			elif (time.time()-Start_time>=self.time_limit):
				print_log('############################################################################################################')
				print_log(f'Time limit exceeded ({self.time_limit} seconds)')
				print_log('############################################################################################################')
				break

		self.export_results(depots=x_hat,veh_depot=y_hat)
		

		print_log('############################################################################################################')
		print_log(f'For animation...')
		print_log(Y_HAT)
		print_log('############################################################################################################')

		return UB,LB,x_hat,y_hat

	def solveSpJava(self,sp,nRoutes=10,gap=0.001,tw=False):
		FO=[]
		clients=sp.D.iloc[:]
		π={i:0 for i in clients.index}

		n_it=0
		HH=[i for i in sp.H if sp.y_hat[i]>0]
		while True:
			n_it+=1			

			term=True			
			for d in HH:
				m=sp.master_problem(relaxed=True)
				time_genPet=time.time()
				m.optimize()
				sp.z_hat={k:kk.x for k,kk in m._z.items()}

				π={i:m._set_covering[i].Pi for i in m._set_covering.keys()}			
				λ={i:m._Num_veh[i].Pi for i in m._Num_veh.keys()}

				clients_h=list(self.Adj[d][self.Adj[d]==1].index)
				clients_h=list(set(clients.index) & set(clients_h))
				clients_h=clients.loc[clients_h]				
				time_genPet=time.time()
				
				r_costs=sp.runJavaPulse(d, clients_h,π,λ[d],nRoutes=nRoutes,tw=tw)

				if r_costs-λ[d]<0:					
					term=False
					#break
			m=sp.master_problem(relaxed=True)
			time_genPet=time.time()
			m.optimize()
			sp.z_hat={k:kk.x for k,kk in m._z.items()}
			#Update route scores
			sp.updateScores()
			if n_it%self.nItemptyRoutes==0:
				sp.updateSetOfRoutes(self.maxNumRoutes)				
			
			print(f'NumRoutes: {len(sp.Rid)}')
			FO.append(m.objVal)

						
			if n_it>2:
				print('\t\tMejora de: '+str((FO[-2]-FO[-1])/FO[-1]))
				print('\t\tVoy en de: '+str(FO[-1]))
				if (FO[-2]-FO[-1])/FO[-1]<gap:
					term=True
			
			if n_it==50 or term:
				print('\t','El numero de iteraciones fue: ',n_it)
				print(FO)
				break
			
		print_log('\tTermine con la GC..')
				
		m=sp.master_problem(relaxed=True)
		m.optimize()

		#Uses this solution to start next time
		sp.z_hat={k:kk.x for k,kk in m._z.items()}	

		#Recovers the duals of the relaxed
		λ={i:m._Num_veh[i].Pi for i in m._Num_veh.keys()}
		π={i:m._set_covering[i].Pi for i in m._set_covering.keys()}	

		return m.objVal, λ, π

	def solveSpJavaNotAllDepots(self,sp,nRoutes=10,gap=0.001):
		'''
		CG-not calling subproblems for every iteration
		'''

		FO=[]
		clients=sp.D.iloc[:]
		π={i:0 for i in clients.index}

		n_it=0
		HH=[i for i in sp.H if sp.y_hat[i]>0]


		while True:
			n_it+=1			
			m=sp.master_problem(relaxed=True)
			time_genPet=time.time()
			m.optimize()
			sp.z_hat={k:kk. x for k,kk in m._z.items()}	

			#print('\t',f'Me demoro {time.time()-time_genPet} segundos resolviendo la relajación')
			FO.append(m.objVal)
			#print('\t',f'el status del problema es {m.getAttr("Status")}')
			π={i:m._set_covering[i].Pi for i in m._set_covering.keys()}			
			λ={i:m._Num_veh[i].Pi for i in m._Num_veh.keys()}
			#print('\t πs')
			#print('\t',π)
			#print('\t λs')
			#print('\t',λ)

			def calcSumPi(d):
				clients_h=list(self.Adj[d][self.Adj[d]==1].index)
				clients_h=list(set(clients.index) & set(clients_h))
				#clients_h=clients.loc[clients_h]							
				return -sum(π[i] for i in clients_h)

			term=True

			HH=sorted(HH,key=calcSumPi)

			n=np.random.randint(1,len(HH)-1)

			for d in HH[:n]:
				#cord=self.V.loc[d][['lat','long']]	
				#clients_h=list(self.V.loc[distance(self.V.loc[:,'lat'],self.V.loc[:,'long'],cord.loc['lat'],cord.loc['long'])<(self.route_time_limit/5) *40].index)			
				clients_h=list(self.Adj[d][self.Adj[d]==1].index)
				clients_h=list(set(clients.index) & set(clients_h))
				clients_h=clients.loc[clients_h]

				#print(f'\t\tEmpcece con petal gen del depot {d} en la it {n_it}')
				time_genPet=time.time()
				
				r_costs=sp.runJavaPulse(d, clients_h,π,λ[d],nRoutes=nRoutes)
				#print(f'\t\tRouteGen{d}')

				#print('\t\t',f'Me demoro {time.time()-time_genPet} segundos')

				if r_costs-λ[d]<0:
					#print('\t\t',r_costs)
					#print('\t\tLas λ dan',λ[d])
					term=False
					#break
			
			#print('\t',f'Voy {n_it} iteraciones del subproblema')
			#print('\t',FO[-1])
			#print(FO)
			#if n_it==1:
			#	break
			#	print(FO)
			if n_it>2:
				print('\t\tMejora de: '+str((FO[-2]-FO[-1])/FO[-1]))
				print('\t\tVoy en de: '+str(FO[-1]))
				if (FO[-2]-FO[-1])/FO[-1]<gap:
					#term=True
					pass
			
			if n_it==50 or term:
				print('\t','El numero de iteraciones fue: ',n_it)
				print(FO)
				break
		print_log('\tTermine con la GC..')
		
		
		m=sp.master_problem(relaxed=True)
		m.optimize()

		#Uses this solution to start next time
		sp.z_hat={k:kk.x for k,kk in m._z.items()}	

		#Recovers the duals of the relaxed
		λ={i:m._Num_veh[i].Pi for i in m._Num_veh.keys()}
		π={i:m._set_covering[i].Pi for i in m._set_covering.keys()}				

		'''
		m=sp.master_problem(relaxed=False)		
		print_log('\tEmpece a resolver con integralidad')		
		time_genPet=time.time()
		m.optimize()		
		print_log('\t',f'Me demoro {time.time()-time_genPet} segundos resolviendo sub problema {sp.id} con integralidad')

		#Guarda las rutas seleccionadas para cada depot en sp.OptRi
		for d in sp.H:
			routes=[i for i,j in m._z.items() if j.x==1 and sp.R[i][0]==d]			
			if d not in sp.OptRi.keys():
				sp.OptRi[d]=routes
			else:
				sp.OptRi[d]+=routes		
		'''
		#if plot:
		if True:
			
			#print_log('\t\t',sp.id,'\n\t\t',sp.R,'\n\t\t',sp.Route_cost,'\n\t\t',sp.OptRi)
			m=sp.master_problem(relaxed=False)
			m.optimize()
			
			sp.export_results({i:m._z[i].x for i in m._z.keys()})
			pet=[i for i,p in enumerate(m._z) if m._z[i].x>.5]
			routes=[sp.R[i] for i in pet]
			sp.OptRi=routes
			#self.plot_dem(sp.D)
			#self.plot_petals(routes,sp.H,pet=True)
			#plt.show()
			'''
			os.chdir('../Plots/Subproblems')
			if sp.id==0:delete_all_files(os.getcwd())
			self.plot_dem(sp.D)
			self.plot_petals(routes,sp.H,pet=False)
			plt.savefig(f'sp_{sp.id}.png')
			#plt.show()
			plt.clf()			
			os.chdir('..')
			os.chdir('../Data')
			'''
			
			foi=sum(sp.Route_cost[r]*m._z[r].x for r in range(len(sp.R)))
			return m.objVal, λ, π
		else:
			return m.objVal, λ, π

	def solveSpNoCG(self,sp):
		m=sp.master_problem(relaxed=True)
		m.optimize()
				
		#Uses this solution to start next time
		sp.z_hat={k:kk.x for k,kk in m._z.items()}	
		#print([ (j,i) for j,i in sp.z_hat.items() if i>0])
		#Update route scores
		sp.updateScores()

		#Recovers the duals of the relaxed
		λ={i:m._Num_veh[i].Pi for i in m._Num_veh.keys()}
		π={i:m._set_covering[i].Pi for i in m._set_covering.keys()}				

		'''
		m=sp.master_problem(relaxed=False)		
		print_log('\tEmpece a resolver con integralidad')		
		time_genPet=time.time()
		m.optimize()		
		print_log('\t',f'Me demoro {time.time()-time_genPet} segundos resolviendo sub problema {sp.id} con integralidad')

		#Guarda las rutas seleccionadas para cada depot en sp.OptRi
		for d in sp.H:
			routes=[i for i,j in m._z.items() if j.x==1 and sp.R[i][0]==d]			
			if d not in sp.OptRi.keys():
				sp.OptRi[d]=routes
			else:
				sp.OptRi[d]+=routes		
		'''
		#if plot:
		if False:
			
			#print_log('\t\t',sp.id,'\n\t\t',sp.R,'\n\t\t',sp.Route_cost,'\n\t\t',sp.OptRi)
			#m=sp.master_problem(relaxed=False)
			#m.optimize()
			
			sp.export_results({i:m._z[i].x for i in m._z.keys()})
			pet=[i for i,p in enumerate(m._z) if m._z[i].x>.5]
			routes=[sp.R[i] for i in pet]
			#self.plot_dem(sp.D)
			#self.plot_petals(routes,sp.H,pet=True)
			#plt.show()
			os.chdir('../Plots/Subproblems')
			if sp.id==0:delete_all_files(os.getcwd())
			self.plot_dem(sp.D)
			self.plot_petals(routes,sp.H,pet=False)
			plt.savefig(f'spNoCG_{sp.id}.png')
			#plt.show()
			plt.clf()			
			os.chdir('..')
			os.chdir('../Data')
			
			foi=sum(sp.Route_cost[r]*m._z[r].x for r in range(len(sp.R)))
			return m.objVal, λ, π
		else:
			return m.objVal, λ, π
	
	def create_cluters(self,N):
		nodes=list(self.V.index)
		points=[[self.V.loc[i]['lat'],self.V.loc[i]['long']] for i in nodes]

		#N=[1,3,6,10]
		
		All_clusters=[]
		Centers=[]
		for n in N:
			# create kmeans object
			kmeans = KMeans(n_clusters=n,random_state=5)
			# fit kmeans object to data
			kmeans.fit(points)

			# print location of clusters learned by kmeans object
			centers=kmeans.cluster_centers_			
			Centers+=list(centers)
			# save new clusters for chart
			y_km = kmeans.fit_predict(points)		

			#Nodes in each class
			casses=[[k for j,k in enumerate(nodes) if y_km[j]==i ] for i in range(n)]
			All_clusters.append(casses)
		
		return All_clusters, Centers

	def centroid(self,nodes, centroid):
		'''
		For a list of nodes and its centroid estiates its medioid, (i.e., the closest node to the centroid)
		Input:
			node (list): List of nodes 
			centroid: centroid of that cluster

		Output:
			n: medioid
		'''
		
		points=[(self.V.loc[i]['lat']-centroid[0])**2+(self.V.loc[i]['long']-centroid[1])**2 for i in nodes]
		n=nodes[points.index(min(points))]
		return n

	def possDepots(self,N):
		clusts,cent=self.create_cluters(N)
		clusts=sum(clusts,[])
		centers=list(map(lambda x: self.centroid(*x),zip(clusts,cent)))
		return set([self.f.loc[[i for i in self.G.nodes if self.dist_m[c][i]*self.t_mile<0.5]].idxmin().iloc[0] for c in centers])

	def centroidNoNocluster(self, centroid):
		'''
		For a list of nodes and its centroid estiates its medioid, (i.e., the closest node to the centroid)
		Input:
			node (list): List of nodes 
			centroid: centroid of that cluster

		Output:
			n: medioid
		'''
		print(centroid)
		points=[(self.V.loc[i]['lat']-centroid[0])**2+(self.V.loc[i]['long']-centroid[1])**2 for i in self.V.index]
		n=self.V.index[points.index(min(points))]
		#print('n',n)
		return n

	def possDepotsNoNocluster(self,cent):
		centers=list(map(lambda x: self.centroidNoNocluster(x),cent))
		#print(centers)
		#
		return set(centers)

	def export_results(self,depots,veh_depot):
		'''
		Exports the first stage results into a .txt
		Input:
			depots (dict): Dictionaty with the solution of the x vatraibles of the benders algorithm
			veh_depot (DICT): Dictionaty with the solution of the y vatraibles of the benders algorithm
		Output:
			None
		'''

		os.chdir('..')
		os.chdir('Results')
		#print(os.getcwd())

		file = open(f"First_stage{self.h}.txt","w")
		file.write('Depot'+'\t'+'Number of vehicles'+'\n')
		for i in depots.keys():
			if depots[i]>0:
				file.write(str(i)+'\t'+str(veh_depot[i])+'\n')

		file.close() 

		os.chdir('..')
		os.chdir('Data')

class sub_problem():
	"""docstring for sub_problem"""
	def __init__(self,master,id,H, D,dm,pos=[]):
		#super().__init__()		

		self.master=master

		self.id=id
		#Route time limit
		self.route_time_limit=9
		#Subset of depots
		self.H=H
		#Demands (day): dict(node:demand)
		self.D=D

		self.Clients=list(D.index)
		#Distance matrix
		self.dm=dm

		#Cost
		self.cost=None

		#y_hat
		self.y_hat=None

		#Cost of positioninng a vehicle
		self.c=30

		#Capacity of each truck
		self.cap=60

		#Artifitial graph
		self.SG=nx.DiGraph()
		self.pos=pos

		#Routes of depot
		self.Ri={}

		#Routes selected for node i
		self.OptRi={}

		#Set of routes
		self.R={}

		#Set of routes
		self.Rid=[]

		#Number of route 

		self.nRoutes=0

		#Route scores		
		self.RScores={}

		#Time in routes set
		self.TInSet={}
		#Route costs
		self.Route_cost={}

		self.t_mile=(1/40)
		self.c_mile=(.7)

		#Adjacency matrix
		self.Adj=None
		#self.m=self.master_problem(relaxed=True)

		self.z_hat={i:0 for i in range(len(self.R))}

		#Depot lower time window
		self.DLtw=0
		#Depot upper time window
		self.DUtw=11

		#Depot lower time window
		self.nLtw=2
		self.nUtw=10
	
	def save(self,name):
		'''
		Writes a binary file with the object
		'''	
		m=self.master
		self.master=None
		file_sp=open(f'{name}.sp','wb')
		pickle.dump(self, file_sp)
		file_sp.close()
		self.master=m

	def master_problem(self,relaxed=False):
		'''
		Defines the master problem
		Input:
			relaxed (boolean): True if relaxed
		Output:
			m: Gurobi model.
		'''
		m=Model('SP master')
		m.setParam('OutputFlag', 0)

		#Decition variables
		if relaxed:
			#penal=(max(100,sum(self.Route_cost))/(max(1,len(self.Route_cost))))
			penal=(30+20)
			#print('\tRoute cost vs penal',sum(self.Route_cost)/len(self.Route_cost),penal)
			m._z={p:m.addVar(vtype=GRB.CONTINUOUS,name="m._z_"+str(p),lb=0, ub=1,obj=self.Route_cost[p]) for p in self.Rid}
			#m._v={i:m.addVar(vtype=GRB.CONTINUOUS,name="m._v_"+str(i),lb=0, ub=10,obj=penal) for i in self.y_hat.keys()}
			m._v={i:m.addVar(vtype=GRB.CONTINUOUS,name="m._v_"+str(i),lb=0, ub=1,obj=penal+min([(self.dm[h][i]+self.dm[i][h])*self.c_mile for h in self.H])) for i in self.Clients}
			#m._v={i:m.addVar(vtype=GRB.CONTINUOUS,name="m._v_"+str(i),lb=0, ub=1,obj=penal) for i in self.Clients}
		else:
			m._z={p:m.addVar(vtype=GRB.BINARY,name="m._z_"+str(p),obj=self.Route_cost[p]) for p in self.Rid}
			#penal=(sum(self.Route_cost)/(max(1,len(self.Route_cost))))*100
			penal=(30+20)
			#print('\tRoute cost vs penal',self.Route_cost,penal)
			#m._v={i:m.addVar(vtype=GRB.INTEGER,name="m._v_"+str(i),lb=0, ub=10,obj=penal) for i in self.y_hat.keys()}
			m._v={i:m.addVar(vtype=GRB.BINARY,name="m._v_"+str(i),lb=0, ub=1,obj=penal+min([(self.dm[h][i]+self.dm[i][h])*self.c_mile for h in self.H])) for i in self.Clients}
			#m._v={i:m.addVar(vtype=GRB.BINARY,name="m._v_"+str(i),lb=0, ub=1,obj=penal) for i in self.Clients}
		#Funcion objetivo
		#fo=quicksum(self.Route_cost[p]*m._z[p] for p in range(len(self.R)))
		#m.setObjective(fo)
		
		#Restricciones
		#1.) Set covering constraint: Each node with positive demand (i.e. i∈D^s) needs to be visited by at least one vehicle.
		
		m._set_covering={i:m.addConstr(quicksum(m._z[p]  for p in self.Rid if i in self.R[p])+m._v[i]>=1) for i in self.Clients}		

		#2.) Not exceeding number of vehicles per depot.

		m._Num_veh={i:m.addConstr(quicksum(m._z[p] for p in self.Rid if self.R[p][0]==i )<=self.y_hat[i]) for i in self.y_hat.keys()}

		
		#Uses saved basis:
		#for i in self.z_hat.keys():
		#	m._z[i].start=self.z_hat[i]

		m.update()

		return m

	def export_results(self,routes):
		'''
		Export second stage results,
		Input:
			routes (dict): Dictionary with the id of the route as key, and the solution of the z variables as values.
		Output:
			None
		'''
		try: 
			os.chdir('..')
			os.chdir('Results')
			#print(os.getcwd())
			file = open("Second_stage_{}.txt".format(self.id),"w")		
			#print(routes)
			for i in self.H:
				file.write('Depot: {}\nNumber of vehicles{}\n'.format(i,self.y_hat[i]))
				file.write('Route\tCost\n')
				for r in routes.keys():
					#print(r)
					if routes[r]>0 and self.R[r][0]==i:
						file.write(str(self.R[r])+'\t'+str(self.Route_cost[r])+'\n')
			file.close()
			os.chdir('..')
			os.chdir('Data')
		except: 
			pass

	def printDymacs(self,depot,clients,π,λ,nRoutes=10):
		
		delete_all_files('../Java/Dymacs/',exceptions=['.jar'])
		auxs=math.ceil(clients.sum()/self.cap)
		
		

		f=open(f'../Java/Dymacs/dymacsSp_{depot}.txt','w')
		ttt=time.time()
		#Adds node 0 (start node for the sp) and its arcs
		f.write('Nodes\n')
		#Node\tdem\treplen\tw1\tw2
		solve=[]
		for j in range(auxs+2):
			if j==0:
				#6-8->0-2
				f.write(f'{j}\t{0}\t{1}\t{self.DLtw*60}\t{self.DUtw*60}\n')
			elif j==auxs+1:
				#8-17->2-11
				f.write(f'{j}\t{0}\t{1}\t{self.nLtw*60}\t{self.DUtw*60}\n')
			else:
				#8-16->2-10
				f.write(f'{j}\t{0}\t{1}\t{self.nLtw*60}\t{self.nUtw*60}\n')
		for i in clients.index:
			#8-16->2-8
			if π[i]>0:
				solve.append(i)
			#TODO: Modificar Tw por para acelerar pulso.... particionar los nodos por depot	
			f.write(f'{i}\t{clients[i]}\t{0}\t{self.nLtw*60}\t{self.nUtw*60}\n')

		nArcs=0
		nNodes=len(clients.index)+auxs+2

		f.write('Arcs\n')
		for ii,i in enumerate(clients.index):
		#for ii,i in enumerate(solve):
			for j in range(auxs+2):				
				if j==0:					
					nArcs+=1
					f.write(str(j)+'\t'+str(i)+'\t'+str(int(100*(self.dm[depot][i]*self.c_mile- π[i])))+'\t'+str(int(60*self.dm[depot][i]*self.t_mile))+'\n')
					#f.write(str(i)+'\t'+str(j)+'\t'+str(int(100*(self.dm[i][depot]*self.c_mile)))+'\t'+str(int(60*self.dm[i][depot]*self.t_mile))+'\n')
				elif j==auxs+1:
					nArcs+=1
					#f.write(str(j)+'\t'+str(i)+'\t'+str(int(100*(self.dm[depot][i]*self.c_mile- π[i])))+'\t'+str(int(60*self.dm[depot][i]*self.t_mile))+'\n')
					f.write(str(i)+'\t'+str(j)+'\t'+str(int(100*(self.dm[i][depot]*self.c_mile)))+'\t'+str(int(60*self.dm[i][depot]*self.t_mile))+'\n')
				else:
					nArcs+=2
					f.write(str(j)+'\t'+str(i)+'\t'+str(int(100*(self.dm[depot][i]*self.c_mile- π[i])))+'\t'+str(int(60*self.dm[depot][i]*self.t_mile))+'\n')
					f.write(str(i)+'\t'+str(j)+'\t'+str(int(100*(self.dm[i][depot]*self.c_mile)))+'\t'+str(int(60*self.dm[i][depot]*self.t_mile))+'\n')			
			
			for j in clients.index[ii+1:]:
			#for j in solve[ii+1:]:
				nArcs+=2
				f.write(str(j)+'\t'+str(i)+'\t'+str(int(100*(self.dm[j][i]*self.c_mile- π[i])))+'\t'+str(int(60*self.dm[j][i]*self.t_mile))+'\n')
				f.write(str(i)+'\t'+str(j)+'\t'+str(int(100*(self.dm[i][j]*self.c_mile- π[j])))+'\t'+str(int(60*self.dm[i][j]*self.t_mile))+'\n')
			
		#print(f'me demoro {tiempo.time()-ttt} segs')
		f.close()

		'''
		DataFile:USA-road-NY.txt
		Number of Arcs:733846
		Number of Nodes:264346
		Time Constraint:943100
		Start Node:1
		End Node:14676
		'''		

		f=open(f'../Java/Dymacs/config.txt','w')
		f.write(f'DataFile:dymacsSp_{depot}.txt'+'\n')
		f.write(f'Number of Arcs:{nArcs}'+'\n')
		f.write(f'Number of Nodes:{nNodes}'+'\n')
		f.write(f'Capacity:{self.cap}'+'\n')
		f.write(f'Start Node:{0}'+'\n')
		f.write(f'End Node:{auxs+1}'+'\n')
		f.write(f'Depot id:{depot}\n')
		f.write(f'λ:{λ*100}\n')
		f.write(f'Num routes:{nRoutes}\n')
		f.write(f'Route id:{len(self.R)}\n')
		f.write(f't mile:{self.t_mile}\n')
		f.write(f'c mile:{self.c_mile}\n')
		f.write(f'time Limit:{5}')
		
		#t mile:0.025
		#c mile:0.7
		f.close()

	def printDymacsTW(self,depot,clients,π,λ,nRoutes=10):
		
		delete_all_files('../Java/Dymacs/',exceptions=['.jar'])
		auxs=math.ceil(clients.sum()/self.cap)
		

		classifier=self.classifierGen(depot,clients)

		f=open(f'../Java/Dymacs/dymacsSp_{depot}.txt','w')
		ttt=time.time()
		#Adds node 0 (start node for the sp) and its arcs
		f.write('Nodes\n')
		#Node\tdem\treplen\tw1\tw2
		solve=[]
		for j in range(auxs+2):
			if j==0:
				#6-8->0-2
				f.write(f'{j}\t{0}\t{1}\t{self.DLtw*60}\t{self.DUtw*60}\n')
			elif j==auxs+1:
				#8-17->2-11
				f.write(f'{j}\t{0}\t{1}\t{self.nLtw*60}\t{self.DUtw*60}\n')
			else:
				#8-16->2-10
				f.write(f'{j}\t{0}\t{1}\t{self.nLtw*60}\t{self.nUtw*60}\n')
		
		#Random order for TW strategy
		ro=np.random.permutation([0,1])
		for i in clients.index:
			#8-16->2-8
			if π[i]>0:
				solve.append(i)
			#print(clients.loc[i][0])
			nLtw,nUtw=classifier(i,ro)
			try:
				f.write(f'{i}\t{clients[i]}\t{0}\t{nLtw*60}\t{nUtw*60}\n')
			except:
				f.write(f'{i}\t{clients.loc[i][0]}\t{0}\t{nLtw*60}\t{nUtw*60}\n')

		nArcs=0
		nNodes=len(clients.index)+auxs+2

		f.write('Arcs\n')
		for ii,i in enumerate(clients.index):
		#for ii,i in enumerate(solve):
			for j in range(auxs+2):				
				if j==0:					
					nArcs+=1
					f.write(str(j)+'\t'+str(i)+'\t'+str(int(100*(self.dm[depot][i]*self.c_mile- π[i])))+'\t'+str(int(60*self.dm[depot][i]*self.t_mile))+'\n')
					#f.write(str(i)+'\t'+str(j)+'\t'+str(int(100*(self.dm[i][depot]*self.c_mile)))+'\t'+str(int(60*self.dm[i][depot]*self.t_mile))+'\n')
				elif j==auxs+1:
					nArcs+=1
					#f.write(str(j)+'\t'+str(i)+'\t'+str(int(100*(self.dm[depot][i]*self.c_mile- π[i])))+'\t'+str(int(60*self.dm[depot][i]*self.t_mile))+'\n')
					f.write(str(i)+'\t'+str(j)+'\t'+str(int(100*(self.dm[i][depot]*self.c_mile)))+'\t'+str(int(60*self.dm[i][depot]*self.t_mile))+'\n')
				else:
					nArcs+=2
					f.write(str(j)+'\t'+str(i)+'\t'+str(int(100*(self.dm[depot][i]*self.c_mile- π[i])))+'\t'+str(int(60*self.dm[depot][i]*self.t_mile))+'\n')
					f.write(str(i)+'\t'+str(j)+'\t'+str(int(100*(self.dm[i][depot]*self.c_mile)))+'\t'+str(int(60*self.dm[i][depot]*self.t_mile))+'\n')			
			
			for j in clients.index[ii+1:]:
			#for j in solve[ii+1:]:
				nArcs+=2
				f.write(str(j)+'\t'+str(i)+'\t'+str(int(100*(self.dm[j][i]*self.c_mile- π[i])))+'\t'+str(int(60*self.dm[j][i]*self.t_mile))+'\n')
				f.write(str(i)+'\t'+str(j)+'\t'+str(int(100*(self.dm[i][j]*self.c_mile- π[j])))+'\t'+str(int(60*self.dm[i][j]*self.t_mile))+'\n')
			
		#print(f'me demoro {tiempo.time()-ttt} segs')
		f.close()

		'''
		DataFile:USA-road-NY.txt
		Number of Arcs:733846
		Number of Nodes:264346
		Time Constraint:943100
		Start Node:1
		End Node:14676
		'''		

		f=open(f'../Java/Dymacs/config.txt','w')
		f.write(f'DataFile:dymacsSp_{depot}.txt'+'\n')
		f.write(f'Number of Arcs:{nArcs}'+'\n')
		f.write(f'Number of Nodes:{nNodes}'+'\n')
		f.write(f'Capacity:{self.cap}'+'\n')
		f.write(f'Start Node:{0}'+'\n')
		f.write(f'End Node:{auxs+1}'+'\n')
		f.write(f'Depot id:{depot}\n')
		f.write(f'lambda:{λ*100}\n')
		f.write(f'Num routes:{nRoutes}\n')
		f.write(f'Route id:{len(self.R)}\n')
		f.write(f't mile:{self.t_mile}\n')
		f.write(f'c mile:{self.c_mile}\n')
		f.write(f'time Limit:{5}')
		
		#t mile:0.025
		#c mile:0.7
		f.close()

	def readDymacs(self,path='../Java/Dymacs/results.txt'):
		f=open(path,'r')
		try:
			cost=float(f.readline().split(':')[1])
		except:
			cost=0
		Routes=[]
		ii=0		
		while True:
			route=[]
			try:			
				lenR=int(f.readline().split(':')[1])
			except:
				break
			#print('la primera tiene\t',lenR)
			for i in range(lenR):
				l=f.readline()
				route.append(int(l))
				#print(l)
			if lenR==0:
				l=f.readline()
			else:
				Routes.append(route)

			if ii==300 or l=='':
				#print(i)
				break
			
			ii+=1
		f.close()
		#print(cost/100,route)
		return cost/100, Routes

	def runJavaPulse(self,depot,clients,π,λ,nRoutes=10,tw=False):

		if tw:
			self.printDymacsTW(depot,clients,π, λ,nRoutes)
		else:
			self.printDymacs(depot,clients,π, λ,nRoutes)

		dir=os.getcwd()
		os.chdir('../Java')
		os.system('java -jar pulse.jar')

		r_costs, Route=self.readDymacs()
		for route in Route:	
			n=self.nRoutes		
			if route[0] in self.Ri.keys():
				self.Ri[route[0]].append(self.nRoutes)
			else:
				self.Ri[route[0]]=[self.nRoutes]			
			
			if route[0] not in self.master.possibleDepots:
				print(f'esto esta pasando {route[0]}')
				raise Exception('ALgo raro raro')
			#time=self.c_mile*sum(self.dm[k][l] for k,l in zip(route[:-1],route[1:]))
			cost=self.c_mile*sum(self.dm[k][l] for k,l in zip(route[:-1],route[1:]))
			
			self.R[n]=route
			self.Rid.append(self.nRoutes)
			self.Route_cost[n]=cost
			self.nRoutes+=1
		os.chdir(dir)
		return r_costs

	def runJavaPetalRecycler(self,R):
		curp=os.getcwd()

		f=open(f'../Java/Preprocess/routes.txt','w')
		for i in R:
			f.write(str(i)[1:-1].replace(' ','')+"\n")
		f.close()

		f=open(f'../Java/Preprocess/demand.txt','w')
		solve=[]

		for i in self.H:
			f.write(f'{i}\t{0}\t{0*60}\t{9*60}\n')		
		for i in self.D.index:
			#8-16->2-8
			if i in self.H:
				f.write(f'{i}\t{self.D[i]}\t{0*60}\t{9*60}\n')
			else:
				f.write(f'{i}\t{self.D[i]}\t{2*60}\t{8*60}\n')
		f.close()

		f=open(f'../Java/Preprocess/config.txt','w')
		f.write(f'DataFile:demand.txt'+'\n')		
		f.write(f'Number of Nodes:{1457}'+'\n')
		f.write(f'Number of Nodes:{len(self.D)}'+'\n')
		f.write(f'Capacity:{self.cap}'+'\n')
		f.write(f'Route id:{len(self.R)}\n')
		f.write(f't mile:{self.t_mile}\n')
		f.write(f'c mile:{self.c_mile}')
		f.close()
		os.chdir('../Java')
		os.system('java -jar preprocess.jar')		
		
		r_costs, Route=self.readDymacs('Preprocess/results.txt')
		for route in Route:
			n=self.nRoutes
			if route[0] in self.Ri.keys():
				self.Ri[route[0]].append(len(self.R))
			else:
				self.Ri[route[0]]=[len(self.R)]
			
			#time=self.c_mile*sum(self.dm[k][l] for k,l in zip(route[:-1],route[1:]))
			cost=self.c_mile*sum(self.dm[k][l] for k,l in zip(route[:-1],route[1:]))
			
			self.R[n]=route
			self.Route_cost[n]=cost

			self.nRoutes+=1
		
		os.chdir(curp)

	def nodeTimeWindow(self,depot,node,order=[0,1]):
		'''
		
		'''		
		o=np.array(self.pos[depot])
		p=np.array(self.pos[node])
		
		p=p-o
		tw=[(self.nLtw,self.nUtw/2),(self.nLtw+self.nUtw/2,self.nUtw)]
		if p[0]<=0:
			return tw[order[0]]
		else:
			return tw[order[1]]
		'''
		if p[0]<=0 and p[1]>=0:
			return (self.nLtw,self.nUtw/4)
		elif p[0]>=0 and p[1]>=0:
			return (self.nLtw+self.nUtw/4,2*self.nUtw/4)
		elif p[0]<=0 and p[1]<=0:
			return (self.nLtw+2*self.nUtw/4,3*self.nUtw/4)
		elif p[0]>=0 and p[1]<=0:
			return (self.nLtw+3*self.nUtw/4,self.nUtw)
		'''
	
	def classifierGen(self,depot,clients):
		
		o=np.array(self.pos[depot])
		XY=np.array([self.pos[i] for i in clients.index])
		X,Y=[i[0] for i in XY],[i[1] for i in XY]
		
		a=normal(o-np.array([np.mean(X),np.mean(Y)]))
		b=a.dot(o)
		minus=0.5
		tw=[(self.nLtw,self.nLtw+(self.nUtw-self.nLtw)/2-minus),(self.nLtw+(self.nUtw-self.nLtw)/2,self.nUtw-minus)]
		tw=list(map(lambda x: (int(x[0]),int(x[1])),tw))
		def classifier(client,order=[0,1]):
			p=np.array(self.pos[client])
			if a.dot(p)>=b:
				return tw[order[0]]
			else: 
				return tw[order[1]]

		return classifier

	def updateScores(self):
		'''
		Updates the scores of the routes
		'''		
		#print(self.Rid)
		for r in self.Rid:
			try:				
				self.TInSet[r]+=1
				self.RScores[r]+=self.z_hat[r]
			except:
				self.TInSet[r]=1
				#print(self.z_hat)
				self.RScores[r]=self.z_hat[r]

	def updateSetOfRoutes(self,n):
		'''
		Uses the scores to keep the first n routes based on the score in the set of routes
		'''
		#self.updateScores()
		
		gamma=3
		beta=2
		tScore=lambda t: exp(gamma/(t**beta))-1

		for i,idis in self.Ri.items():
			
			if n<len(idis):
				R=sorted(idis,key=lambda x: -(self.RScores[x] +tScore(self.TInSet[x]) ))
				print(f'\t\tScores:\n\t\t{[ (self.RScores[x],self.TInSet[x]) for x in R]}')
				self.Ri[i]=R[:n]
				Rno=R[n+1:]
				for ii in Rno:
					self.Rid.remove(ii)
					del self.RScores[ii]
					del self.R[ii]
					del self.Route_cost[ii]
					del self.TInSet[ii]
			#print(f'\t\tDepot: {i}\n\t\t#Routes:{len(self.Ri[i])}')		
		#list(map(self.RScores.__delitem__, filter(self.RScores.__contains__,Rno)))
		print(f'\t\tRoutes per depot: {[(i,len(l)) for i,l in self.Ri.items()]}')
	
	def restartScores(self):
		for ii in self.Rid:
			self.RScores[ii]=0
			self.TInSet[ii]=0

#Some functions
def normal(x):
	'''
	Returns a normal vector to the one given by parameter (only for x \in \mathbb{R}^2)
	'''
	return np.array([-x[1],x[0]])
def delete_all_files(path,exceptions=[]):
	'''
	Delts all files from a path with th exeption of exception
	Input:
		path: path from where the files will be deleted
		exception (list): List with exceptions for deleting (e.g., ['.pdf','.tex']).
	'''
	#files = glob.glob(path)

	for filename in os.listdir(path):
		file_path = os.path.join(path, filename)

		if sum(i in filename for i in exceptions)==0:		
			try:
				if os.path.isfile(file_path) or os.path.islink(file_path):
					os.unlink(file_path)
				elif os.path.isdir(file_path):
					shutil.rmtree(file_path)
			except Exception as e:
				print('Failed to delete %s. Reason: %s' % (file_path, e))
def print_log(*args):
	global log_text,log_path
	try: 
		log_text=open(log_path,'a')
		line=''
		for a in args:		
			line+=str(a)
		log_text.write(line+'\n')
		log_text.close()
	except:
		pass 
	


if __name__=='__main__':
	pass
