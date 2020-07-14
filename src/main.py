from gurobipy import *
import pandas as pd
import os 
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation
import random
import numpy as np
import scipy.stats as st
import math

import time

from sklearn.cluster import KMeans

import os,shutil
import glob

#import TSP
#import pulse_mod as pulse

from matplotlib import cm

from pylatex import Document, Section, Subsection, Tabular, Math, TikZ, Axis,Plot, Figure, Matrix, Alignat,Command
from pylatex.utils import italic, NoEscape

import matplotlib.patches as patches


from scipy.optimize import curve_fit
from scipy.special import factorial
import scipy.stats as sts

try: 
	from concorde.tsp import TSPSolver
	from concorde.tests.data_utils import get_dataset_path
except:
	print("Warning: Concorde cannot be used in Windows operating system")
	pass 

import community as community_louvain

import dill
import sys

import pickle
'''
try:
	os.chdir('/Users/davidcorredor/Universidad de los Andes/MOPTA - MOPTA/Implementation/src')
except:
	pass
'''
plt.rcParams.update({'figure.max_open_warning': 0})

global fsize

scal=1
fsize=(16*scal, 10*scal)
#plt.figure(num=None, figsize=(16*scal, 10*scal), dpi=80, facecolor='w', edgecolor='k')



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
	

class master():

	"""docstring for master"""
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

				#print(self.dist_m.head())		
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

		
		

		#List for animation...
		self.AnimNVeh=[{15051: 2.0, 16627: 1.0, 18617: 1.0}, {15051: 2.0, 17017: 3.0, 16627: 1.0}, {15051: 2.0, 17017: 3.0, 16627: 1.0, 18617: 1.0}, {15051: 2.0, 16627: 1.0, 18617: 1.0, 18074: 4.0}, {15051: 2.0, 17017: 3.0, 16627: 1.0, 18074: 4.0}, {15051: 2.0, 16627: 1.0, 18617: 14.0}, {15051: 2.0, 17017: 16.0, 16627: 1.0}, {15051: 11.0, 16627: 2.0, 18617: 7.0}, {15051: 8.0, 17017: 7.0, 16627: 7.0}, {15051: 2.0, 17017: 3.0, 16627: 1.0, 18617: 1.0, 18074: 4.0}, {15051: 2.0, 17017: 3.0, 16627: 2.0, 18617: 11.0}, {15051: 7.0, 16627: 5.0, 18617: 1.0, 18074: 4.0}, {15051: 2.0, 16627: 2.0, 18617: 10.0, 18074: 4.0}, {15051: 11.0, 17017: 3.0, 16627: 3.0, 18617: 3.0}, {15051: 2.0, 17017: 3.0, 16627: 11.0, 18074: 4.0}, {15051: 7.0, 16627: 6.0, 18617: 17.0}, {15051: 11.0, 17017: 3.0, 16627: 1.0, 18617: 1.0, 18074: 4.0}, {15051: 12.0, 16627: 1.0, 18617: 9.0, 18074: 4.0}, {15051: 12.0, 17017: 3.0, 16627: 1.0, 18617: 14.0}, {15051: 12.0, 17017: 20.0, 16627: 1.0}, {15051: 2.0, 17017: 3.0, 16627: 10.0, 18617: 20.0, 18074: 4.0}, {15051: 20.0, 17017: 13.0, 16627: 1.0, 18074: 4.0}, {15051: 2.0, 16627: 20.0, 18617: 8.0, 18074: 4.0}, {15051: 20.0, 17017: 3.0, 16627: 1.0, 18617: 20.0, 18074: 4.0}, {15051: 2.0, 17017: 11.0, 16627: 20.0, 18617: 2.0}, {15051: 20.0, 16627: 11.0, 18617: 5.0, 18074: 4.0}, {15051: 20.0, 17017: 3.0, 16627: 20.0, 18617: 1.0, 18074: 4.0}, {15051: 20.0, 17017: 3.0, 16627: 14.0, 18074: 4.0}, {15051: 5.0, 16627: 3.0, 18617: 2.0, 18074: 5.0}, {15051: 5.0, 16627: 3.0, 18617: 4.0, 18074: 4.0}]
		#S[{15632: 1.0, 16434: 1.0, 16669: 1.0, 16923: 1.0, 18660: 1.0, 18844: 1.0, 19310: 1.0}, {15632: 5.0, 16434: 1.0, 16669: 1.0, 16923: 1.0, 18660: 1.0, 18844: 1.0, 19310: 1.0}, {15632: 1.0, 16434: 1.0, 16669: 1.0, 16923: 1.0, 18660: 1.0, 18844: 1.0, 19310: 6.0}, {15632: 3.0, 16434: 1.0, 16669: 2.0, 16923: 1.0, 18660: 1.0, 18844: 1.0, 19310: 3.0}, {15632: 4.0, 16434: 2.0, 16669: 1.0, 16923: 1.0, 18660: 2.0, 18844: 1.0, 19310: 3.0}, {15632: 4.0, 16434: 1.0, 16669: 4.0, 16923: 1.0, 18660: 1.0, 18844: 1.0, 19310: 3.0}, {15632: 3.0, 16434: 1.0, 16669: 3.0, 16923: 1.0, 18660: 1.0, 18844: 3.0, 19310: 3.0}, {15632: 1.0, 16434: 7.0, 16669: 3.0, 16923: 1.0, 18660: 6.0, 18844: 1.0, 19310: 1.0}, {15632: 4.0, 16434: 6.0, 16669: 2.0, 16923: 1.0, 18660: 6.0, 18844: 1.0, 19310: 1.0}, {15632: 2.0, 16434: 7.0, 16669: 4.0, 16923: 1.0, 18660: 3.0, 18844: 1.0, 19310: 3.0}, {15632: 3.0, 16434: 7.0, 16669: 3.0, 16923: 1.0, 18660: 3.0, 18844: 1.0, 19310: 3.0}, {15632: 19.0, 16434: 1.0, 16669: 1.0, 16923: 1.0, 18660: 1.0, 18844: 2.0, 19310: 8.0}, {15632: 9.0, 16434: 1.0, 16669: 3.0, 16923: 1.0, 18660: 2.0, 18844: 2.0, 19310: 3.0}, {15632: 4.0, 16434: 1.0, 16669: 2.0, 16923: 1.0, 18660: 2.0, 18844: 1.0, 19310: 4.0}, {15632: 4.0, 16434: 1.0, 16669: 2.0, 16923: 1.0, 18660: 2.0, 18844: 1.0, 19310: 4.0}, {16249: 1.0, 18660: 1.0}, {16249: 11.0, 18660: 1.0}, {16249: 4.0, 18660: 10.0}, {16249: 13.0, 18660: 10.0}, {16249: 6.0, 18660: 5.0}, {16249: 6.0, 18660: 5.0}]
		#[{15632: 1.0, 16434: 1.0, 16669: 1.0, 16923: 1.0, 18660: 1.0, 18844: 1.0, 19310: 1.0}, {15632: 5.0, 16434: 1.0, 16669: 1.0, 16923: 1.0, 18660: 1.0, 18844: 1.0, 19310: 1.0}, {15632: 1.0, 16434: 1.0, 16669: 1.0, 16923: 1.0, 18660: 1.0, 18844: 1.0, 19310: 6.0}, {15632: 3.0, 16434: 1.0, 16669: 2.0, 16923: 1.0, 18660: 1.0, 18844: 1.0, 19310: 3.0}, {15632: 4.0, 16434: 2.0, 16669: 1.0, 16923: 1.0, 18660: 2.0, 18844: 1.0, 19310: 3.0}, {15632: 4.0, 16434: 1.0, 16669: 4.0, 16923: 1.0, 18660: 1.0, 18844: 1.0, 19310: 3.0}, {15632: 3.0, 16434: 1.0, 16669: 3.0, 16923: 1.0, 18660: 1.0, 18844: 3.0, 19310: 3.0}, {15632: 1.0, 16434: 7.0, 16669: 3.0, 16923: 1.0, 18660: 6.0, 18844: 1.0, 19310: 1.0}, {15632: 4.0, 16434: 6.0, 16669: 2.0, 16923: 1.0, 18660: 6.0, 18844: 1.0, 19310: 1.0}, {15632: 2.0, 16434: 7.0, 16669: 4.0, 16923: 1.0, 18660: 3.0, 18844: 1.0, 19310: 3.0}, {15632: 3.0, 16434: 7.0, 16669: 3.0, 16923: 1.0, 18660: 3.0, 18844: 1.0, 19310: 3.0}, {15632: 19.0, 16434: 1.0, 16669: 1.0, 16923: 1.0, 18660: 1.0, 18844: 2.0, 19310: 8.0}, {15632: 9.0, 16434: 1.0, 16669: 3.0, 16923: 1.0, 18660: 2.0, 18844: 2.0, 19310: 3.0}, {15632: 4.0, 16434: 1.0, 16669: 2.0, 16923: 1.0, 18660: 2.0, 18844: 1.0, 19310: 4.0}, {15632: 4.0, 16434: 1.0, 16669: 2.0, 16923: 1.0, 18660: 2.0, 18844: 1.0, 19310: 4.0}]
		#[{15632: 1.0, 16434: 1.0, 16669: 1.0, 16923: 1.0, 18660: 1.0, 18844: 1.0, 19310: 1.0}, {15632: 6.0, 16434: 1.0, 16669: 1.0, 16864: 1.0, 16923: 1.0, 18701: 1.0, 19310: 1.0}, {15632: 4.0, 16434: 1.0, 16669: 1.0, 16923: 1.0, 18660: 2.0, 18701: 1.0, 19310: 2.0}, {15632: 3.0, 16436: 0.999997750420129, 16669: 1.0, 16923: 0.999998500629347, 18660: 1.0, 18844: 1.0, 19310: 3.9999986672970103}, {15632: 3.0, 16434: 2.0, 16669: 1.0, 16864: 1.0, 16915: 1.0, 18660: 1.0, 18844: 1.0, 19310: 3.0}, {15632: 3.0, 16434: 1.0, 16669: 2.0, 16923: 1.0, 18660: 1.0, 18701: 1.0, 18844: 1.0, 19310: 3.0}, {15632: 3.0, 16434: 1.0, 16669: 2.0, 16864: 1.0, 16915: 1.0, 18701: 1.0, 19310: 5.0}, {15632: 3.0, 16436: 1.0, 16669: 3.0, 16923: 1.0, 18660: 1.0, 18701: 2.0, 19310: 3.0}, {15632: 3.0, 16434: 2.0, 16669: 2.0, 16864: 1.0, 16915: 1.0, 18660: 1.0, 18701: 1.0, 19310: 3.0}, {15632: 4.0, 16434: 1.0, 16669: 1.0, 16864: 1.0, 16915: 1.0, 18701: 2.0, 18844: 1.0, 19310: 3.0}, {15632: 3.0, 16434: 1.0, 16436: 1.0, 16669: 1.0, 16923: 1.0, 18660: 7.0, 18844: 1.0, 19310: 1.0}, {15632: 3.0, 16434: 2.0, 16669: 1.0, 16923: 1.0, 18651: 1.0, 18660: 4.0, 19310: 3.0}, {15632: 3.0, 16434: 1.0, 16669: 2.0, 16915: 1.0, 16923: 1.0, 18660: 4.0, 18844: 1.0, 19310: 2.0}, {15632: 3.0, 16434: 1.0000008230984083, 16436: 0.9999991782699773, 16669: 1.999999178269977, 16923: 1.0, 18660: 2.0, 18701: 1.0, 19310: 3.0}, {15632: 4.0, 16436: 2.0, 16669: 1.0, 16864: 1.0, 16923: 1.0, 18660: 3.0, 18844: 1.0, 19310: 2.0}, {15632: 3.0, 16434: 2.0, 16669: 1.0, 16864: 1.0, 16923: 1.0, 18660: 2.0, 18701: 1.0, 19310: 3.0}, {15632: 7.0, 16434: 1.0, 16669: 1.0, 16882: 1.0, 16915: 1.0, 18660: 1.0, 18844: 2.0, 19310: 3.0}, {15632: 5.0, 16436: 2.0, 16669: 1.0, 16923: 1.0, 18660: 1.0, 18701: 1.0, 18844: 2.0, 19310: 5.0}, {15632: 1.0, 16434: 2.0, 16669: 10.0, 16864: 9.0, 16923: 1.0, 18660: 1.0, 18844: 1.0, 19310: 3.0}, {15632: 2.0, 16434: 1.0, 16669: 4.0, 16864: 12.0, 16915: 1.0, 16923: 1.0, 18660: 3.0, 18844: 1.0, 19310: 3.0}, {15632: 3.0, 16434: 2.0, 16669: 3.0, 16864: 14.0, 16923: 1.0, 18701: 1.0, 18844: 1.0, 19310: 3.0}, {15632: 3.0, 16434: 1.0, 16669: 2.0, 16864: 14.0, 16915: 1.0, 18651: 1.0, 18660: 3.0, 19310: 3.0}, {15632: 4.0, 16350: 1.0, 16434: 2.0, 16669: 2.0, 16923: 1.0, 18660: 1.0, 18844: 1.0, 19310: 3.0}, {15632: 4.0, 16434: 2.0, 16669: 2.0, 16882: 1.0, 16923: 1.0, 18660: 1.0, 18844: 1.0, 19310: 3.0}, {15632: 4.0, 16434: 2.0, 16669: 2.0, 16882: 1.0, 16923: 1.0, 18660: 1.0, 18844: 1.0, 19310: 3.0}]




		self.possibleDepots=[17728, 16130, 15427, 16650, 19372, 17582, 15824, 18014, 18463]#list(self.possDepots(N=[9]))
		n=len(self.possibleDepots)
		k=self.h

		self.posDisplays=self.calcPosDisp()
		#print(self.possibleDepots)
		
		'''
		plot adjacency matrix...

		plt.matshow( (self.dist_m*self.t_mile).applymap(lambda x: 1 if x<=2 else 0))
		cb = plt.colorbar()
		cb.ax.tick_params(labelsize=14)
		plt.show()		
		'''	

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
	
	def compMinVeh(self,α,n=1000):
		'''
		Estmates the minimum number of vehicles needed for each node

		input:
			α: confindence level (for percentile)
			n: Number of samples for Monte Carlo
		Output:
			None
			Saves a file with name 'num_veh.csv'
		'''
		minVeh=pd.DataFrame({'minVeh':[0 for i in self.Demands.index]},index=self.Demands.index)
		p=1/len(self.Demands.index)*self.Demands.applymap(lambda x: 1 if x>0 else 0).sum(axis=1)

		for i in self.G.nodes:
			dems=[sts.bernoulli.rvs(p=self.prob[i], loc=0, size=n, random_state=None)*sts.poisson.rvs(mu=self.demRate, loc=0, size=n, random_state=None) for j in self.G.nodes if self.Adj[i][j]==1]
			#print(self.Adj[i].sum())
			#print(len(dems))
			#print(dems)
			#plt.hist(dems)
			#plt.show()
			dems=sum(dems)
			minVeh.loc[i]=math.ceil(np.percentile(a=dems, q=α*100)/self.cap)
		
		minVeh.to_csv('num_veh.csv')

	def plot_G(self,ax=None):
		'''
		Plots G
		'''

		if ax!=None:
			plt.figure(num=None, figsize=(16*scal, 10*scal), dpi=80, facecolor='w', edgecolor='k')
			nx.draw_networkx(self.G,pos=self.pos,node_size=10,with_labels=False,ax=ax)
		else:	
			plt.figure(num=None, figsize=(16*scal, 10*scal), dpi=80, facecolor='w', edgecolor='k')
			plt.figure(num=None, figsize=(16*scal, 10*scal), dpi=80, facecolor='w', edgecolor='k')
			nx.draw_networkx(self.G,pos=self.pos,node_size=10,with_labels=False)
		#plt.show()

	def plot_day(self,day):		
		'''
		Plots demand for a specified day
		'''

		ax.clear()
		day=day+1
		dem=self.Demands[day]		
		nodes=[i for i in dem.index if dem[i]>0]		

		SG=self.G.subgraph(nodes)
		nx.draw_networkx(self.G,pos=self.pos,node_size=10,with_labels=False,ax=ax)		
		max_d=max(dem)
		n_size=[200*(dem[i]/max_d) for i in nodes]
		nx.draw_networkx(SG,pos=self.pos,with_labels=False,node_size=n_size,node_color='red',ax=ax)

	def plot_day_i(self,day):		
		'''
		Plots demand for a specified day
		'''
		day=day+1
		dem=self.Demands[day]		
		nodes=[i for i in dem.index if dem[i]>0]		

		SG=self.G.subgraph(nodes)
		nx.draw_networkx(self.G,pos=self.pos,node_size=10,with_labels=False)		
		max_d=max(dem)
		n_size=[200*(dem[i]/max_d) for i in nodes]
		nx.draw_networkx(SG,pos=self.pos,with_labels=False,node_size=n_size,node_color='red')		

		#plt.show()
	
	def plot_dem(self,dem,node_color='red'):
		'''
		Plots demand for a specified day 
		'''
		#ax.clear()
		#day=day+1
		#print(dem>0)
		#print(dem.index)
		#dem=self.Demands[day]
		#plt.figure(num=None, figsize=(16*scal, 10*scal), dpi=80, facecolor='w', edgecolor='k')

		nodes=[i for i in dem.index if dem.loc[i]>0]

		SG=self.G.subgraph(nodes)
		nx.draw_networkx(self.G,pos=self.pos,node_size=10,with_labels=False)
		max_d=15#max(dem)
		#print(nodes)
		n_size=[200*(dem.loc[i]/max_d) for i in nodes]
		#print(n_size)
		nx.draw_networkx(SG,pos=self.pos,with_labels=False,node_size=n_size,node_color=node_color)
		#plt.show()

	def plot_petals(self, petals, h,pet=True,pet_col='orange',ax=None):
		'''
		For each petal in petals plots the its tour with color pet_col. of pet==True, it does not plots the actual route in the graph from node to node, if pet == False, it plots the route between each node of each depot.
		For each i in h (set of depots)in plots a depot.
		
		Input:
			petals (list): list of petals to be draw
			h (list): set of depots to be draw
			pet (boolean): False if the routes frome node to node want to be drwe, true o.t.w
			pet_col= color of the petals.
		Output: 
			None
		'''
		#plt.figure(num=None, figsize=(16*scal, 10*scal), dpi=80, facecolor='w', edgecolor='k')
		for p in petals:			
			#d=sum(dem[i] for i in p)-2*dem[p[0]]
			#print('p: ',p,' ',d)
			if pet:
				path=list(zip(p[:-1],p[1:]))
				nx.draw_networkx_edges(self.G, self.pos,edgelist=path,edge_color=pet_col,width=2,ax=ax)
			else:			
				for i,j in zip(p[:-1],p[1:]):				
					path=nx.shortest_path(self.G,i,j,weight='c')
					path=list(zip(path[:-1],path[1:]))
					#print('este pat', path)
					nx.draw_networkx_edges(self.G, self.pos,edgelist=path,edge_color=pet_col,width=3,ax=ax)
			
		nx.draw_networkx_nodes(self.G, self.pos,nodelist=h,node_shape='s',node_size=200,node_color='green',ax=ax)
	
	def plotDepots(self,numVeh,ax=None):
		'''
		Plots the depots

		Input: 
			numVeh (dictionary) where the keys are the depots and the values are the number of vehicles assignet to that depot. Only the depots with positive number of vehicles are in the dictionary...
		Output:
			None. Plots in ax the depots.
		'''

		
		h=numVeh.keys()
		if ax==None:
			plt.figure(num=None, figsize=(16*scal, 10*scal), dpi=80, facecolor='w', edgecolor='k')
			self.plot_G()			
			self.Clusters=[list(self.Adj[i][self.Adj[i]==1].index) for i in h]
			self.plot_clusters(self.Clusters)

			nx.draw_networkx_nodes(self.G, self.pos,nodelist=h,node_shape='s',node_size=200,node_color='lime')
			try:
				lab={i:int(numVeh[i]) for i in numVeh.keys()}
			except:
				lab={i:'' for i in numVeh.keys()}

			nx.draw_networkx_labels(self.G,self.pos,labels=lab,font_size=16)
		else:
			ax.clear()
			self.plot_G(ax=ax)
			self.Clusters=[list(self.Adj[i][self.Adj[i]==1].index) for i in h]
			self.plot_clusters(self.Clusters,ax=ax)

			nx.draw_networkx_nodes(self.G, self.pos,nodelist=h,node_shape='s',node_size=200,node_color='lime',ax=ax)
			try:
				lab={i:int(numVeh[i]) for i in numVeh.keys()}
			except:
				lab={i:'' for i in numVeh.keys()}

			nx.draw_networkx_labels(self.G,self.pos,labels=lab,font_size=16,ax=ax)

	def animation(self,days):
		'''
		Aimates the demand of the time horizon.

		Input:
			days (int): inex of the days that want to be animated (5) animates the demand for the first 5 days.
			
		Output: 
			None
		
		'''
		global fig, ax
		fig = plt.figure(figsize=fsize)
		ax=fig.add_subplot()
		#plt.xlim(0,10)
		#plt.ylim(0,1)
		ani = matplotlib.animation.FuncAnimation(fig, self.plot_day, frames=days,interval=1000, repeat=True)
		plt.show()
		return ani

	def Benders_animation(self):
		'''
		Aimates the demand of the time horizon.

		Input:
			days (int): inex of the days that want to be animated (5) animates the demand for the first 5 days.
			
		Output: 
			None
		
		'''
		global fig, ax
		fig = plt.figure(figsize=fsize)
		ax=fig.add_subplot()
		#plt.xlim(0,10)
		#plt.ylim(0,1)
		f=lambda x: self.plotDepots(self.AnimNVeh[x],ax)
		ani = matplotlib.animation.FuncAnimation(fig, f, frames=len(self.AnimNVeh),interval=1000, repeat=True)
		#plt.show()
		return ani

	def plot_dem_timeseries(self, client):
		'''
		Plots the demand of one client as a time series.

		Input:
			client (DataFrame): dataframe of clients demand.
			
		Output: 
			None
		'''
		plt.figure(num=None, figsize=(16*scal, 10*scal), dpi=80, facecolor='w', edgecolor='k')
		plt.plot(self.Demands.columns, self.Demands.loc[client,:])

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
			#m=pickle.load('BMP.pkl')

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
				file_sp=open(f'sp{ss}.sp','wb')
				pickle.dump(s, file_sp)
				

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
				file_sp=open(f'sp{ss}.sp','wb')
				pickle.dump(s, file_sp)
				

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
					file_sp=open(f'sp{s.id}.sp','wb')
					pickle.dump(s, file_sp)
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
			if ((UB[-1]-LB[-1])<self.epsilon*UB[-1]):								
				break
			elif (time.time()-Start_time>=self.time_limit):				
				break
			#Export results
			self.export_results(depots=x_hat,veh_depot=y_hat)
			
		return UB,LB,x_hat,y_hat

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
				file_sp=open(f'sp{s.id}.sp','wb')
				pickle.dump(s, file_sp)
				

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
	
	def solve_sp(self,sp,plot=False):
		'''
		Solves the subproble inputed by parameter.
		
		Column generation with the pulse-split as slave.

		Input:
			sp (sub_problem): sub problem that wants to be solved				
		Output: 			
			FO: Returns the objective function of the problem with integrality!
			Lambdas: Returns the dual variables for the benders cuts (relaxed problem.)
		
		'''

		FO=[]

		clients=sp.D.iloc[:]
		π={i:0 for i in clients.index}

		'''
		Creates a first set of feasible paths using tge split algprithm over a neighborhood of each depot. 
		Sub problem Warm start
		If a set of routes already exists departing from some depot, (i.e d in sp.Ri.keys()), then this procedure is not done.
		'''
		time_1=time.time()
		
		#print_log('\t','Clients:\n',sp.R)
		for d in sp.H:						
			close=[i for i in sp.Ri.keys() if sp.t_mile*sp.dm[d][i]<0.5 and i!=d]
			if len(close)!=0:
				sp.petal_recicler(d,close)
			else:				
				if d not in sp.Ri.keys():
					clients_h=list(self.Adj[d][self.Adj[d]==1].index)				
					clients_h=list(set(clients.index) & set(clients_h))			
					clients_h=clients.loc[clients_h]
					sp.petal_generator(d, clients_h, π)
			#petals=sp.R[-1]			
			#print(petals)
		
		'''
		alc=sum(sp.R,[])	
		for i in clients.index:
			if i not in alc:
				for d in sp.H:
					route=[d,i,d]
					cost=sp.c_mile*sum(sp.dm[k][l] for k,l in zip(route[:-1],route[1:]))
					sp.R.append(route)
					sp.Route_cost.append(cost)
					#print_log('por ',self.Adj[d][i])
		'''
		print_log('\t',f'En la fase constructiva del sub problama me tardo {time.time()-time_1}')
		

		'''
		improves the solution witha column generation
		'''

		print_log('\t',f'Empece GC sub_problem {sp.id}')
		n_it=0
		while True:
			n_it+=1			
			m=sp.master_problem(relaxed=True)
			time_genPet=time.time()
			m.optimize()
			sp.z_hat={k:kk. x for k,kk in m._z.items()}	

			#print_log('\t',f'Me demoro {time.time()-time_genPet} segundos resolviendo la relajación')
			FO.append(m.objVal)
			#print('\t',f'el status del problema es {m.getAttr("Status")}')
			π={i:m._set_covering[i].Pi for i in m._set_covering.keys()}			
			λ={i:m._Num_veh[i].Pi for i in m._Num_veh.keys()}
			print_log('\t πs')
			print_log('\t',π)
			print_log('\t λs')
			print_log('\t',λ)

			term=True
			for d in sp.H:
				#cord=self.V.loc[d][['lat','long']]	
				#clients_h=list(self.V.loc[distance(self.V.loc[:,'lat'],self.V.loc[:,'long'],cord.loc['lat'],cord.loc['long'])<(self.route_time_limit/5) *40].index)			
				clients_h=list(self.Adj[d][self.Adj[d]==1].index)
				clients_h=list(set(clients.index) & set(clients_h))
				clients_h=clients.loc[clients_h]

				#print_log(f'\t\tEmpcece con petal gen del depot {d} en la it {n_it}')
				time_genPet=time.time()				
				r_costs=sp.petal_generator(d, clients_h,π)
				#print_log('\t\t',f'Me demoro {time.time()-time_genPet} segundos')

				if sum(r_costs)-λ[d]<0:
					#print_log('\t\t',r_costs)
					#print_log('\t\tLas λ dan',λ[d])
					term=False
					break

			#print('\t',f'Voy {n_it} iteraciones del subproblema')
			#print('\t',FO[-1])

			if n_it==10 or term:
				print_log('\t','El numero de iteraciones fue: ',n_it)
				break
		print_log('\tTermine con la GC..')
		
		
		m=sp.master_problem(relaxed=True)
		m.optimize()

		#Uses this solution to start next time
		sp.z_hat={k:kk.x for k,kk in m._z.items()}	

		#Recovers the duals of the relaxed
		λ={i:m._Num_veh[i].Pi for i in m._Num_veh.keys()}
		π={i:m._set_covering[i].Pi for i in m._set_covering.keys()}				

		
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
		
		#if plot:
		if True:
			
			#print_log('\t\t',sp.id,'\n\t\t',sp.R,'\n\t\t',sp.Route_cost,'\n\t\t',sp.OptRi)
			#m=sp.master_problem(relaxed=False)
			#m.optimize()
			
			sp.export_results({i:m._z[i].x for i in m._z.keys()})
			pet=[i for i,p in enumerate(m._z) if m._z[i].x>.5]
			routes=[sp.R[i] for i in pet]
			sp.OptRi=routes
			#self.plot_dem(sp.D)
			#self.plot_petals(routes,sp.H,pet=True)
			#plt.show()
			os.chdir('../Plots/Subproblems')
			if sp.id==0:delete_all_files(os.getcwd())
			self.plot_dem(sp.D)
			self.plot_petals(routes,sp.H,pet=False)
			plt.savefig(f'sp_{sp.id}.png')
			#plt.show()
			plt.clf()			
			os.chdir('..')
			os.chdir('../Data')
			
			foi=sum(sp.Route_cost[r]*m._z[r].x for r in range(len(sp.R)))
			return m.objVal, λ, π
		else:
			return m.objVal, λ, π

	def solveSpJava(self,sp,nRoutes=10,gap=0.001,tw=False):
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

			term=True			
			for d in HH:

				#cord=self.V.loc[d][['lat','long']]	
				#clients_h=list(self.V.loc[distance(self.V.loc[:,'lat'],self.V.loc[:,'long'],cord.loc['lat'],cord.loc['long'])<(self.route_time_limit/5) *40].index)			
				clients_h=list(self.Adj[d][self.Adj[d]==1].index)
				clients_h=list(set(clients.index) & set(clients_h))
				clients_h=clients.loc[clients_h]

				#print(f'\t\tEmpcece con petal gen del depot {d} en la it {n_it}')
				time_genPet=time.time()
				
				r_costs=sp.runJavaPulse(d, clients_h,π,λ[d],nRoutes=nRoutes,tw=tw)
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

	def solve_spAl(self,sp,plot=False,Alternative=True):
		'''
		Solves the subproble inputed by parameter.
		
		Column generation with the pulse-split as slave.

		Input:
			sp (sub_problem): sub problem that wants to be solved				
		Output: 			
			FO: Returns the objective function of the problem with integrality!
			Lambdas: Returns the dual variables for the benders cuts (relaxed problem.)
		
		'''

		FO=[]

		clients=sp.D.iloc[:]
		π={i:0 for i in clients.index}

		'''
		Creates a first set of feasible paths using tge split algprithm over a neighborhood of each depot. 
		Sub problem Warm start
		If a set of routes already exists departing from some depot, (i.e d in sp.Ri.keys()), then this procedure is not done.
		'''
		time_1=time.time()
		
		#print_log('\t','Clients:\n',sp.R)
		for d in sp.H:						
			close=[i for i in sp.Ri.keys() if sp.t_mile*sp.dm[d][i]<0.5 and i!=d]
			if len(close)!=0:
				sp.petal_recicler(d,close)
			else:				
				if d not in sp.Ri.keys():
					clients_h=list(self.Adj[d][self.Adj[d]==1].index)				
					clients_h=list(set(clients.index) & set(clients_h))			
					clients_h=clients.loc[clients_h]
					sp.petal_generator(d, clients_h, π)
			#petals=sp.R[-1]			
			#print(petals)
		
		alc=sum(sp.R,[])	
		for i in clients.index:
			if i not in alc:
				for d in sp.H:
					route=[d,i,d]
					cost=sp.c_mile*sum(sp.dm[k][l] for k,l in zip(route[:-1],route[1:]))
					sp.R.append(route)
					sp.Route_cost.append(cost)
					#print_log('por ',self.Adj[d][i])
		print_log('\t',f'En la fase constructiva del sub problama me tardo {time.time()-time_1}')
		

		'''
		improves the solution witha column generation
		'''

		n_it=0
		while True:
			n_it+=1
			if Alternative:
				print_log('\tEstoy alternativo...')
				m=sp.master_problemA(relaxed=True)
			else:
				m=sp.master_problem(relaxed=True)
			time_genPet=time.time()
			m.optimize()
			sp.z_hat={k:kk. x for k,kk in m._z.items()}	

			#print_log('\t',f'Me demoro {time.time()-time_genPet} segundos resolviendo la relajación')
			FO.append(m.objVal)
			#print('\t',f'el status del problema es {m.getAttr("Status")}')
			π={i:m._set_covering[i].Pi for i in m._set_covering.keys()}			
			λ={i:m._Num_veh[i].Pi for i in m._Num_veh.keys()}


			term=True
			for d in sp.H:
				#cord=self.V.loc[d][['lat','long']]	
				#clients_h=list(self.V.loc[distance(self.V.loc[:,'lat'],self.V.loc[:,'long'],cord.loc['lat'],cord.loc['long'])<(self.route_time_limit/5) *40].index)			
				clients_h=list(self.Adj[d][self.Adj[d]==1].index)
				clients_h=list(set(clients.index) & set(clients_h))
				clients_h=clients.loc[clients_h]

				#print_log(f'Empcece con petal gen del depot {d} en la it {n_it}')
				time_genPet=time.time()				
				r_costs=sp.petal_generator(d, clients_h,π)
				#print_log('\t',f'Me demoro {time.time()-time_genPet} segundos')

				if sum(r_costs)-λ[d]<0:
					#print_log('\t',r_costs)
					#print_log('\tLas λ dan',λ[d])
					term=False
					break

			#print('\t',f'Voy {n_it} iteraciones del subproblema')
			#print('\t',FO[-1])

			if n_it==100 or term:
				print_log('\t','El numero de iteraciones fue: ',n_it)
				break
		print_log('\tTermine con la GC..')
		
		if Alternative:
				m=sp.master_problemA(relaxed=True)
		else:
				m=sp.master_problem(relaxed=True)
		m.optimize()
		#Uses this solution to start next time
		sp.z_hat={k:kk.x for k,kk in m._z.items()}	

		#Recovers the duals of the relaxed
		λ={i:m._Num_veh[i].Pi for i in m._Num_veh.keys()}
		π={i:m._set_covering[i].Pi for i in m._set_covering.keys()}				

		print_log('\tEmpece a resolver con integralidad')
		if Alternative:
				m=sp.master_problemA(relaxed=False)
		else:
				m=sp.master_problem(relaxed=False)		
		time_genPet=time.time()		
		m.optimize()
		print_log('\t',f'Me demoro {time.time()-time_genPet} segundos resolviendo con integralidad')

		#Guarda las rutas seleccionadas para cada depot en sp.OptRi
		for d in sp.H:
			routes=[i for i,j in m._z.items() if j.x==1 and sp.R[i][0]==d]			
			if d not in sp.OptRi.keys():
				sp.OptRi[d]=routes
			else:
				sp.OptRi[d]+=routes		
		
		#if plot:
		if True:
			
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
			plt.savefig(f'sp_{sp.id}.png')
			#plt.show()
			plt.clf()			
			os.chdir('..')
			os.chdir('../Data')
			
			foi=sum(sp.Route_cost[r]*m._z[r].x for r in range(len(sp.R)))
			return m,foi, λ, π
		else:
			if Alternative:
				foi=sum(sp.Route_cost[r]*m._z[r].x for r in range(len(sp.R)))
				return m,foi, λ, π
			else:	
				return m.objVal, λ, π
	
	def buil_aux_slaveG(self,clients,π,H):			
		nod=list(π.keys())+H
		
		'''
		Builds the reduce cost matrix: r_ij = d_ij -π_i
		'''
				
		rc_mat=self.dist_m.loc[nod,nod]
		
		

		for i in nod:
			try:
				rc_mat.loc[i,:]=rc_mat.loc[i,:]-π[i]
			except:
				rc_mat.loc[i,:]=rc_mat.loc[i,:]

		nod=list(rc_mat.index)	

		#print([d in nod for d in H])
		rc_mat = rc_mat.to_numpy()

		time_mat=(self.t_mile*self.dist_m.loc[nod,nod]).to_numpy()
		cap_mat=np.array([[clients[j] if j in clients.index else 0 for j in nod] for i in nod])
		Repl_matrix=np.array([[0 for j in nod] for i in nod])

		dt = [('Cost', float), ('Time', float),('Cap', int),('Replen', int)]
		A=np.dstack((rc_mat,time_mat,cap_mat,Repl_matrix))
		A=np.array([[tuple(A[i][j]) for j in range(len(nod))] for i in range(len(nod))],dtype=dt)

		slave_Graph=nx.DiGraph(nx.from_numpy_matrix(A))

		mapping={i:l for i,l in enumerate(nod)}
		nx.relabel_nodes(slave_Graph, mapping,copy=False)

		return slave_Graph

	def warm_start(self):

		'''
		Creates a first feasible solution for the first stage, estimates the cost of this solution solving TSP for each cluster.		

		Input: None
		Output: y_hat, x_hat, η_hat, Cost
		'''


		nodes=list(self.V.index)		
		points=[[self.V.loc[i]['lat'],self.V.loc[i]['long']] for i in nodes]

		n=self.h
		# create kmeans object
		kmeans = KMeans(n_clusters=n,random_state=5)
		# fit kmeans object to data
		kmeans.fit(points)

		# print location of clusters learned by kmeans object
		centers=kmeans.cluster_centers_

		#Save clusters
		y_km = kmeans.fit_predict(points)		

		#Nodes in each class
		self.Clusters=[[k for j,k in enumerate(nodes) if y_km[j]==i ] for i in range(n)]

		H=[self.centroid(self.Clusters[i],centers[i]) for i in range(n)]
		

		x_hat={i:1 if i in H else 0 for i in self.G.nodes }
		

		η_hat={}

		#For each scenario,  and for each cluster, solves a TSP to estimate the cost of that scenario,
		n_veh={i:0 for i in H}
		for i,s in enumerate(self.SPS):
			cost=0
			clients=s.D.iloc[:]
			for j,h in enumerate(H):
				#cord=self.V.loc[d][['lat','long']]					
				#clients_h=list(self.V.loc[distance(self.V.loc[:,'lat'],self.V.loc[:,'long'],cord.loc['lat'],cord.loc['long'])<(self.route_time_limit/5) *40].index)			
				clients_h=self.Clusters[j]
				clients_h=list(set(clients.index) & set(clients_h))			
				clients_h=clients.loc[clients_h]
				print(clients_h)
				dist, tour=self.solve_tsp(list(clients_h.index))
				cost+=dist*self.c_mile				
				n_veh[h]=max(n_veh[h], math.ceil(sum(clients_h.values)/self.cap))
			η_hat[i]=cost
		y_hat={i:n_veh[i] if i in H else 0 for i in self.G.nodes }
		
		first_stage=sum(self.f['cost'].loc[i]*x_hat[i]+ self.c*y_hat[i] for i in self.G.nodes)
		E_second_stage=sum(η_hat[s] for s,i in enumerate(self.SPS))
		Cost=first_stage

		return y_hat, x_hat, η_hat, Cost
		'''
		colors=['blue','green','red','cyan','magenta','yellow','black','lime','white']
		nx.draw(p.G,p.pos,node_size=1)
		for i in range(n):
			print(i)
			print(colors[i])
			print(len(self.Clusters[i]))
			nx.draw_networkx_nodes(p.G, p.pos,nodelist=self.Clusters[i],node_size=50,node_color=colors[i])
			
		#plt.show()
		'''

	def clusters_louvain(self): 
		'''
		Uses the algorithm for clustering communities of louvain
		and returns a list of lists of clusters 
		
		Input:
			None
		Output:
			All_clusters (list): List of list of clusters for different levels in dendogram.
		'''
		nodes = list(self.pos.keys())
		dendo = community_louvain.generate_dendrogram(self.G)

		All_clusters=[]
		for level in range(len(dendo)-1):
			# save new clusters for chart
			y_km = community_louvain.partition_at_level(dendo, level)

			#Nodes in each class
			clusters=[[k for j,k in enumerate(nodes) if y_km[k]==i ] for i in range(len(set(y_km.values())))]
			All_clusters.append(clusters)

		return All_clusters

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

	def analysis_cluster(self,cluster,color,index,j):
		'''
		Bunch of analysis for a specific cluster of  nodes:
		Input:
			cluster (list): List of the nodes belonging to that cluster
			color: Color to plot that cluster
			index(int): Index of the cluster
			j (int): Index of the level of the clusters (how many clusters are in that level of clusterings)
		Output:
			TSP_cost*self.t_mile: The time of the TSP for this cluster
			TSP_cost*self.c_mile: The cost of the TSP for this cluster
		'''

		#os.chdir('../Cluster_anal')		
		####### Plots cluster.
		self.plot_cluster(cluster, color)		
		plt.savefig('{}_cluster_{}.png'.format(j,index))
		#plt.show()
		plt.clf()

		dem=self.Demands.loc[cluster]
		'''
		
		####### Plots zero histogram for that cluster.
		zeros = dem[dem == 0].count(axis=1)		
		plt.hist(zeros)
		plt.xlabel('Number of non-order days per client')
		plt.ylabel('Frecuency')
		plt.savefig('{}_zero_hist_{}.png'.format(j,index))
		#plt.show()
		plt.clf()
		'''

		'''
		####### Plots non zero histogram for that cluster.
		zeros = dem[dem > 0].count(axis=1)		
		plt.hist(zeros)
		plt.xlabel('Number of order days per client')
		plt.ylabel('Frecuency')
		plt.savefig('{}_non_zero_hist_{}.png'.format(j,index))
		#plt.show()
		plt.clf()
		'''

		####### Plots histograms of the number of units in an order.

		dat=[]
		for i in dem.index:
			vasl=dem.loc[i][dem.loc[i]>0]
			dat+=list(vasl)		
		
		mean=np.mean(dat)

		popt, hist, bins, center=fit_poisson(dat,p0=mean)

		N=sum(hist)
		plt.bar(bins[:-1],[i/N for i in hist], label='Data')
		plt.plot(center, poisson(center, popt), 'r--', label='Poisson fit')

		plt.legend(loc = 'best')
		plt.xlabel('Order size (Number of units in an order)')
		plt.ylabel('Density')

		plt.text(x=1, y=0.12, s='Mean: {}'.format(round(mean, 2)))
		plt.text(x=1, y=0.11, s=r'Rate ($\lambda$): {}'.format(round(popt[0], 2)))

		plt.savefig('{}_order_size_{}.png'.format(j,index))
		#plt.show()
		plt.clf()

		############ Solves TSP for cluster:
		#
		#nx.draw_networkx(SG,self.pos,with_labels=False,node_size=10)
		#plt.show()

		#for i,j in SG.edges():
		#	print (SG[i][j])
		
		try: 
			TSP_cost,tour=self.solve_tsp(cluster)
			SG=self.G.subgraph(cluster)
			self.plot_petals(petals=[tour +[tour[0]]],h=[],pet=False,pet_col=color)
			nx.draw_networkx_nodes(SG, self.pos,node_size=10)

			plt.title('Total cost: {}$ \nTotal time: {} h'.format(round(TSP_cost*self.c_mile,2),round(TSP_cost*self.t_mile,2)),loc='left')		
			plt.savefig('{}_tsp_{}.png'.format(j,index))
			#plt.show()
			plt.clf()
			
			time=[]
			cap=[]
			for i in self.Demands.columns:
				
				day=i
				dem=self.Demands[day]		
				nodes=[i for i in dem.index if dem[i]>0]


				cost,tour=self.solve_TSP_day(demand=dem, day=i, cluster=cluster)
				time.append(cost*self.t_mile)
				cap.append(sum(dem[cluster]))

				#if i>20:
				#	break
			

			############ Vehicles needed per day
			veh=list(map(lambda x: math.ceil(x/self.cap),cap))

			plt.hist(veh)
			plt.xlabel('Vehicles needed in a day')
			plt.ylabel('Frequency')
			
			plt.savefig('{}_Vehicles_per_day_{}.png'.format(j,index))
			#plt.show()
			plt.clf()		

			############ Distribution of the time in cluster i
			veh=list(map(lambda x: math.ceil(x/self.cap),cap))

			plt.hist(time)
			plt.xlabel('Time of the TSP solved in a day')
			plt.ylabel('Frequency')
			
			plt.savefig('{}_Time_TSP_cluster_{}.png'.format(j,index))
			#plt.show()
			plt.clf()		
			
			############ Clients per day with positive demand.
			
			dem=self.Demands.loc[cluster]
			C_per_day=dem[dem > 0].count()

			plt.xlabel('Clients per day in cluster')
			plt.ylabel('Density')
			plt.hist(C_per_day)
			
			plt.savefig('{}_Clients_per_day_{}.png'.format(j,index))
			#plt.show()
			plt.clf()
			#os.chdir('../Data')
			
			return TSP_cost*self.t_mile,TSP_cost*self.c_mile
		except: 
			print("Warning: Concorde cannot be used in Windows operating system")

	def gen_report_clusters(self, doc ,clusters):
		'''
		Generates the analysys for eack cluster un clusters and writes it in doc
		
		Input:
			doc (pyLatex doc): Document in which the analyses are written
			clusters: List of clusters to analyse
		Output:
			None
		'''
		#os.chdir('../Cluster_anal')
		
		
		############################
		#Color map

		n=len(clusters)
		viridis = cm.get_cmap('rainbow', int(n))
		colors = viridis(np.linspace(0, 1, int(n)))
		############################
		#Plots clusters
		self.plot_clusters(clusters)
		plt.savefig(f'{n}_clusters.png')
		plt.clf()

		############################
		#Creates plots
		
		clusters=sorted(clusters, key=len)
		clusters=clusters[::-1]
		
		
		nn=len(clusters)
		if nn>=20:
			nn=20		

		lcost,ltime=[],[]
		for i,c in enumerate(clusters):
			c,t=self.analysis_cluster(cluster=c, color=colors[i], index=i,j=n)
			lcost.append(c)
			ltime.append(t)
		

		######################################
		#Summary plots for the TSP time and cost.
		plt.hist(ltime)
		plt.xlabel('Time of the TSP in the clusters formed')
		plt.ylabel('Frecuency')
		plt.savefig('{}_TSP_T_hist.png'.format(n))
		#plt.show()
		plt.clf()

		plt.hist(lcost)
		plt.xlabel('Cost of the TSP in the clusters formed')
		plt.ylabel('Frecuency')
		plt.savefig('{}_TSP_C_hist.png'.format(n))
		#plt.show()
		plt.clf()
		
		lens=list(map(len,clusters))

		plt.scatter(x=lens,y=ltime)
		plt.xlabel('Size of the cluster')
		plt.ylabel('Time of the TSP')
		plt.savefig('{}_t_vs_size.png'.format(n))
		plt.clf()

		plt.scatter(x=lens,y=lcost)
		plt.xlabel('Size of the cluster')
		plt.ylabel('Cost of the TSP')
		plt.savefig('{}_c_vs_size.png'.format(n))
		plt.clf()

		with doc.create(Section(f'Clusters ({n})')):
			doc.append(NoEscape(f'The {n} clusters formed are presented in figure' r'\ref{'+ f'fig:closters_{n}'+r'}'))

			with doc.create(Figure(position='h!')) as pic:
				pic.add_image(f'{n}_clusters.png', width='400px')
				pic.add_caption(f'Closters ({n})')
				doc.append(NoEscape(r'\label{fig:closters_'+f'{n}'+r'}'))


		'''
		with doc.create(Subsection('Clusters ({})'.format(n))):
			doc.append('One by one:')
			doc.append(NoEscape(subplots(n,nn, 'cluster','Single clusters')))
		'''

		'''
		with doc.create(Subsection('Order size per cluster ({})'.format(n))):
			doc.append(NoEscape(r'Figure \ref{fig:order_size'+f'_{n}'+'} presents the distribution of the number of units ordered by client (if an order is made) for the' +f'{nn}' +r' bigges clusters.'))
			doc.append(NoEscape(subplots(n,nn, 'order_size','Order size by cluster')))
			doc.append('\n As we can see, the parameter of the poisson distribution fitted dosent seems to change for each cluster, what makes us think that this number of units ordered for each client is a global pattern.')
		'''


		with doc.create(Subsection('TSP for each cluster ({})'.format(n))):
			doc.append(NoEscape(r'For each cluster a TSP is solved to optimality (using concorde).'+r'\newline'))#' The cost (USD) and time (Hours) for the' +f'{nn}' +r' bigges clusters is presented in each plot of figure \ref{fig:tsp'+f'_{n}'+'}.'+r'\newline'))			
			#doc.append(NoEscape(subplots(n,nn, 'tsp','TSP solved by cluster')))
			
			doc.append(NoEscape(r'\begin{figure}[h!]'))
			doc.append(NoEscape(r'\subfloat['+'TSP Time {}'.format(n+1)+r']{\includegraphics[width=80mm]{'+'{}_{}.png'.format(n,'TSP_T_hist')+r'}}'+''))
			doc.append(NoEscape(r'\subfloat['+'TSP time v.s. cluster size {}'.format(n+1)+r']{\includegraphics[width=80mm]{'+'{}_{}.png'.format(n,'t_vs_size')+r'}}'+''))
			doc.append(NoEscape(r'\caption{Times of the TSP for the clusters'+f' with {n} clusters'+r'}'+''))
			doc.append(NoEscape(r'\label{fig:'+'t_TSP'+f'_{n}'+r'}'+''))
			doc.append(NoEscape(r'\end{figure}'))
			doc.append(NoEscape(r'Figure \ref{'+'fig:'+'t_TSP'+f'_{n}'+r'} shows the distribution of the times of the TSP solved for each cluster, and a scatter plot of the size of the cluster vs the time of the solved TSP.'))			
			doc.append(NoEscape(r'For each day, a TSP is solved to optimality with the active clients in the cluster, this histogram is presented in figure \ref{fig:Time_TSP_cluster'+f'_{n}'+r'} for each cluster. Additionally, figure\ref{fig:Vehicles_per_day'+f'_{n}'+'} shows the number of vehicles needed each day in each cluster.'+r'\newline'))			

			doc.append(NoEscape(subplots(n,nn, 'Time_TSP_cluster','TSP solved by cluster each day')))
			doc.append(NoEscape(subplots(n,nn, 'Vehicles_per_day','Number of petals needed in a day')))



		with doc.create(Subsection('Number of active clients per day for each cluster ({})'.format(n))):
			doc.append(NoEscape(r'For each cluster we wanted to know how many clients with positive demand lie in the cluster each day. Figure \ref{fig:Clients_per_day'+f'_{n}'+'} presents a histogram of these values for the' +f'{nn}' +r' bigges clusters.'))
			doc.append(NoEscape(subplots(n,nn, 'Clients_per_day','Number of active clientes per day for cluster')))
			doc.append('\nAs we can see is not very uniform, we could use the mean?')

		'''
		with doc.create(Subsection('Number of times a client orders in a year ({})'.format(n))):
			doc.append(NoEscape(r'Figure \ref{fig:non_zero_hist'+f'_{n}'+'} presents the number of times a client orders in a year. As we can see the mayority of the clients do not order very often, since the mass is acumulated near to zero.'))
			doc.append(NoEscape(subplots(n,nn, 'non_zero_hist','Number of times a client orders in a year')))
		'''
		#os.chdir('../Data')
	
	def gen_report(self,All_clusters):
		'''
		Generates a report for many level of clusters. Using pylatex generates a .latex and .pdf with the report.

		Input:
			All_clusters: List of lists of clusters (various levels)			
		Output:
			None
		'''
		os.chdir('../Cluster_anal')		
		delete_all_files(os.getcwd())
		print(os.getcwd())
		#Generates the path for the given image.
		#im_path=lambda image_filename: os.path.join(os.path.dirname(__file__), image_filename)

		geometry_options = {"tmargin": "1cm", "lmargin": "4cm"}
		doc = Document(geometry_options=geometry_options)

		############################
		#plots total cluster.
		packages=['graphicx','subfig','amsmath']

		for pack in packages:
			doc.preamble.append(Command('usepackage',pack))

		
		#############################
		#Writes titel

		doc.preamble.append(Command('title', 'Analysis by clusters'))
		doc.preamble.append(Command('author', NoEscape(r'M$\alpha$D')))
		doc.preamble.append(Command('date', NoEscape(r'\today')))

		doc.append(NoEscape(r'\maketitle'))
		
		for clusters in All_clusters:
			self.gen_report_clusters(doc, clusters)
			doc.append(NoEscape(r'\clearpage'))
		
		#Generates the pfd		
		doc.generate_pdf('Analysis by cluster', clean_tex=False)
		delete_all_files(os.getcwd(),['pdf','tex'])
		os.chdir('../Data')

	def solve_tsp(self,cluster):
		'''
		Solves a TSP for the given cluster.
		Input:
			List of nodes to solve TSP
		Output:
			val: Optimal value of the TSP
			tour: sequence of the optimal tour
		'''

		di_mat=self.dist_m.loc[cluster,cluster]
		index=di_mat.index
		di_mat=di_mat.values		
		fname='TSP.tsp'
		fp=open(fname,'w')
		write_tsp_file_dmat(fp, distance_matrix=di_mat, name='cluster_i')
		fp.close()

		try: 

			solver = TSPSolver.from_tspfile(fname)		
			solution = solver.solve(verbose = False)
			print(help(solver.solve))
			tour=[index[i] for i in solution.tour]

			return solution.optimal_value, tour
		except: 
			print("Warning: Concorde cannot be used in Windows operating system")
			pass 
		
	def plot_cluster(self,cluster,color,ax=None):
		'''
		plots a cluster given by parameter with the color
		Input:
			cluster: List of nodes
			color: color
			ax: optional ax to plot in.
		Output:
			None
		'''
		sG=self.G.subgraph(cluster)
		n_size=self.Demands.sum(axis=1)
		n_size=n_size*(500/max(n_size))
		if ax==None:			
			nx.draw_networkx(sG,pos=self.pos,with_labels=False,node_list=cluster,node_size=n_size,node_color=color,alpha=0.5)
		else:
			nx.draw_networkx(sG,pos=self.pos,with_labels=False,node_list=cluster,node_size=n_size,node_color=color,ax=ax,alpha=0.5)
	
	def plot_clusters(self, clusters,ax=None):
		'''
		plots many clustrs
		Input:
			clusters: List of cluster 
			ax: Optional ax to plot in
		Output:
			None
		'''
		n=len(clusters)
		viridis = cm.get_cmap('rainbow', int(n))
		colors = viridis(np.linspace(0, 1, int(n)))
		ctoc=lambda x: matplotlib.colors.to_hex(x, keep_alpha=True)
		colors=list(map(ctoc,colors))


		if len(clusters)==1:
			colors=['green']
		for i,c in enumerate(clusters):
			self.plot_cluster(c,colors[i],ax=ax)

	def solve_TSP_day(self, demand,day, cluster):
		'''
		Solves the TSP for a given demand, or a given day
		Input:
			demand: demand to solve the TSP in
			day: day in which the demand is going to be solved
			cluster: the cluster.
		Output:
			val: value of the TSP
			tour: Optimal tour

		'''
		if len(demand)==0:
			day=day+1
			dem=self.Demands[day]		
			nodes=[i for i in dem.index if dem[i]>0 and i in cluster]	
		else:
			nodes=[i for i in demand.index if demand[i]>0 and i in cluster]

		di_mat=self.dist_m.loc[nodes,nodes]
		index=di_mat.index

		di_mat=di_mat.values		
		fname='TSP.tsp'
		fp=open(fname,'w')
		write_tsp_file_dmat(fp, distance_matrix=di_mat, name='cluster_i')
		fp.close()
		
		try:
			solver = TSPSolver.from_tspfile(fname)		
			solution = solver.solve()		
			tour=[index[i] for i in solution.tour]
		
			return solution.optimal_value, tour
		except:
			return 0,[]

	def readResultsFS(self):
		'''
		Reads the results from a .txt

		Input:
			file: Name of the file to read
		Output:
			n_veh: dictionary with the opened depots as keys and the number of vehicles as values
		'''
		os.chdir('..')
		os.chdir('Results')

		f=open('First_stage.txt')
		n_veh={}
		i=0
		for line in f:
			if i>0:
				j,n=list(map(lambda x: int(float(x)),line.replace('\n','').split('\t')))
				n_veh[j]=n
			i+=1

		#self.plotDepots(n_veh)
		#plt.show()

		return n_veh
		os.chdir('..')
		os.chdir('Data')

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

def subplots(n,nn,lab,caption):
	'''
	Latex code to plot subplots.
	Input:
		n, nn: parameters for indexing the subplots
		lab: Label of the plot
		caption: Camption of the plot
	'''
	t=''

	for i in range(nn):
		#print(t)
		if i%4==0:
			if i!=0:
				t+=r'\end{figure}'
				t+='\n'
			t+=r'\begin{figure}[h!]'
			t+='\n'
		t+=r'\subfloat['+'Cluster {}'.format(i+1)+r']{\includegraphics[width=80mm]{'+'{}_{}_{}.png'.format(n,lab,i)+r'}}'+'\n'
		if (i-1)%2==0:
			t+=r'\\'+'\n'
		if i==nn-1:
			t+=r'\caption{'+caption +f' with {n} clusters'+r'}'+'\n'
			t+=r'\label{fig:'+lab+f'_{n}'+r'}'+'\n'
			


	if t[-8:]!=r'{figure}':
		t+=r'\end{figure}'
	return t

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

def poisson(t, rate): return (rate**(t)/factorial(t))*np.exp(-rate)

def fit_poisson(data,p0):
	'''
	Fits Poisson distr to data.
	'''
	hist, bins = np.histogram(data, bins=max(data))
	center = (bins[:-1]+bins[1:])/2
	mean=np.mean(data)
	popt, pcov = curve_fit(poisson, center, hist,p0=p0)		

	return popt, hist, bins, center

def epsilon_optimal(M,where):
	'''
	Calback for solving M to epsilon optimality or to many nodes explored.
	'''
	if where == GRB.Callback.MIP:
		nodecnt = M.cbGet(GRB.Callback.MIP_NODCNT)
		objbst = M.cbGet(GRB.Callback.MIP_OBJBST)
		objbnd = M.cbGet(GRB.Callback.MIP_OBJBND)
		solcnt = M.cbGet(GRB.Callback.MIP_SOLCNT)
		#print(abs(objbst - objbnd)/abs(objbst))
		if abs(objbst - objbnd)<= M._epsilon*abs(objbst):
			M.terminate()
			#print('Stop early - gap 0.1% achived')
		if nodecnt >= 10000 and solcnt:
			#print('Stop early - 10000 nodes explored')
			M.terminate()
	elif  where== GRB.Callback.MIPNODE:
		nodecnt = M.cbGet(GRB.Callback.MIP_NODCNT)
		objbst = M.cbGet(GRB.Callback.MIP_OBJBST)
		objbnd = M.cbGet(GRB.Callback.MIP_OBJBND)
		solcnt = M.cbGet(GRB.Callback.MIP_SOLCNT)

def write_tsp_file_dmat(fp,distance_matrix, name):
	""" Write data to a TSPLIB file.
	"""

	fp.write("NAME: {}\n".format(name))
	fp.write("TYPE: TSP\n")
	fp.write("DIMENSION: {}\n".format(len(distance_matrix)))
	fp.write("EDGE_WEIGHT_TYPE: EXPLICIT\n")	
	fp.write('EDGE_WEIGHT_FORMAT: FULL_MATRIX\n')
	fp.write("EDGE_WEIGHT_SECTION\n")

	for i in range(len(distance_matrix)):
		t=''
		for j in range(len(distance_matrix)):
			t+=str(int(round(distance_matrix[i][j],0)))+' '
		fp.write(t[:-1]+'\n')	
	fp.write("EOF\n")

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

		#Routes per node
		self.Ri={}

		#Routes selected for node i
		self.OptRi={}

		#Set of routes
		self.R=[]

		#Route costs
		self.Route_cost=[]

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
	
	def tsp(self,clients,π):
		'''
		Input:
			depot: Depot from wher the routes will be generated
			clients: Clients that need to be covered
			π: dual variables for each node

		Output:

			r_costs: The reduced cost of the petals generated.
		'''		
		
		#π=DataFrame(π.values(),index=π.keys())
		#print(π)
		
		di_mat=self.dm.loc[clients,clients]
		#print(di_mat)
		for i in clients[1:]:			
			di_mat[i]-=π[i]/self.c_mile
		#print(di_mat)
		tsp=TSP()
		depot=clients[0]
		tour=tsp.nearestNeighbor(di_mat,depot)

		return tour
		
	def tspA(self,clients,π):
		'''
		Input:
			depot: Depot from wher the routes will be generated
			clients: Clients that need to be covered
			π: dual variables for each node

		Output:

			r_costs: The reduced cost of the petals generated.
		'''		

		di_mat=self.dm.loc[clients,clients]
		index=di_mat.index
		di_mat=di_mat.values		
		fname='TSP.tsp'
		try:
			fp=open(fname,'w')
			write_tsp_file_dmat(fp, distance_matrix=di_mat, name='clients_i')
			fp.close()
		except:
			fname='TSP1.tsp'
			fp=open(fname,'w')
			write_tsp_file_dmat(fp, distance_matrix=di_mat, name='clients_i')
			fp.close()

		try: 
			solver = TSPSolver.from_tspfile(fname)
			solution = solver.solve(verbose=True)
			tour=[index[i] for i in solution.tour]
			return tour
		except: 
			print("Warning: Concorde cannot be used in Windows operating system")
			solver = TSPSolver.from_tspfile(fname)
			solution = solver.solve(verbose=True)
			tour=[index[i] for i in solution.tour]
			return tour
			pass 
		
	def tspMatch(self,clients,π):
		'''
		Solves a TSP for clients with dual variables π
		TODO: Use Concorde
		Input:
			clients: nodes for the TSP
			π: Duals for each node
		Output:
			tsp: Secuence of the optimal tour.
		'''
		self.SG=nx.DiGraph()		
		for i in clients:
			for j in clients:
				self.SG.add_edge(u_of_edge=i, v_of_edge=j,c=self.c_mile*self.dm[i][j]-π[i])
				self.SG.add_edge(u_of_edge=j, v_of_edge=i,c=self.c_mile*self.dm[i][j]-π[j])
		
		#nx.draw_networkx(self.SG,pos=self.pos,node_size=10,with_labels=False)		
		#plt.show()
		t=TSP.TSP(self.SG, self.pos)
		
		tsp=t.run_match_TSP(gur=True)
		return tsp

	def petal_generator(self,depot,clients,π):
		'''
		Heuristic for the generation of the Routes. Modification of the Split algorithm. Uses the pulse algorithm instead of a shortest path to solve the split graph.
		The routs generated are saved to the set of routes of the object (i.e., self.R), such as their costs.
		Input:
			depot: Depot from wher the routes will be generated
			clients: Clients that need to be covered
			π: dual variables for each node
		Output:
			r_costs: The reduced cost of the routes generated.
		'''	
		#self.print_dymacs(depot,clients,π)

		G=nx.DiGraph()
		l=[depot]+list(clients.index)
		
		#print('l',l)
		#TSP for given clients
		#print_log('\t\t\tEmpece con TSP')
		if len(l)>2:
			TSP=self.tsp(l,π)
		else:
			TSP=np.random.permutation(l)
		
		try:
			TSP = list( dict.fromkeys(TSP) )
			TSP.remove(depot)			
		except:
			pass
		#Create split graph.
		#print(depot)
		#print(TSP)
		#N=[depot]+list(TSP)

		N=[depot]+list(TSP)		
		#print('\t\t\tEmpece a crear splitG')
		ttt=tiempo.time()
		G.add_nodes_from(N)
		for i,s in enumerate(N):
			for j,t in enumerate(N[i+1:]):				
				dem=sum(clients[k] for k in N[i+1:i+j+2])
				r=[depot]+N[i+1:i+j+2]+[depot]				
				rz=list(zip(r[:-1],r[1:]))					
				dist=sum(self.dm[k][l] for k,l in rz)


				if dem<=self.cap and self.t_mile*dist<=self.route_time_limit:
					#print(self.t_mile*dist)
					cost=self.c_mile*dist-sum(π[i] for i in r[1:-1])
					time=self.t_mile*dist
					#Discounts the time from depot to first node
					#Check this part
					################################################################################
					if s==depot:
						time-= self.dm[s][N[1]]*self.t_mile
					G.add_edge(s,t,Cost=cost,Time=time,a=False)
		
		#print('\t\t',f'Me demoro {tiempo.time()-ttt} en split')
		'''
		Artificial arcs ar created. Making the instances feasible!
		'''
		#print('el num de arcos es: ', len(G.edges))
		G.add_node(0)
		G.add_edge(N[-1], 0, Cost=0,Time=0)

		
		#print_log('\t\t\tEmpece con el pulso...')
		PG=pulse.pulse_graph(G=G,source=depot,target=0,depot=depot,r=['Time'],r_limit=[self.route_time_limit])
		PG.preprocess()
		for i in N:
			'''
			Add arcs from each node to the end node with zero time, and high cost.
			This arc means than the rout ended without visiting the nodes that correspond to the artificial arc.
			'''
			if i!= depot:				
				if PG.G.nodes[i]['s_cost']>0:
					cost=PG.G.nodes[i]['s_cost']*10.2
				else:
					cost=PG.G.nodes[i]['s_cost']*0.08
				
				PG.G.add_edge(i,PG.target,Cost=cost,Time=0,a=True)

		
		#Print Nodes and depot for current split
		#print_log("#####################")
		#print_log('Nodes: ', N)
		#print_log('Nodes: ', len(N))
		#print_log('depot: ',depot)
		
		ttt=tiempo.time()
		P,c,t=PG.run_pulse()
		#print('\t\t',f'En el pulso me demoro {tiempo.time()-ttt}')
		#print_log("#####################")
		
		#print_log('pulse: ',P,c,t)
		#print_log('pulse: ',P,sum(G[s][t]['Cost']for s,t in zip(P[:-1],P[1:]) if not G[s][t]['a']),sum(G[s][t]['Time']for s,t in zip(P[:-1],P[1:])if not G[s][t]['a']))
		
		'''
		Some recoursive atempt of generating more routes... doest work well...
		
		try:
			if P[-1]!=N[-1]:
				
				#If the instance was infeasible, i.e. was not possible to create a route visiting all nodes meeting the time constraint, another route is created with the missing nodes for the split.
				
				c=N[N.index(P[-1])+1:]
				#print_log(f'\t\tPasa esto, con\nc={len(c)}\ny la lisya era {len(N)}')
				#self.petal_generator(depot, clients[c],π)
		except Exception as e:
			print('Este error:\n',e)
			#print_log('Este problema raro...')
			print(P)
			print(N)
			pass
		#petals=[[depot]+N[N.index(s)+1:N.index(t)+1]+[depot] for s,t in zip(P[:-1],P[1:])] 
		'''
		P=P[:-1]
		route=N
		route=route[:N.index(P[-1])+1]
		#print(len(N))
		#print_log(len(route))
		#print_log(route)
		#print_log(P)
		for i in P[1:]:
			k=route.index(i)
			route=route[:k+1]+[depot]+route[k+1:]
			#print_log('\t',route)
		

		r_costs=[G[s][t]['Cost']for s,t in zip(P[:-1],P[1:])]
		

		times=[G[s][t]['Time']for s,t in zip(P[:-1],P[1:])]		
		cost=self.c_mile*sum(self.dm[k][l] for k,l in zip(route[:-1],route[1:]))

		#print('depot index',[i for i in route if i == depot])

		#if len([i for i in route if i == depot])>20:
		#	pass
			#print_log('Routes: ',route)
			#print_log('Costs: ',costs,sum(costs))
			#print_log('Times: ',times,sum(times),'\n')
			#print_log('#############')

		#TODO: Solo meter rutas con costo reducido negativo???
		if route[0] in self.Ri.keys():
			self.Ri[route[0]].append(len(self.R))
		else:
			self.Ri[route[0]]=[len(self.R)]
		
		self.R.append(route)
		self.Route_cost.append(cost)

		return sum(r_costs)

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
			m._z={p:m.addVar(vtype=GRB.CONTINUOUS,name="m._z_"+str(p),lb=0, ub=1,obj=self.Route_cost[p]) for p in range(len(self.R))}
			#m._v={i:m.addVar(vtype=GRB.CONTINUOUS,name="m._v_"+str(i),lb=0, ub=10,obj=penal) for i in self.y_hat.keys()}
			m._v={i:m.addVar(vtype=GRB.CONTINUOUS,name="m._v_"+str(i),lb=0, ub=1,obj=penal+min([(self.dm[h][i]+self.dm[i][h])*self.c_mile for h in self.H])) for i in self.Clients}
			#m._v={i:m.addVar(vtype=GRB.CONTINUOUS,name="m._v_"+str(i),lb=0, ub=1,obj=penal) for i in self.Clients}
		else:
			m._z={p:m.addVar(vtype=GRB.BINARY,name="m._z_"+str(p),obj=self.Route_cost[p]) for p in range(len(self.R))}
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

		m._set_covering={i:m.addConstr(quicksum(m._z[k] for k,p in enumerate(self.R) if i in p)+m._v[i]>=1) for i in self.Clients}		

		#2.) Not exceeding number of vehicles per depot.

		m._Num_veh={i:m.addConstr(quicksum(m._z[k] for k,p in enumerate(self.R) if p[0]==i )<=self.y_hat[i]) for i in self.y_hat.keys()}

		
		#Uses saved basis:
		for i in self.z_hat.keys():
			m._z[i].start=self.z_hat[i]

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
		
	def petal_recicler(self,depot,close):		
		'''
		Ricicles routes from close nodes that have already solved the problem

		Input:
			depot for which we wanto to create routes
			close: Nodes to look from
		Output:
			None
			The list self.R is updated with routes staring at depot.
		'''

		num=0
		for n in close:
			for i in self.OptRi[n]:
				r=self.R[i]
				cost=self.Route_cost[i]
				time=self.t_mile*cost/self.c_mile		
				indexes=[]
				cambios=0	
				for ii,j in enumerate(r):
					if j==n:
						indexes.append(ii)
						if ii==0:
							rest=self.dm[r[ii]][r[ii+1]]
							cost-=self.c_mile*rest
							time-=min(self.t_mile*rest,2)					
							r[ii]=depot					
							cost+=self.c_mile*(self.dm[r[ii]][r[ii+1]])

						elif ii!=len(r)-1:
							rest=(self.dm[r[ii-1]][r[ii]]+self.dm[r[ii]][r[ii+1]])
							cost-=self.c_mile*rest
							time-=self.t_mile*rest
							r[ii]=depot
							suma=(self.dm[r[ii-1]][r[ii]]+self.dm[r[ii]][r[ii+1]])
							cost+=self.c_mile*suma
							time+=self.t_mile*suma

						else:					
							rest=self.dm[r[ii-1]][r[ii]]
							cost-=self.c_mile*rest
							t_fin=self.t_mile*rest
							time-=t_fin

							r[ii]=depot
							cost+=self.c_mile*(self.dm[r[ii-1]][r[ii]])				
				
				try:
					if time<=8 and time +t_fin<=9:
						num+=1
						if r[0] in self.Ri.keys():
							self.Ri[r[0]].append(len(self.R))
						else:
							self.Ri[r[0]]=[len(self.R)]								
						self.R.append(r)
						self.Route_cost.append(cost)
				except:
					pass
					
					#print_log(depot,r)
					
		print_log('\t\t######################################################')
		print_log('\t\t',f'reciclé {num} rutas')
		print_log('\t\t######################################################')

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
		f.write(f'λ:{λ*100}\n')
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
			if route[0] in self.Ri.keys():
				self.Ri[route[0]].append(len(self.R))
			else:
				self.Ri[route[0]]=[len(self.R)]
			
			#time=self.c_mile*sum(self.dm[k][l] for k,l in zip(route[:-1],route[1:]))
			cost=self.c_mile*sum(self.dm[k][l] for k,l in zip(route[:-1],route[1:]))
			
			self.R.append(route)
			self.Route_cost.append(cost)
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
			if route[0] in self.Ri.keys():
				self.Ri[route[0]].append(len(self.R))
			else:
				self.Ri[route[0]]=[len(self.R)]
			
			#time=self.c_mile*sum(self.dm[k][l] for k,l in zip(route[:-1],route[1:]))
			cost=self.c_mile*sum(self.dm[k][l] for k,l in zip(route[:-1],route[1:]))
			
			self.R.append(route)
			self.Route_cost.append(cost)
		
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
		tw=[(self.nLtw,self.nUtw/2),(self.nLtw+self.nUtw/2,self.nUtw)]
		
		def classifier(client,order=[0,1]):
			p=np.array(self.pos[client])
			if a.dot(p)>=b:
				return tw[order[0]]
			else: 
				return tw[order[1]]

		return classifier

def normal(x):
	'''
	Returns a normal vector to the one given by parameter (only for x \in \mathbb{R}^2)
	'''
	return np.array([-x[1],x[0]])

from math import sin, cos, sqrt, atan2, radians
def distance(lat1,lon1,lat2,lon2):
	'''
	Computes the distance between to georeferenced points
	Input: coordinates
	Output: distance
	'''

	R = 3958.8

	lat1 = lat1.map(radians)
	lon1 = lon1.map(radians)
	lat2 = radians(lat2)
	lon2 = radians(lon2)

	dlon = lon2 - lon1
	dlat = lat2 - lat1

	#a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
	a = dlat.map(lambda x: sin(x/2)**2) + lat1.map(cos) * cos(lat2) * dlon.map(lambda x:sin(x / 2)**2)

	c = a.map(lambda x: 2 * atan2(sqrt(x), sqrt(1 - x)))

	distance = R * c
	return distance

def fit_to_all_distributions(data,dist_names):	
	'''
	Recieves dataset and list with name of distributions
	Returns a dictionari {dist_name:params} with the parameters of the fit of each distribution passed by parameter.
	'''
	params = {}
	for dist_name in dist_names:
		try:
			dist = getattr(st, dist_name)
			param = dist.fit(data)

			params[dist_name] = param
		except Exception:
			print("Error occurred in fitting")
			params[dist_name] = "Error"

	return params 

def get_best_distribution_using_chisquared_test(data,params,distr_names):
	'''
	Gets the best distribution fit
	'''

	histo, bin_edges = np.histogram(data, bins='auto', normed=False)
	number_of_bins = len(bin_edges) - 1
	observed_values = histo
	
	dist_results = []

	for dist_name in distr_names:
		param = params[dist_name]
		if (param != "Error"):
			# Applying the SSE test
			arg = param[:-2]
			loc = param[-2]
			scale = param[-1]
			#print(getattr(st, dist_name))
			cdf = getattr(st, dist_name).cdf(bin_edges, loc=loc, scale=scale, *arg)
			expected_values = len(data) * np.diff(cdf)
			c , p = st.chisquare(observed_values, expected_values, ddof=number_of_bins-len(param))
			dist_results.append([dist_name, c, p])
	
	# select the best fitted distribution
	best_dist, best_c, best_p = None, sys.maxsize, 0

	for item in dist_results:
		name = item[0]
		c = item[1]
		p = item[2]
		if (not math.isnan(c)):
			if (c < best_c):
				best_c = c
				best_dist = name
				best_p = p

	# print the name of the best fit and its p value
	'''
	print("Best fitting distribution: " + str(best_dist))
	print("Best c value: " + str(best_c))
	print("Best p value: " + str(best_p))
	print("Parameters for the best fit: " + str(params[best_dist]))
	'''

	return best_dist, best_c, params[best_dist], dist_results

def fit_and_plot():
	'''
	Fits and plots distributions to each node...
	'''
	p=master()
	dist_names = ['laplace','norm','lognorm' ,'uniform','gamma']

	#for n in random.choice(list(self.V.index)):
	#for n in p.V.index:
	for i in range(10):
		n=random.choice(list(p.V.index))
		fig, ax = plt.subplots(1, 1)
		data=p.Demands.loc[n,:].loc[p.Demands.loc[n,:]>0]
		
		try:
			params = fit_to_all_distributions(data,dist_names)
			best_dist_chi, best_chi, params_chi, dist_results_chi = get_best_distribution_using_chisquared_test(data, params,dist_names)	
			arg = params_chi[:-2]
			loc = params_chi[-2]
			scale = params_chi[-1]

			x = np.linspace(getattr(st, best_dist_chi).ppf(0.001,loc=loc, scale=scale, *arg), getattr(st, best_dist_chi).ppf(0.999,loc=loc, scale=scale, *arg), 100)
			ax.plot(x, getattr(st, best_dist_chi).pdf(x,loc=loc, scale=scale, *arg),'g-', lw=2, alpha=1, label=best_dist_chi+' pdf')
		except:
			pass	
		ax.hist(data, density=True, histtype='stepfilled',alpha=0.5,label=str(n)+"'s Logaritmic returns")
		plt.ylabel("Density")
		ax.legend(loc='best', frameon=False)
		#plt.show()

		os.chdir('..')
		os.chdir('./Distrs')
		#plt.savefig(f'distr_{n}.png')
		#plt.clf()
		plt.show()



if __name__ == '__main__':
	m=master()
	#print(m.dist_m)
	#print(m.SPS[0].D)
	s=max(m.SPS,key=lambda x: sum(x.D))
	print(sum(s.D))
	s.y_hat={17728:5,16130:2,15427:5,15824:4,18014:6}
	s.H=list(s.y_hat.keys())
	m.solveSpJavaNotAllDepots(s)
	#m.Benders_algoMix(read=False)





	




