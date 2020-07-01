
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
import matplotlib
from matplotlib import cbook, docstring, rcParams

import pylab
import matplotlib.animation
from scipy.linalg import expm, sinm, cosm

import math
import time as tiemm
from typing import List

import pandas as pd

L_string = List[str]

	

class pulse_graph():
	"""docstring for pulse_graph"""
	def __init__(self, G: nx.DiGraph,depot: int,source: int,target: int,r: L_string,r_limit: list,n=None,replen=dict(),tightness=0.5 ,pos=[]):
		super(pulse_graph, self).__init__()
		
		'''
		NetworkX graph, with an atribute of 'Cost' and for each i \in rescources.
		'''

		self.G = G
		self.Primal_Bound=float("inf")
		self.Fpath=[]
		self.R=r

		self.Resource=[0 for i in self.R]

		'''
		Time constraint
		'''
		self.R_max=r_limit
		
		'''
		Source node
		'''
		self.source=source
		
		'''
		Target node
		'''
		self.target=target

		'''
		If replenishment is allowed is a dictionary that has as key the rescource that restarts, and as value, the name of the atribute that indicates if the arc is a replenishment arc or not.

		'''	

		self.replen=replen

		'''
		Real depot node
		'''
		self.depot= depot

		'''
		Parameters for boundind strategy whe runing elemetary shortest path problem.
		'''
		self.n=n
		if n!=None:
			
			'''

			'''
			self.t_hat=self.R_max[0]

			'''
			nonnegative time step
			'''
			self.Δ=self.t_hat/n

			'''
			Bower bound matrix denoted by B=􏰖[r(vi,τ)􏰒􏰜􏰕􏰗]
			'''
			self.B=pd.DataFrame(index =self.G.nodes)
			print(self.B) 

		self.anim=[]
		self.pos=pos
		self.bound=0
		self.infeas=0
		self.dom=0
		self.tightness=tightness

		
	def calc_cost(self,p):
		'''
		Computes cost and time for the given path.
		'''
		edges=zip(p,p[1:])
		cost=0
		time=0
		for (i,j) in edges:
			cost+=self.G[i][j]["Cost"]
			time+=self.G[i][j]["Time"]
		return (cost, time)
	
	def preprocess(self):
		'''
		Preprocess the graph labeling every node with the shortest cost to the end node (target)
		'''

		#nx.johnson(self.G,weight='Cost')

		#G_r=nx.DiGraph(self.G).reverse(copy=True)
		

		for i,r in enumerate(self.R):
			t=nx.shortest_path_length(self.G,weight=r,target=self.target)
			attrs={i:{"s_"+r:t[i]} for i in t.keys()}
			attrs.update({i:{"s_"+r:float("inf")} for i in self.G.nodes() if i not in t.keys()})
			nx.set_node_attributes(self.G, attrs)
			#self.minimum_time=attrs[self.source]["s_time"]		

			if self.R_max[i]==0:
				try:
					self.R_max[i]=self.G[self.source]['s_'+r] *(1+self.tightness)
				except:
					print("Infeasible") 
		
		if self.n==None:
			
			#p=nx.shortest_path_length(self.G,weight="Cost",target=self.target)
			
			pred, p=nx.bellman_ford_predecessor_and_distance(nx.reverse(self.G,copy=True),weight="Cost",source=self.target)
			
			attrs={i:{"labels":[],"s_cost":p[i]} for i in p.keys()}
			attrs.update({i:{"labels":[],"s_cost":float("inf")} for i in self.G.nodes() if i not in p.keys()})
			nx.set_node_attributes(self.G, attrs)
		else:
			self.bounding_scheme()

		for i in self.G.nodes:
			self.G.nodes[i]["labels"]=[]


	def bounding_scheme(self):
		
		τ=self.t_hat
		while τ>0:
			τ-=self.Δ
			col = pd.DataFrame({τ: [float('inf') for i in self.G.nodes]}, index=self.G.nodes)
			self.B=self.B.join(col)
			print('#############################')
			
			print(τ)
			print (self.B)


			nod=self.sort_by(self.G.nodes,by='s_'+self.R[0])
			for i in nod:
				print('Voy desde ',i)
				print('Con tiempo de ', self.G.nodes[i]['s_'+self.R[0]])

				self.Fpath=[]
				self.Primal_Bound=float("inf")
				self.Resource=[τ if i==self.R[0] else 0 for i in self.R]
				#print(self.Fpath)
				self.pulse(vk=i,c=0,r=[τ if i==self.R[0] else 0 for i in self.R],P=self.Fpath)
				print(self.Primal_Bound)
				self.B.loc[i][τ]=self.Primal_Bound

		return None




	def C_loops(self,vk,P):
		'''
		Checks for loops
		'''
		bool=True	#We assume that the current path passes the dominance check (i.e is not dominated)
		if vk in P:# and vk!=self.depot:
			bool=False

		return bool
	'''
	def C_Dominance(self,vk,c,t):
		
		#Checks dominance
		
		bool=True	#We assume that the current path passes the dominance check (i.e is not dominated)
		for (i,(cc,tt)) in enumerate(self.G.nodes[vk]["labels"]):
			if c>cc and t>tt:
				bool=False
				self.dom+=1
		return bool
	'''
	
	def C_Feasibility(self,vk,r):
		'''
		Check Feasibility
		'''
		bool=True #We assume that the current path passes the Feasibility check (i.e is feasible)

		'''
		print(sum(r_i+self.G.nodes[vk]["s_"+s_r]>R_max for r_i,R_max,s_r in zip(r,self.R_max,self.R))>0)
		print([r_i+self.G.nodes[vk]["s_"+s_r] for r_i,R_max,s_r in zip(r,self.R_max,self.R)])
		print(self.R_max)
		'''

		if sum(r_i+self.G.nodes[vk]["s_"+s_r]>R_max for r_i,R_max,s_r in zip(r,self.R_max,self.R))>0:
			bool=False
			self.infeas+=1
		return bool
	
	def LB_r(self,vk,t):
		T=self.B.columns
		j=max([k for k in T if k <= t])
		#print('r(vk,t): ',t, j)
		return self.B.loc[vk][j]

	def C_Bounds(self,vk,c,t):
		'''
		Check Bounds
		'''
		if self.n==None:
			bool=True #We assume that the current path passes the primal_Bound check (i.e in the bes casenario the path is better than the PB)
			if c+self.G.nodes[vk]["s_cost"]>self.Primal_Bound:
				#print("bound")
				bool=False
				self.bound+=1
		else:
			bool=True #We assume that the current path passes the primal_Bound check (i.e in the bes casenario the path is better than the PB)
			if c+self.LB_r(vk,t) >self.Primal_Bound:
				#print("bound")
				bool=False
				self.bound+=1

		return bool
	'''
	def path_completion(self,vk,c,t,P):
		
		#Check path completion
		
		bool=True #We assume that the current path passes the path_completion check (i.e is not possible to complete the path)
		if (c + self.G.nodes[vk]["s_cost"]<self.Primal_Bound) and (t+self.G.nodes[vk]["s_time"]<=self.T_max):
			#self.update_primal_bound(c + self.G.nodes[vk]["s_cost"],t+self.G.nodes[vk]["s_time"],P)
			bool=True
			#print("its working")
		return(bool)
	#Update the labels of a given node vk
	'''
	'''
	def update_labels(self,vk,c,t):	
		self.G.nodes[vk]["labels"].append((c,t))
	'''
	def sort(self,sons):
		return(sorted(sons,key=lambda x: self.G.nodes[x]["s_cost"] ))

	def sort_by(self,sons,by):
		return(sorted(sons,key=lambda x: self.G.nodes[x][by]))

	def update_primal_bound(self,c,r:list,P):
		if c<=self.Primal_Bound and sum(r_i<=R_max for r_i,R_max in zip(r,self.R_max))==len(self.R_max):
			self.Primal_Bound=c
			self.Fpath=P
			self.Resource=r
			#print("Nuevo PB, costo: ",self.Primal_Bound,"tiempo: ",self.Resource)
	
	def update_R_consuption(self,vk,i,r):
		rr=[r_i + self.G.edges[vk,i][s_r] for r_i,s_r in zip(r,self.R)]
		for r_i in self.replen.keys():
			if self.G.edges[vk,i][self.replen[r_i]]==1:
				rr[self.R.index(r_i)]=0
		return rr
	
	def pulse(self,vk:int,c:float,r:list,P:list):
		#self.update_labels(vk,c,r)

		if vk==self.target:
			if vk not in P:
				self.update_primal_bound(c,r,P+[vk])
				#print("LLegue a ",vk, "Con tiempo de ",r, " Y costo de ",c)
		#print("LLegue a ",vk, "Con tiempo de ",r, " Y costo de ",c)
		#self.C_Dominance(vk,c,t) and and self.path_completion(vk,c,t,P)
		if ( self.C_Feasibility(vk,r) and self.C_Bounds(vk,c,r[0]) and self.C_loops(vk,P)):
			PP=P.copy()			
			PP.append(vk)
			
			try:
				suc=self.sort(self.G.successors(vk))
			except:
				
				#suc=sorted(self.G.successors(vk),key=lambda x:self.LB_r(x,r[0]))
				suc=self.sort_by(self.G.successors(vk),by='s_'+self.R[0])
			for i in suc:
				cc=c+self.G.edges[vk,i]["Cost"]				
				rr=self.update_R_consuption(vk,i,r)
				#rr=[r_i + self.G.edges[vk,i][s_r] for r_i,s_r in zip(r,self.R)]
				self.pulse(i,cc,rr,PP)
	
	def run_pulse(self):
		self.Fpath=[]
		self.Primal_Bound=float("inf")
		self.Resource=[0 for i in self.R]
		#print(self.Fpath)
		self.preprocess()
		if self.G.nodes[self.source]["s_cost"]!=np.Infinity:
			self.pulse(vk=self.source,c=0,r=[0 for i in self.R],P=self.Fpath)
		else:
			print("The instance is infeasible")
		
		return self.Fpath, self.Primal_Bound,self.Resource

	
	#Draws the graph with a given position and a given path
	def draw_graph(self,path=[],bgcolor="white",edge_color="black",arc_color="gray",path_color="red"):
		if self.pos==[]:
			self.pos = nx.random_layout(self.G)
		if path==[]:
			path=self.Fpath

		fig= plt.figure(figsize=(12,6))
		edge_labels={e:(int(self.G.edges[e]["Cost"]),int(self.G.edges[e]["Time"])) for e in self.G.edges}
		BGnodes=set(self.G.nodes()) - set(path)
		nx.draw_networkx_edges(self.G, pos=self.pos, edge_color=arc_color)
		null_nodes = nx.draw_networkx_nodes(self.G, pos=self.pos, nodelist=BGnodes, node_color=bgcolor)#node_size=1000
		null_nodes.set_edgecolor(edge_color)
		nx.draw_networkx_labels(self.G, pos=self.pos, labels=dict(zip(BGnodes,BGnodes)),  font_color="black")
		try:
			query_nodes=nx.draw_networkx_nodes(self.G,pos=self.pos,nodelist=path,node_color=path_color)#,node_size=1000
			query_nodes.set_edgecolor(path_color)
		except:
			pass
		nx.draw_networkx_labels(self.G,pos=self.pos,labels=dict(zip(path,path)),font_color="black")
		edgelist = [path[k:k+2] for k in range(len(path) - 1)]
		nx.draw_networkx_edges(self.G, pos=self.pos, edgelist=edgelist, width=4,edge_color=path_color)

	    #Edge labels
		nx.draw_networkx_edge_labels(self.G, self.pos, edge_labels = edge_labels )
		plt.axis('off')
		#plt.figure(num=None, figsize=(6, 3), dpi=80*3, facecolor='w', edgecolor='k')
		#plt.show()	
		#return fig
		



'''
insts=["USA-road-BAY.txt","network10.txt"]

G=create_graph(insts[1], headder=False)
#PG=pulse_graph(G=G,T_max=3500,source=1,target=3301)
# 6236
#(4-5)*tiddes +tmin
PG=pulse_graph(G=G,T_max=9000,source=0,target=6236)

#pos=nx.spring_layout(G)
sol=PG.run_pulse()
print(sol)
#PG.draw_graph(path=sol[0])

'''




