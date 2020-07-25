import os
import sys
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animat
import pandas as pd
import random as rnd

sys.path.append('../src')
from main import *


def readNet():
	os.chdir('SiouxFalls')

	G=nx.DiGraph()

	f=open('SiouxFalls_net.txt')
	f.readline()
	for l in f:		
		l=list(map(float,l.replace('\n','').split('\t')))
		G.add_edge(100*int(l[0]),100*int(l[1]),d=int(l[2]/20))
	f.close()

	f=open('SiouxFalls_pos.txt')
	f.readline()
	pos={}
	for l in f:				
		l=list(map(float,l.replace('\n','').split('\t')[:-1]))
		pos[100*int(l[0])]=(l[1],l[2])		
		
	f.close()
	os.chdir('..')

	return G, pos

def randomDemand(G):
	act=np.random.binomial(1,0.4,len(G.nodes()))
	dems=dict(zip(G.nodes(),np.random.poisson(lam=10,size=len(G.nodes()))*act))
	return dems

def drawDem(G,pos,dem,ax):	
	dem=dict(filter(lambda x: x[1]>0,dem.items()))
	
	try:
		m=max(dem.values())
		size=[500*i/m for i in dem.values()]
		nx.draw_networkx_nodes(G,pos,nodelist=list(dem.keys()),node_size=size,node_color='red',alpha=0.6,ax=ax)
	except:
		pass
	nx.draw_networkx_nodes(G,pos,node_size=30,node_color='black',ax=ax)#nodelist=set(G.nodes())-set(dem.keys())
	nx.draw_networkx_edges(G,pos,ax=ax)
		
	

	return ax

def demandAnimation(n):
	G, pos=readNet()
	#fig,ax= plt.subplots()
	fig=plt.figure(figsize=(10,8))
	grid=fig.add_gridspec(21, 21, wspace=0, hspace=0)
	#grid = plt.GridSpec(21, 21, wspace=0, hspace=0)
	
	ax=plt.subplot(grid[1:20, 0:20])	
	axk =plt.subplot(grid[0, 0])
	
	axk.axis('off')
	ax.axis('off')
	


	def f(k):
		ax.clear()
		axk.clear()
		axk.text(0,0,f'Day {k+1}',fontweight='bold')
		drawDem(G,pos,randomDemand(G),ax)		
		axk.axis('off')
		ax.axis('off')

	ani = animat.FuncAnimation(fig, f, frames=n,interval=1000, repeat=True)
	return ani

def saveAnim(anim,name):
	Writer = animat.writers['ffmpeg']
	writer = Writer(fps=1, metadata=dict(artist='Me'), bitrate=2800)
	anim.save(f'media/{name}.mp4', writer=writer)

def calcDistM(G):
	n=G.nodes()
	d=dict(nx.all_pairs_dijkstra_path_length(G,weight='d'))
	#mat=np.array([[d[i][j]for j in n]for i in n])
	mat=pd.DataFrame(d)

	return mat

def createMaster(n):
	G, pos=readNet()
	dist_m=calcDistM(G)	
	
	m=master(load=False)

	m.G=G
	m.pos = pos
	m.SPS=[]
	m.dist_m=calcDistM(G)
	m.Adj=(m.dist_m*m.t_mile).applymap(lambda x: 1 if x<=100000 else 0)
	#print(m.dist_m)
	for i in range(n):
		#di=self.Demands[i][self.Demands[i]>0]
		di=randomDemand(G)
		di=dict(filter(lambda x: True if x[1]>0 else False,di.items()))
		di=pd.DataFrame(di.values(),index=di.keys())
		
		#di_mat=self.dist_m.loc[list(di.index),list(di.index)]
		m.SPS.append(sub_problem(id=len(m.SPS),H=[400,1500],D=di,dm=m.dist_m,pos=m.pos,master=m))
		m.SPS[-1].y_hat={400:2,1500:2}

	return m


def paintRoute(p,ax):
	pet_col='orange'	
	for i,j in zip(p[:-1],p[1:]):				
		path=nx.shortest_path(m.G,i,j,weight='c')
		path=list(zip(path[:-1],path[1:]))
		#print('este pat', path)
		nx.draw_networkx_edges(m.G, m.pos,edgelist=path,edge_color=pet_col,width=3,ax=ax)
	
	


def plotSol(sp,ax):
	dem=sp.D.to_dict()[0]
	drawDem(sp.master.G,sp.master.pos,dem,ax)
	petals=sp.OptRi
	h=sp.H
	m.plot_petals(petals,h, pet=False,ax=ax)
	#print(sp.D)
	#print(sp.OptRi)
	#print(sorted([i for i in dem.keys() if dem[i]>0]))

def solAnimation(n):
	global m
	m=createMaster(n)

	for sp in m.SPS:		
		m.solveSpJava(sp,gap=0.000001)	
	
	
	fig=plt.figure(figsize=(10,8))
	grid=fig.add_gridspec(21, 21, wspace=0, hspace=0)
	#grid = plt.GridSpec(21, 21, wspace=0, hspace=0)
	
	ax=plt.subplot(grid[1:20, 0:20])	
	axk =plt.subplot(grid[0, 0])
	
	#axk.axis('off')
	#ax.axis('off')
	

	def f(k):
		ax.clear()
		axk.clear()
		axk.text(0,0,f'Day {k+1}',fontweight='bold')		
		plotSol(m.SPS[k],ax)
		#drawDem(m.G,m.pos,randomDemand(m.G),ax)		
		axk.axis('off')
		ax.axis('off')

	ani = animat.FuncAnimation(fig, f, frames=n,interval=1000, repeat=True)
	return ani


def firstStageAnim(n):
	G, pos=readNet()
	fig,ax=plt.subplots(figsize=(10,8))
	ax.axis('off')
	def f(k):
		ax.clear()
		kk=rnd.randint(2,4)
		nod=rnd.sample(list(G.nodes()),kk)
		nx.draw_networkx_nodes(G,pos,node_size=30,node_color='black',ax=ax)
		nx.draw_networkx_edges(G,pos,ax=ax)
		nodeSize=[rnd.randint(2,4)*400 for i in nod]
		nx.draw_networkx_nodes(G,pos,nodelist=nod,node_shape='s',node_size=nodeSize,node_color='green',ax=ax)		
		ax.axis('off')

	ani = animat.FuncAnimation(fig, f, frames=n,interval=1000, repeat=True)
	return ani


def paintRoutes():	
	sp=m.SPS[0]
	dem=sp.D.to_dict()[0]
	os.chdir('../src')
	m.solveSpJava(sp)
	os.chdir(path)
	delete_all_files(path='media/Routes',exceptions=[])
	fig,ax= plt.subplots(figsize=(10,8))
	
	h=[i for i,y in sp.y_hat.items() if y>0]
	for i,p in enumerate(sp.R):
		if random.random()>0.9:
			fig,ax= plt.subplots(figsize=(10,8))
			drawDem(m.G,m.pos,dem,ax)
			nx.draw_networkx_nodes(m.G, m.pos,nodelist=h,node_shape='s',node_size=200,node_color='green',ax=ax)	
			paintRoute(p,ax)		
			ax.axis('off')
			plt.savefig(f'media/Routes/r{i}.png')
			#plt.show()
	print(sp.z_hat.items())
	for i,z in sp.z_hat.items():
		if z>0:
			fig,ax= plt.subplots(figsize=(10,8))
			drawDem(m.G,m.pos,dem,ax)
			nx.draw_networkx_nodes(m.G, m.pos,nodelist=h,node_shape='s',node_size=200,node_color='green',ax=ax)	
			paintRoute(sp.R[i],ax)		
			ax.axis('off')
			plt.savefig(f'media/Routes/rSol{i}.png')
			#plt.show()


if __name__=='__main__':
	
	path=os.getcwd()
	m=createMaster(10)

	paintRoutes()
	pass
	











