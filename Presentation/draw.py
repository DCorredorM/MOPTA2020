import os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animat
p=os.getcwd()
os.chdir('../src')
print(os.listdir())
from main import *
os.chdir(p)

def readNet():
	os.chdir('SiouxFalls')

	G=nx.DiGraph()

	f=open('SiouxFalls_net.txt')
	f.readline()
	for l in f:		
		l=list(map(float,l.replace('\n','').split('\t')))
		G.add_edge(int(l[0]),int(l[1]))	
	f.close()

	f=open('SiouxFalls_pos.txt')
	f.readline()
	pos={}
	for l in f:				
		l=list(map(float,l.replace('\n','').split('\t')[:-1]))
		pos[int(l[0])]=(l[1],l[2])		
		
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

def animation(G,pos,n):
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





if __name__=='__main__':
	
	G, pos=readNet()	
	dem=randomDemand(G)
	fig,ax=plt.subplots(figsize=(10,8))
	drawDem(G,pos,dem,ax)
	ax.axis('off')	
	plt.savefig(f'media/netDem.png')
	plt.clf()




	#ani=animation(G,pos,n=20)
	#saveAnim(ani,'demandDays')
	#plt.show()





