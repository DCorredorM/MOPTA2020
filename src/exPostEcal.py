from main import *
import pickle
import time
import pandas as pd
import os
p=master()
p.Demands=p.import_data('mopta2020_q2018.csv',h=0,names=['id']+list(range(1,366))).set_index('id')
mainDir=os.getcwd()
os.chdir('../Results')
resDir=os.getcwd()
h=3
y_hat={}
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
		pickle_in = open(f'sp{id}.sp','rb')
		sp = pickle.load(pickle_in)
		p.SPS.append(sp)		
		#print(f'pikled {id}')
	except:		
		p.SPS.append(sub_problem(id=len(p.SPS),H=[],D=di,dm=p.dist_m,pos=p.pos))
		#print(f'No pikle for {id}')
		pass
	id+=1

'''

'''
day=0


sp=p.SPS[day]
tspi=time.time()
sp.y_hat=y_hat#{15427:6.0,15824:6.0,18014:9.0}#
sp.H=list(sp.y_hat.keys())
os.chdir(mainDir)
FOi,λ,π=p.solveSpJava(sp)
m=sp.master_problem()
m.optimize()

os.chdir(resDir)
os.chdir('./SecondStage')

file_sp=open(f'sp{day}.sp','wb')
pickle.dump(sp, file_sp)


sp.z_hat={k:kk.x for k,kk in m._z.items()}
petals=[sp.R[i] for i,z in sp.z_hat.items() if z>0.5]
p.plot_dem(sp.D,'green')
p.plot_petals(petals=petals,h=sp.H,pet=False)
plt.show()


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
		NoAtendidos+=1


f=open(f'summary{h}.txt','a')
f.write(f'{day}\t{m.objVal}\t{NoAtendidos}'+'\n')
f.close()

'''

'''
def dataTable(df,name,AIMMSname):
	f=open(name,'w')
	f.write(AIMMSname+":=DATA TABLE\n")
	f.write('\t\t'+''.join([f"{i}"+'\t' for i in df.columns])+'\n')
	f.write('!\t\t'+"---------\t"*len(df.columns)+'\n')
	for i in df.index:
		f.write('\t'+f"{i}\t"+''.join([str(round(df[j][i],2))+'\t' for j in df.columns])+'\n')
	f.close()



d2018=p.import_data('mopta2020_q2018.csv',h=0,names=['id']+list(range(1,366))).set_index('id')
d2019=p.import_data('mopta2020_q2019.csv',h=0,names=['id']+list(range(367,367+365))).set_index('id')

print(d2018)
print(d2019)

demands=pd.concat([d2018,d2019],axis=1, sort=False) 
print(demands)



dataTable(demands,'DeandDays.txt','Demand(i,t)')
'''
