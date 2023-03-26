import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
def pq2ind1(p,q,P):
	ind=p+(q-1)*P
	return ind
P=50; Q=50
k=P*Q
sol=np.load('sol.npy')
time1=np.arange(0,1001,0.1)
time=time1[0::10]

for t in np.arange(0,len(time),1):
	fig, ax = plt.subplots()
	plt.title('time={}'.format(time[t]))
	#N=sol[t][:k] #to plot Notch
	D=sol[t][k:2*k] #to plot Delta
	#J=sol[t][2*k:3*k] #to plot Jagged
	#I=sol[t][3*k:] #to plot NICD
	colors=[]
	patches=[]
	for p0 in np.arange(1,P+1,1):
		for q0 in np.arange(1,Q+1,1):
			ind = pq2ind1(p0,q0,P)
			colors.append(D[ind-1])
			s=np.sqrt(3)/4; q=q0*3/4; p=p0*2*s
			if q0/2==round(q0/2):
				p=p+s
			hex=Polygon([[q-0.5,p],[q-0.25,p+s],[q+0.25,p+s],[q+0.5,p],[q+0.25,p-s],[q-0.25,p-s]],edgecolor='k',lw=1)
			patches.append(hex)
	collection = PatchCollection(patches,cmap=matplotlib.cm.jet,match_original=True,alpha=1)
	collection.set_array(np.array(colors))
	ax.add_collection(collection)
	fig.colorbar(collection,ax=ax)
	#collection.set_clim([0,50])
	ax.autoscale_view()
	fig.tight_layout()
	ax.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, right=False, left=False, labelleft=False)
	plt.savefig('figs/{}'.format(t))
