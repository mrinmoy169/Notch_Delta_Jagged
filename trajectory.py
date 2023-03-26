import numpy as np
import scipy.integrate

def pq2ind(p,q,P):
	ind=p+q*P
	return ind

def ind2pq(ind,P):
	q=np.floor(ind/P)
	p=ind-q*P
	return (p,q)

def findneighborhex(ind,P,Q):
	p=ind2pq(ind,P)[0]
	q=ind2pq(ind,P)[1]
	out1=pq2ind(np.mod(p+1,P),q,P)
	out2=pq2ind(np.mod(p-1,P),q,P)
	qleft = np.mod(q-1,Q)
	qright = np.mod(q+1,Q)
	if q/2!=round(q/2):
		pup = np.mod(p+1,P)
		pdown = p
	else:
		pup = p
		pdown=np.mod(p-1,P)
	out3=pq2ind(pup,qleft,P)
	out4=pq2ind(pdown,qleft,P)
	out5=pq2ind(pup,qright,P)
	out6=pq2ind(pdown,qright,P)
	return (out1,out2,out3,out4,out5,out6)

def getconnectivityM(P,Q):
	k=P*Q
	M=np.zeros((k,k))
	w=1/6
	for s in np.arange(0,k,1):
		kneighbor=findneighborhex(s,P,Q)
		for r in np.arange(0,6,1):
			M[s][int(kneighbor[r])]=w
	return M

#ode
def eq(y,lambdaN,lambdaD,lambdaJ):
	kc=0.25; kt=0.04; gamma=0.1; gammaI=0.5
	k=P*Q
	M=getconnectivityM(P,Q)
	N=y[0:k]; D=y[k:2*k]; J=y[2*k:3*k]; I=y[3*k:4*k]
	Next=np.dot(M,y[0:k]); Dext=np.dot(M,y[k:2*k]); Jext=np.dot(M,y[2*k:3*k])
	dN=lambdaN*(1+I**2/(1+I**2))-kc*N*(D+J)-kt*N*(Dext+Jext)-gamma*N
	dD=lambdaD/(1+I**2)-kc*D*N-kt*D*Next-gamma*D
	dJ=lambdaJ*(1+I**5/(1+I**5))-kc*J*N-kt*J*Next-gamma*J
	dI=kt*N*(Dext+Jext)-gammaI*I
	list=[dN,dD,dJ,dI]
	flat_list = []
	for sublist in list:
		for item in sublist:
			flat_list.append(item)
	return flat_list

#parameters
P=50; Q=50
lambdaN=5.0; lambdaD=10.0; lambdaJ=0.5
noise=0.5
t=np.arange(0,1001,0.1)

#initial condition
k=P*Q
N0=[]
for i in np.arange(0,k,1):
        N0.append(lambdaN)
D0=[]
for i in np.arange(0,k,1):
        D0.append(0.00001*lambdaD*(1+noise*(np.random.uniform(0,1)-1/2)))
J0=[]
for i in np.arange(0,k,1):
        J0.append(0.00001*lambdaJ*(1+noise*(np.random.uniform(0,1)-1/2)))
I0=np.zeros(k)
list=[N0,D0,J0,I0]
inic = []
for sublist in list:
	for item in sublist:
		inic.append(item)

#integration
def Euler(eq, X0, t):
        dt = t[1] - t[0]
        X  = [np.zeros(len(X0)) for x in range(len(t))]
        X[0] = X0
        for i in range(len(t)-1):
                X[i+1] = X[i] + np.dot(eq(np.array(X[i]),lambdaN,lambdaD,lambdaJ),dt)
        return X
sol=Euler(eq,inic,t)
np.save('sol.npy',sol[0::10]) #save every 10th frame
