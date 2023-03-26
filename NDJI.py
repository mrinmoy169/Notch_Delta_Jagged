import sympy
import numpy as np
import scipy.integrate
from functools import partial
#from scipy.optimize import fsolve
sympy.init_printing()
NA, NB, NC, DA, DB, DC, JA, JB, JC, IA, IB, IC = sympy.symbols("NA, NB, NC, DA, DB, DC, JA, JB, JC, IA, IB, IC")
kc, kt, gamma, gammaI, lambdaN, lambdaD, lambdaJ = sympy.symbols("kc, kt, gamma, gammaI, lambdaN, lambdaD, lambdaJ")

#parameters
gamma=0.1; gammaI=0.5; nN=2; nD=2; nJ=5

#ode
dNAdt = lambdaN*(1+IA**nN/(1+IA**nN))-NA*(kc*(DA+JA)+0.5*kt*(DB+DC+JB+JC))-gamma*NA
dNBdt = lambdaN*(1+IB**nN/(1+IB**nN))-NB*(kc*(DB+JB)+0.5*kt*(DC+DA+JC+JA))-gamma*NB
dNCdt = lambdaN*(1+IC**nN/(1+IC**nN))-NC*(kc*(DC+JC)+0.5*kt*(DA+DB+JA+JB))-gamma*NC
dDAdt = lambdaD/(1+IA**nD)-DA*(kc*NA+0.5*kt*(NB+NC))-gamma*DA
dDBdt = lambdaD/(1+IB**nD)-DB*(kc*NB+0.5*kt*(NC+NA))-gamma*DB
dDCdt = lambdaD/(1+IC**nD)-DC*(kc*NC+0.5*kt*(NA+NB))-gamma*DC
dJAdt = lambdaJ*(1+IA**nJ/(1+IA**nJ))-JA*(kc*NA+0.5*kt*(NB+NC))-gamma*JA
dJBdt = lambdaJ*(1+IB**nJ/(1+IB**nJ))-JB*(kc*NB+0.5*kt*(NC+NA))-gamma*JB
dJCdt = lambdaJ*(1+IC**nJ/(1+IC**nJ))-JC*(kc*NC+0.5*kt*(NA+NB))-gamma*JC
dIAdt = 0.5*kt*NA*(DB+DC+JB+JC)-gammaI*IA
dIBdt = 0.5*kt*NB*(DC+DA+JC+JA)-gammaI*IB
dICdt = 0.5*kt*NC*(DA+DB+JA+JB)-gammaI*IC

sys = sympy.Matrix([dNAdt, dNBdt, dNCdt, dDAdt, dDBdt, dDCdt, dJAdt, dJBdt, dJCdt, dIAdt, dIBdt, dICdt])
var = sympy.Matrix([NA, NB, NC, DA, DB, DC, JA, JB, JC, IA, IB, IC])
jac = sys.jacobian(var)

#convert jac to a function:
jacobian = sympy.lambdify((NA, NB, NC, DA, DB, DC, JA, JB, JC, IA, IB, IC, kc, kt,lambdaN, lambdaD, lambdaJ), jac, dummify=False)

#steady states
def eq(p,kc,kt,lambdaN,lambdaD,lambdaJ):
        NA,NB,NC,DA,DB,DC,JA,JB,JC,IA,IB,IC=p
        dNAdt = lambdaN*(1+IA**nN/(1+IA**nN))-NA*(kc*(DA+JA)+0.5*kt*(DB+DC+JB+JC))-gamma*NA
        dNBdt = lambdaN*(1+IB**nN/(1+IB**nN))-NB*(kc*(DB+JB)+0.5*kt*(DC+DA+JC+JA))-gamma*NB
        dNCdt = lambdaN*(1+IC**nN/(1+IC**nN))-NC*(kc*(DC+JC)+0.5*kt*(DA+DB+JA+JB))-gamma*NC
        dDAdt = lambdaD/(1+IA**nD)-DA*(kc*NA+0.5*kt*(NB+NC))-gamma*DA
        dDBdt = lambdaD/(1+IB**nD)-DB*(kc*NB+0.5*kt*(NC+NA))-gamma*DB
        dDCdt = lambdaD/(1+IC**nD)-DC*(kc*NC+0.5*kt*(NA+NB))-gamma*DC
        dJAdt = lambdaJ*(1+IA**nJ/(1+IA**nJ))-JA*(kc*NA+0.5*kt*(NB+NC))-gamma*JA
        dJBdt = lambdaJ*(1+IB**nJ/(1+IB**nJ))-JB*(kc*NB+0.5*kt*(NC+NA))-gamma*JB
        dJCdt = lambdaJ*(1+IC**nJ/(1+IC**nJ))-JC*(kc*NC+0.5*kt*(NA+NB))-gamma*JC
        dIAdt = 0.5*kt*NA*(DB+DC+JB+JC)-gammaI*IA
        dIBdt = 0.5*kt*NB*(DC+DA+JC+JA)-gammaI*IB
        dICdt = 0.5*kt*NC*(DA+DB+JA+JB)-gammaI*IC
        return(dNAdt, dNBdt, dNCdt, dDAdt, dDBdt, dDCdt, dJAdt, dJBdt, dJCdt, dIAdt, dIBdt, dICdt)

def findroot(func,init):
        sol, info, convergence, sms = scipy.optimize.fsolve(func, init, full_output=1)
        if convergence == 1:
                return sol
        return np.array([np.nan]*len(init))

def unique_rows(aa):
        bb = np.ascontiguousarray(aa)
        unique = np.unique(bb.view([('', bb.dtype)]*bb.shape[1]))
        return unique.view(bb.dtype).reshape((unique.shape[0], bb.shape[1]))

def find_unique_sol(kc,kt,lambdaN,lambdaD,lambdaJ):
        s=[]
        for i in np.arange(1,300,1):
                a=np.random.uniform(0,50)               #decide the range from the solutions of N
                b=np.random.uniform(0,50)
                c=np.random.uniform(0,50)
                d=np.random.uniform(0,50)
                inic=(np.random.uniform(0,50),np.random.uniform(0,50),a,np.random.uniform(0,50),np.random.uniform(0,50),b,np.random.uniform(0,50),np.random.uniform(0,50),c,np.random.uniform(0,50),np.random.uniform(0,50),d)
                sol=findroot(lambda x: eq(x,kc,kt,lambdaN,lambdaD,lambdaJ),inic)
                s.append(sol)
        s1=np.array(np.around(s,decimals=5))
        s2=unique_rows(s1)
        s3=[row for row in s2 if not np.isnan(row).any()]   # remove nan
        s4=[row for row in s3 if all(row[:][:] > 0)]   # remove negative sol
        s5=[row for row in s4 if np.isclose((row[:][1]-row[:][2]),0)]   #kepping those sol only where NB=NC
        return s5

def stable_I(kc,kt,lambdaN,lambdaD,lambdaJ):
        ss = find_unique_sol(kc,kt,lambdaN,lambdaD,lambdaJ)
        ss3=[]
        for i in np.arange(0,len(ss),1):
                eigv = np.linalg.eigvals(jacobian(ss[i][0],ss[i][1],ss[i][2],ss[i][3],ss[i][4],ss[i][5],ss[i][6],ss[i][7],ss[i][8],ss[i][9],ss[i][10],ss[i][11],kc,kt,lambdaN,lambdaD,lambdaJ))
                ss1=np.sum(np.real(eigv)>0)
                if np.isclose(ss1,0):
                        ss2=((ss[i][0],ss[i][1],ss[i][3],ss[i][4],ss[i][6],ss[i][7],ss[i][9],ss[i][10]))
                        ss3.append(ss2)
        ss4=unique_rows(ss3)
        return ss4

#calculating (N,D,J,I) at A and B [(NDJI) at B=(NDJI) at C] as a function of lambdaJ
lambdaJ_space=np.arange(0.001,3.5,0.01)
u=[]
for lambdaJ in lambdaJ_space:
        u.append(stable_I(0.1,0.04,5.0,10.0,lambdaJ))
        print(lambdaJ)
x=lambdaJ_space
y=u
w = [[x[i]] * len(y[i]) for i in range(len(y))]
x_to_plot = [item for sublist in w for item in sublist]
y_to_plot = [item for sublist in y for item in sublist]
np.savetxt('NDJI_N5.0_D20.0.txt',np.column_stack((x_to_plot,y_to_plot)),fmt='%12.6lf')
