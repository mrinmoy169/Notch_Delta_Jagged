import sympy
import numpy as np
import scipy.integrate
from functools import partial
sympy.init_printing()
NA, NB, NC, DA, DB, DC, JA, JB, JC, IA, IB, IC = sympy.symbols("NA, NB, NC, DA, DB, DC, JA, JB, JC, IA, IB, IC")
kc, kt, gamma, gammaI, lambdaN, lambdaD, lambdaJ = sympy.symbols("kc, kt, gamma, gammaI, lambdaN, lambdaD, lambdaJ")

#parameters
kc=0.1; kt=0.04; gamma=0.1; gammaI=0.5; nJ=5

#ode
dNAdt = lambdaN*(1+IA**2/(1+IA**2))-NA*(kc*(DA+JA)+0.5*kt*(DB+DC+JB+JC))-gamma*NA
dNBdt = lambdaN*(1+IB**2/(1+IB**2))-NB*(kc*(DB+JB)+0.5*kt*(DC+DA+JC+JA))-gamma*NB
dNCdt = lambdaN*(1+IC**2/(1+IC**2))-NC*(kc*(DC+JC)+0.5*kt*(DA+DB+JA+JB))-gamma*NC
dDAdt = lambdaD/(1+IA**2)-DA*(kc*NA+0.5*kt*(NB+NC))-gamma*DA
dDBdt = lambdaD/(1+IB**2)-DB*(kc*NB+0.5*kt*(NC+NA))-gamma*DB
dDCdt = lambdaD/(1+IC**2)-DC*(kc*NC+0.5*kt*(NA+NB))-gamma*DC
dJAdt = lambdaJ*(1+IA**nJ/(1+IA**nJ))-JA*(kc*NA+0.5*kt*(NB+NC))-gamma*JA
dJBdt = lambdaJ*(1+IB**nJ/(1+IB**nJ))-JB*(kc*NB+0.5*kt*(NC+NA))-gamma*JB
dJCdt = lambdaJ*(1+IC**nJ/(1+IC**nJ))-JC*(kc*NC+0.5*kt*(NA+NB))-gamma*JC
dIAdt = 0.5*kt*NA*(DB+DC+JB+JC)-gammaI*IA
dIBdt = 0.5*kt*NB*(DC+DA+JC+JA)-gammaI*IB
dICdt = 0.5*kt*NC*(DA+DB+JA+JB)-gammaI*IC

sys = sympy.Matrix([dNAdt, dNBdt, dNCdt, dDAdt, dDBdt, dDCdt, dJAdt, dJBdt, dJCdt, dIAdt, dIBdt, dICdt])
var = sympy.Matrix([NA, NB, NC, DA, DB, DC, JA, JB, JC, IA, IB, IC])
jac = sys.jacobian(var)
#print(jac)

#convert jac to a function:
jacobian = sympy.lambdify((NA, NB, NC, DA, DB, DC, JA, JB, JC, IA, IB, IC, lambdaN, lambdaD, lambdaJ), jac, dummify=False)

#steady states
def eq(p,lambdaN,lambdaD,lambdaJ):
        NA,NB,NC,DA,DB,DC,JA,JB,JC,IA,IB,IC=p
        dNAdt = lambdaN*(1+IA**2/(1+IA**2))-NA*(kc*(DA+JA)+0.5*kt*(DB+DC+JB+JC))-gamma*NA
        dNBdt = lambdaN*(1+IB**2/(1+IB**2))-NB*(kc*(DB+JB)+0.5*kt*(DC+DA+JC+JA))-gamma*NB
        dNCdt = lambdaN*(1+IC**2/(1+IC**2))-NC*(kc*(DC+JC)+0.5*kt*(DA+DB+JA+JB))-gamma*NC
        dDAdt = lambdaD/(1+IA**2)-DA*(kc*NA+0.5*kt*(NB+NC))-gamma*DA
        dDBdt = lambdaD/(1+IB**2)-DB*(kc*NB+0.5*kt*(NC+NA))-gamma*DB
        dDCdt = lambdaD/(1+IC**2)-DC*(kc*NC+0.5*kt*(NA+NB))-gamma*DC
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

def find_unique_sol(lambdaN,lambdaD,lambdaJ):
        s=[]
        for i in np.arange(1,300,1):
                a=np.random.uniform(0,50)               #decide the range from the solutions of N
                b=np.random.uniform(0,50)
                c=np.random.uniform(0,50)
                d=np.random.uniform(0,50)
                inic=(np.random.uniform(0,50),np.random.uniform(0,50),a,np.random.uniform(0,50),np.random.uniform(0,50),b,np.random.uniform(0,50),np.random.uniform(0,50),c,np.random.uniform(0,50),np.random.uniform(0,50),d)
                sol=findroot(lambda x: eq(x,lambdaN,lambdaD,lambdaJ),inic)
                s.append(sol)
        s1=np.array(np.around(s,decimals=5))
        s2=unique_rows(s1)
        s3=[row for row in s2 if not np.isnan(row).any()]   # remove nan
        s4=[row for row in s3 if all(row[:][:] > 0)]   # remove negative sol
        s5=[row for row in s4 if np.isclose((row[:][1]-row[:][2]),0)]   #kepping those sol only where NB=NC
        return s5

#instability
def unstable_modes(lambdaN,lambdaD,lambdaJ):
        global nature   #otherwise nonlocal bound error
        ss = find_unique_sol(lambdaN,lambdaD,lambdaJ)
        ss4=[]
        for i in np.arange(0,len(ss),1):
                eigv = np.linalg.eigvals(jacobian(ss[i][0],ss[i][1],ss[i][2],ss[i][3],ss[i][4],ss[i][5],ss[i][6],ss[i][7],ss[i][8],ss[i][9],ss[i][10],ss[i][11],lambdaN,lambdaD,lambdaJ))
                ss1=np.around((ss[i][3]-ss[i][4])+ss[i][6]-ss[i][7],4) #calculating Delta(D+J)
                if ss1==-0.0:
                        ss1+=0.0
                ss2=np.sum(np.real(eigv)>0)
                if np.isclose(ss1,0) and np.isclose(ss2,0):
            #nature="Uniform, Stable"
                        nature=1
                if np.isclose(ss1,0) and np.isclose(ss2,1):
            #nature="Uniform, 1 Untable Mode"
                        nature=2
                if np.isclose(ss1,0) and np.isclose(ss2,2):
            #nature="Uniform, 2 Untable Modes"
                        nature=3
                if ss1>0 and np.isclose(ss2,0):
            #nature="Hexagon, Stable"
                        nature=4
                if ss1>0 and np.isclose(ss2,1):
            #nature="Hexagon, 1 Untable Mode"
                        nature=5
                if ss1>0 and np.isclose(ss2,2):
            #nature="Hexagon, 2 Untable Modes"
                        nature=6
                if ss1<0 and np.isclose(ss2,0):
            #nature="Antihexagon, Stable"
                        nature=7
                if ss1<0 and np.isclose(ss2,1):
            #nature="Antiexagon, 1 Untable Mode"
                        nature=8
                if ss1<0 and np.isclose(ss2,2):
            #nature="Antihexagon, 2 Untable Mode"
                        nature=9
                ss3=np.append(ss1,nature)
                ss4.append(ss3)
        ss5=unique_rows(ss4)
        return ss5


#bifurcation diagram
lambdaD_space=np.arange(1,4.01,0.01)
u=[]
for lambdaD in lambdaD_space:
        u.append(unstable_modes(1.822,lambdaD,0.1))
        print(lambdaD)
x=lambdaD_space
y=u
w = [[x[i]] * len(y[i]) for i in range(len(y))]
x_to_plot = [item for sublist in w for item in sublist]
y_to_plot = [item for sublist in y for item in sublist]
np.savetxt('N_1.822_D_J_0.1.txt',np.column_stack((x_to_plot,y_to_plot)),fmt='%12.6lf')
