import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp

from qutip_extension import adjoint_superoperator

import time as time
import sys as sys

plt.rcParams.update({
    "text.latex.preamble": r"\usepackage{amsmath}",
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 8

})

#####################
# Global Parameters #
#####################

N = 80 # Clock accuracy
M = 2   # Maximal tick number -> dimension M+1
Q = 4   # Dimension of target space

# Time range
times = np.linspace(0,4,300)


########################
# Operators and states #
########################

def simulate(N,times,full=True):
    """
    Simulate the toy example for a Bell state
    Input
        N   accuracy
    Output
        rhoT    target state as a function of time (dim,dim,len(times))
        rhoC    clock register state (3,len(times))
    """
    Gamma = 1.0*N # Stochastic clock rate

    # Define instruction Hamiltonians
    HH = -(-np.pi / 2.0 * np.eye(2) + np.pi / 2.0 * 1 / np.sqrt(2) * np.array([[1,1],[1,-1]])) # Hadamard Hamiltonian
    HH0 = np.kron(HH,np.eye(2))
    HCNOT = np.pi*np.kron(np.array([[0,0],[0,1]]),1/2 *np.array([[1,-1],[-1,1]]))   # CNOT Hamiltonian

    LT_HH0 = -1j*adjoint_superoperator(HH0)
    LT_HCNOT = -1j*adjoint_superoperator(HCNOT)

    instructions = [LT_HH0,LT_HCNOT]

    # Tick projectors
    R0_list = []
    for m in range(M):
        R0_m = np.zeros((M+1))
        R0_m[m] = 1.0
        R0_list.append(np.kron(np.eye(N),np.diag(R0_m)))

    # Define interaction Hamiltonian
    LFull = np.kron(R0_list[0],instructions[0])
    LFull += np.kron(R0_list[1],instructions[1])

    # Clock internal jumps
    L_internal = np.zeros((N,N))
    for n in range(N-1):
        L_internal[n+1,n] = Gamma
        L_internal[n,n] = -Gamma

    L_internal = np.kron(L_internal,np.kron(np.eye(M+1),np.eye(Q**2)))

    # Clock tick
    L_tick_C = np.zeros((N,N))
    L_tick_Cd = np.zeros_like(L_tick_C)

    L_tick_R0 = np.zeros((M+1,M+1))
    L_tick_R0d = np.zeros_like(L_tick_R0)

    L_tick_C[0,-1] = Gamma
    L_tick_Cd[-1,-1] = -Gamma

    for m in range(M):
        L_tick_R0[m+1,m] = 1.0
        L_tick_R0d[m,m] = 1.0

    L_tick = np.kron(np.kron(L_tick_C,L_tick_R0),np.eye(Q**2))
    L_tickd = np.kron(np.kron(L_tick_Cd,L_tick_R0d),np.eye(Q**2))

    # LFull = np.kron(np.eye(N),LFull)

    LFull += L_tick
    LFull += L_tickd
    LFull += L_internal

    # Initial state
    C_init = np.zeros((N))
    C_init[0] = 1.0

    R0_init = np.zeros((M+1))
    R0_init[0] = 1.0

    T_init = np.zeros((4,4),dtype=complex)
    T_init[0,0] = 1.0

    rho_init = np.kron(C_init,np.kron(R0_init,T_init))

    #############
    # Evolution #
    #############

    # ODE
    def fun(t,y):
        return LFull.dot(y)

    # Integration
    if full:
        res = solve_ivp(fun,(min(times),max(times)),np.ravel(rho_init),t_eval=times)
    else:
        res = solve_ivp(fun,(min(times),max(times)),np.ravel(rho_init))
    
    len_t = res.y.shape[-1]

    rho_full = np.reshape(res.y,(N,(M+1),Q,Q,len_t))

    rhoT = np.sum(rho_full,axis=(0,1))

    rhoR = np.sum(np.trace(rho_full,axis1=2,axis2=3),axis=0)

    return rhoT, rhoR

# some states
psi_plus = np.kron(1/np.sqrt(2)*np.array([1,1]),np.array([1,0]))
psi_bell = 1/np.sqrt(2)*np.array([1.0,0,0,1.0])

def distance(psi_T,rho):
    proj = np.outer(psi_T,psi_T)
    return 1.-np.real_if_close(np.trace(proj.dot(rho)))

def avgN(X):
    return np.real_if_close(np.dot(np.linspace(0,1,M+1,endpoint=True),X))

dist_plus = lambda rho : distance(psi_plus,rho)
dist_bell = lambda rho : distance(psi_bell,rho)

rhoT, rhoR = simulate(N,times)

plt.figure(figsize=(3.375,2.0))
plt.plot(times,1-dist_plus(rhoT),label=r"$\langle +,0|\rho_T(t)|+,0\rangle$",color="orange")
plt.plot(times,1-dist_bell(rhoT),label=r"$\langle\Psi_+|\rho_T(t)|\Psi_+\rangle$",color="red")
plt.plot(times,avgN(rhoR),label=r"avg. ticks / 2",color="black",linestyle="--")
plt.xlabel(r"time $t=[\tau]$")
plt.ylabel(r"fidelity $\mathcal{F}=[1]$")
plt.ylim(-0.1,1.1)
plt.grid(linestyle='--')
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('evolution.jpg',dpi=600)
plt.show()

N_range = np.arange(2,230,10)
N_range_ex = np.arange(np.floor(min(N_range)/2),max(N_range)*2,20)
F_range = np.zeros_like(N_range,dtype=float)

for k, N in enumerate(N_range):
    print(N)
    rhoT, rhoR = simulate(N,times,full=False)
    F_range[k] = dist_bell(rhoT[:,:,-1])

fig, ax1 = plt.subplots(figsize=(3.375,3.375))
ax2 = ax1.twinx()

ax1.loglog(N_range,F_range,label=r"aQPU simulation",marker='x',color='red',linestyle='')
ax2.loglog(N_range_ex,N_range_ex*2,label=r"entropy lower bound",color='blue',linestyle='--')
ax1.loglog(N_range_ex,F_range[-1]*N_range[-1]/N_range_ex,label=r"linear slope $1-\mathcal{F}\propto N^{-1}$",color='orange',linestyle='--')

ax1.set_xlabel(r"master-clock accuracy $N=[1]$")
ax1.set_ylabel(r"infidelity $1-\mathcal{F}_{\mathcal{A}}=[1]$")
ax2.set_ylabel(r"entropy per tick $\Sigma_{\mathrm{cw},\tau}=[k_B]$")
ax1.set_xbound(min(N_range)/2,max(N_range)*2)
ax2.set_ybound(min(F_range)/3,max(N_range)*6)
ax1.set_ybound(min(F_range)/3,max(N_range)*6)

ax1.grid(linestyle='--')
ax2.legend(loc="upper left")
ax1.legend(loc="lower left")
plt.tight_layout()
plt.savefig('fidelity.jpg',dpi=600)
plt.show()