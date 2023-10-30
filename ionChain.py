import numpy as np
from scipy.optimize import minimize, minimize_scalar
import matplotlib.pyplot as plt
from scipy.misc import derivative
import numdifftools as nd
from scipy import constants
from scipy.integrate import quad, dblquad
import scipy
from numpy import exp, sin
import sympy as sp
import qutip
from qutip import tensor, sigmax, sigmay, sigmaz, qeye, basis, create, destroy;
from scipy.linalg import expm
from functools import reduce

def Energy_x(N, u, x, p):
    E = (x**2).sum()
    for i in range(N):
        for j in range(i):
            E+=1/p/((u[i]-u[j])**2 + (x[i]-x[j])**2)**0.5
    return E

def modes(N, d, omega_z):
    '''
    omega_z: radial trap frequency [2pi MHz]
    d: axial distance between adjacent ions [m]
    returns the mode frequencies and mode vectors.
    '''
    u=np.arange(N)

    K = constants.e**2/constants.epsilon_0/4/np.pi 
    p = 0.5*171*constants.atomic_mass*omega_z**2*d**3/K # nondimensionalize
    E = lambda x: Energy_x(N, u, x, p)
    # x=minimize(E, np.ones(N)*0.1, tol=1e-12).x
    x=np.zeros(N)
    U = nd.Hessian(E)
    eig, b = np.linalg.eigh(U(x))
    return (eig/2)**0.5*omega_z, b

def eta(N, ion_space, radial_freq):
    '''
    calculate lamb-dicke parameter.
    eta[j, k] is the l-d parameter of j-th ion and k-th mode
    '''
    omega, b=modes(N, ion_space, radial_freq)
    kz=2*np.pi/740e-9
    eta_k=kz*(constants.hbar/2/171/constants.atomic_mass/omega)**0.5
    return eta_k * b.T

def amp_of_t(amp, Nseg, T):
    return lambda t: amp[int(max(0, min(t//(T/Nseg), Nseg-1)))]

def amp_plot(amp, Nseg, T):
    tlist=np.linspace(0, T, 1000)
    amp=amp_of_t(amp, Nseg, T)
    plt.plot(tlist, [amp(t)for t in tlist])

def ideal_ms(theta):
    return (1j*theta*tensor([sigmax(), qeye(2)]) * tensor([qeye(2), sigmax()])).expm()

        
class ionChain:
    '''
    simulate an 1d ion chain for ms-gate 
    '''

    def __init__(self, N, ion_space=5e-6, radial_freq=1e7, omega_d=0.925e7, cut_off=10) -> None:
        self.N=N
        self.omega, self.b=modes(N, ion_space, radial_freq)
        self.eta=eta(N, ion_space, radial_freq)
        self.omega_d = omega_d
        self.delta = self.omega - omega_d
        self.phonon_cut_off = cut_off
        self.sim_init()
    
    def sim_init(self):
        '''
        initialize operators that will be used in gate simulation
        '''
        cut_off=self.phonon_cut_off

        self.sigmax1 = tensor(tensor(sigmax(), qeye(2)), tensor([qeye(cut_off)]*self.N))
        self.sigmax2 = tensor(tensor(qeye(2), sigmax()), tensor([qeye(cut_off)]*self.N))
        self.sigmaz1 = tensor(tensor(sigmaz(), qeye(2)), tensor([qeye(cut_off)]*self.N))
        self.sigmaz2 = tensor(tensor(qeye(2), sigmaz()), tensor([qeye(cut_off)]*self.N))
        spin_identity = tensor(qeye(2), qeye(2))
        a, a_dagger=[], []
        for k in range(self.N):
            a.append(tensor(spin_identity, tensor([qeye(cut_off)]*k+[destroy(cut_off)]+[qeye(cut_off)]*(self.N-k-1))))
            a_dagger.append(tensor(spin_identity, tensor([qeye(cut_off)]*k+[create(cut_off)]+[qeye(cut_off)]*(self.N-k-1))))
        self.a=a
        self.a_dagger=a_dagger

    def simulate_interaction(self, j1, j2, amp, T, Nseg, ad=0, sd=0, psi0=None):
        if psi0==None:
            spin=tensor([basis(2, 0), basis(2, 0)])
            phonon=tensor([basis(self.phonon_cut_off, 0)]*self.N)
            psi0=tensor(spin, phonon)

        H0 = reduce(lambda x,y: x+y, [-self.delta[k]*self.a_dagger[k]*self.a[k] for k in range(self.N)])
        H_int1 = reduce(lambda x,y: x+y,
            [0.5*(self.eta[j1, k]*self.sigmax1+self.eta[j2, k]*self.sigmax2)*(self.a[k]+self.a_dagger[k]) for k in range(self.N)])
        H_int2 = -H0 + ad/2*(self.sigmaz1+self.sigmaz2)
        tseg=T/Nseg
        psi=psi0
        for l in range(Nseg):
            psi = (-1j*tseg*(H_int1*amp[l]+H_int2)).expm()*psi

        # return psi
        return (1j*T*H0).expm()*psi
    
    def simulate_schrodinger(self, j1, j2, amp, T, Nseg, ad=0, sd=0, psi0=None):
        if psi0==None:
            spin=tensor([basis(2, 0), basis(2, 0)])
            phonon=tensor([basis(self.phonon_cut_off, 0)]*self.N)
            psi0=tensor(spin, phonon)
        amp=amp_of_t(amp, Nseg, T)
        H = []
        for k in range(self.N):
            H.append([(self.sigmax1*self.eta[j1, k] + self.sigmax2*self.eta[j2, k])*self.a_dagger[k],
                       lambda t, args: 0.5*amp(t)*np.exp(1j*self.delta[k]*t)])
            H.append([(self.sigmax1*self.eta[j1, k] + self.sigmax2*self.eta[j2, k])*self.a[k],
                       lambda t, args: 0.5*amp(t)*np.exp(-1j*self.delta[k]*t)])
        H.append([(self.sigmaz1+self.sigmaz2), lambda t, args: ad/2])
        return qutip.sesolve(H, psi0, np.linspace(0, T, 1000), )
            

    def optimize_amp(self, j1, j2, theta, T, Nseg, pulse_symmetry):
        '''
        j1, j2: ions with laser shined on
        T: gatetime
        Nseg: number of segments
        '''

        # this ensures the kernel of A exists.
        if pulse_symmetry:
            assert ((Nseg + 1) // 2) > 2*self.N
        else:
            assert Nseg > 2*self.N

        # define Omega(t)=Omega*f(t), to avoid precision error
        Omega=1e6

        A = np.zeros((self.N*2, Nseg))
        M = np.zeros((Nseg, Nseg))
        tseg=T/Nseg
        
        for m in range(self.N):
            for l in range(Nseg):
                # print(m, l)
                delta=self.delta[m]
                Aml = Omega*(np.exp(1j*delta*l*tseg)*((1-(l+1)/Nseg)*T*(np.exp(1j*delta*tseg)-1)+(np.exp(1j*delta*tseg)-1)/1j/delta-tseg)/1j/delta)
                A[m+self.N, l]=np.imag(Aml)
                A[m, l]=np.real(Aml)
        
        for i in range(Nseg):
            for j in range(i):
                for m in range(self.N):
                    delta=self.delta[m]
                    M[i, j]-=0.5*Omega**2*self.eta[j1, m]*self.eta[j2, m]*\
                    (((np.cos(delta*tseg)-1)*np.sin(delta*tseg*(i-j)))/delta**2)
                M[j, i]=M[i, j]
            for m in range(self.N):
                delta=self.delta[m]
                M[i, i]-= 0.5*Omega**2*self.eta[j1, m]*self.eta[j2, m]*\
                ((np.sin(delta*tseg)/delta**2-tseg/delta))

        if pulse_symmetry:
            for i in range(Nseg//2):
                A[:, i]+=A[:, Nseg-i-1]
                M[i]+=M[Nseg-i-1]
                M[:, i]+=M[:, Nseg-i-1]
            size = (Nseg+1)//2 
            A = A[:, :size]
            M = M[:size, :size]
        
        kernel=scipy.linalg.null_space(A)
        M_tilde=kernel.T @ M @ kernel
        eig, eigv=np.linalg.eig(M_tilde)
        if theta>0:
            ind=np.argmax(eig)
        else:
            ind=np.argmax(-eig)
        multiplier = (theta/eig[ind])**0.5

        pulse = kernel @ eigv[ind] * multiplier * Omega
        if not pulse_symmetry:
            return pulse
        if Nseg % 2 == 1:
            pulse=np.concatenate((pulse, pulse[-2::-1]))
        else:
            pulse=np.concatenate((pulse, pulse[::-1]))
        return pulse

    
    def optimize_amp_sympy(self, j1, j2, T, Nseg):
        '''
        this function proves the result of optimize_amp is correct.
        it is outdated for its low effciency.

        j1, j2: ions with laser shined on
        T: gatetime
        Nseg: number of segments
        '''
        def f_l(tt, ll):
            return sp.Piecewise((1, ((tt>tseg*ll)&(tt<=tseg*(ll+1)))), (0, True))
        
        t, t1, t2=sp.symbols("t, t_1, t_2", positive=True)
        tseg=T/Nseg
        A = np.zeros((self.N, Nseg), np.complex128)
        M = np.zeros((Nseg, Nseg))
        
        for m in range(self.N): 
            for l in range(Nseg):
                # print(m, l)
                intermediate = sp.integrate(f_l(t1, l)*sp.exp(1j*self.delta[m]*t1), (t1, 0, t),)
                A[m, l]=sp.integrate(intermediate, (t, 0, T))       
        
        for i in range(Nseg):
            for j in range(i+1):
                for m in range(self.N):  
                    print(i, j, m) 
                    expr=(f_l(t1, i)*f_l(t2,j)+f_l(t2, i)*f_l(t1, j))*sp.sin(self.delta[m]*(t2-t1))
                    intermediate=sp.integrate(expr, (t2, 0, t1))   
                    M[i, j]-=0.5*self.eta[m, j1]*self.eta[m, j2] * sp.integrate(intermediate, (t1, 0, T))
                M[j, i]=M[i, j]
        return A, M


if __name__ == '__main__':
    ic=ionChain(17)
    ic.optimize_amp(9, 10, 3e-4, 19)        