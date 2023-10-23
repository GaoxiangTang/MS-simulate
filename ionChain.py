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
    '''
    omega, b=modes(N, ion_space, radial_freq)
    kz=2*np.pi/740e-9
    eta_k=kz*(constants.hbar/2/171/constants.atomic_mass/omega)**0.5
    return eta_k * b


class ionChain:
    '''
    simulate an 1d ion chain for ms-gate 
    '''

    def __init__(self, N, ion_space=5e-6, radial_freq=1e7, omega_d=0.925e7) -> None:
        self.N=N
        self.omega, self.b=modes(N, ion_space, radial_freq)
        self.eta=eta(N, ion_space, radial_freq)
        self.omega_d = omega_d
        self.delta = self.omega - omega_d

    def simulate(self, j1, j2, amp):
        pass

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

        A = np.zeros((self.N*2, Nseg))
        M = np.zeros((Nseg, Nseg))
        tseg=T/Nseg
        Omega=1e9
        
        for m in range(self.N):
            for l in range(Nseg):
                # print(m, l)
                delta=self.delta[m]
                Aml = Omega*np.exp(1j*delta*l*tseg)*((1-(l+1)/Nseg)*T*(np.exp(1j*delta*tseg)-1)+(np.exp(1j*delta*tseg)-1)/1j/delta-tseg)/1j/delta
                A[m+self.N, l]=np.imag(Aml)
                A[m, l]=np.real(Aml)
        
        for i in range(Nseg):
            for j in range(i):
                for m in range(self.N):
                    delta=self.delta[m]
                    M[i, j]-=0.5*self.eta[m, j1]*self.eta[m, j2]*\
                    (Omega*(2*(np.cos(delta*tseg)-1)*np.sin(delta*tseg*(i-j)))/delta**2)
                M[j, i]=M[i, j]
            for m in range(self.N):
                delta=self.delta[m]
                M[i, i]-= 0.5*self.eta[m, j1]*self.eta[m, j2]*\
                (Omega*(2*np.sin(delta*tseg)/delta**2-2*tseg/delta))
        
        kernel=scipy.linalg.null_space(A)
        M_tilde=kernel.T @ M @ kernel
        eig, eigv=np.linalg.eig(M_tilde)
        ind=np.argmax(abs(eig))
        multiplier = theta/eig[ind]
        print(kernel.shape, eigv[ind].shape, multiplier)
        return kernel @ eigv[ind] * multiplier
    
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