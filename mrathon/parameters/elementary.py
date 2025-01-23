from typing import Any

import numpy as np
from numpy import pi

from mrathon.parameters.geometry import LineGeometry

# Constants Used
mu0 = 4*pi*1e-7 # H/m
eps0 = 8.8541878188e-12 # F/m
L = mu0/(2*pi)



class ExternalImpedence:
    '''Carson Equations Impedences (Inductive Only!)'''

    def __init__(self, LG: LineGeometry) -> None:

        self.D = LG.D

        # Element wise recipricol (1/D)
        self.rD = np.reciprocal(self.D)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        ''' 
        Description:
            Returns the Per-Meter Impedence Using Bessel Functions
        '''

        # Scalar Freq argument
        w = args[0]

        # Mutual Impedence Calculation
        return 1j*w*L*np.log(self.rD)

class ShuntAdmittance:

    def __init__(self, LG: LineGeometry, G=0) -> None:
        
        # Diaganol element shunt conductance
        self.G = G*np.eye(3)

        # Calculate Capacitance from Line Geometry
        D, Dp = LG.D, LG.Dp
        Cinv = np.log(Dp/D)/(2*pi)

        self.C = eps0*np.linalg.inv(Cinv)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        
        w = args[0]

        return self.G + 1j*w*self.C


class InternalImpedence:


    mu = 1 *mu0
    eps = 1 * eps0

    def __init__(self, LG: LineGeometry, resistivity)-> None:
        
        # Resistivity and Conductivity
        self.rho = resistivity
        self.sig = 1/resistivity

        # Radius
        self.rade = LG.crad

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        ''' 
        Description:
            Returns the Per-Meter Impedence Using Bessel Functions
        '''

        w = args[0]

        # Inifinite Series of Admittance
        Y = 0
        for k in range(1, 2000):
            Y += self.Yk(w, k)

        Z = 1/Y 

        return Z

    def besselroot(self, k):
        ''' 
        Description:
            An approximate formula for the k-th bessel root
        '''
        return pi*(k - 1/4)
    
    def Yk(self, w, k):
        ''' 
        Description:
            Returns the k-th Resistance of skin effect series
        '''

        # Resistance and Inductance
        r = self.besselroot(k)**2/(4*pi*self.sig*self.rade**2)
        l = self.mu/(4*pi)
        
        # K-th Admittance
        return 1/(r+1j*w*l)
    

class Gamma:

    def __init__(self, Z, Y) -> None:
        self.Z = Z
        self.Y = Y
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        
        w = args[0]

        gamma = np.sqrt(self.Z(w)*self.Y(w))
        return gamma
    
class CharacteristicAdmittance:

    def __init__(self, Z, Y) -> None:
        self.GAM = Gamma(Z, Y)
        self.Z = Z

    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        
        w = args[0]

        return (1/self.Z(w))*self.GAM(w)
    
   
class TerminalAdmittance:

    def __init__(self, Rth, Lth) -> None:
        
        self.Rth = Rth
        self.Lth = Lth

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        
        w = args[0]

        R = self.Rth 
        L = self.Lth

        Zth = R + 1j*w*L

        return 1/Zth
    
    def __domain(self, fmin, fmax, nsamp):
        return 2*pi*np.linspace(fmin, fmax, nsamp).reshape(-1,1)

    
    def fitYth(self, fmin, fmax, nsamp, order, iterations):

        # Domain of Interest
        w = self.__domain(fmin, fmax, nsamp)
        self.w = w

        # Vector Fitting Model
        #model = VectorFitter(
        #    s = 1j*w, 
        #    f = self(w), 
        #    numpoles = order
        #)
        model = None

        # Iteration
        model.iterate(iterations)

        self.model = model
