from typing import Any

import numpy as np
from numpy import pi

class ShuntAdmittance:

    def __init__(self, G, C) -> None:
        
        self.G = G
        self.C = C

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        
        w = args[0]

        return self.G + 1j*w*self.C


class SeriesImpedance:

    # Permeability  and Permittivity
    mu0 = 1.256637e-6 # Netwonts per (Amp squared)
    eps0 = 8.854187e-12 # Farads per Meter

    mu = 1 *mu0
    eps = 1 * eps0

    def __init__(self, resistivity, radius_external, Lext) -> None:
        

        # Resistivity and Conductivity
        self.rho = resistivity
        self.sig = 1/resistivity

        # External Inductance
        self.Lext = Lext

        # Radius
        self.rade = radius_external

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

        Z = 1/Y + 1j*w*self.Lext

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
