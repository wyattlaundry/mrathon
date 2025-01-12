

from scipy.linalg import sqrtm, expm
from numpy.linalg import inv
import numpy as np

class LineParameters:
    '''
    Description:
        Analytical alterinative to generate samples for vector fitting H and Yc.
        Must provide functions for Z and Y that evaluate a scalar frequnecy (radians)
    '''

    def __init__(self, ell, Z, Y, shape=(3, 3)) -> None:
        self.ell = ell
        self.Z = Z
        self.Y = Y
        self.shape = shape

    def shunt_and_series(self, w):
        return self.Z(w), self.Y(w)
    
    def gamma(self, w):
        Z, Y = self.shunt_and_series(w)
        return sqrtm(Z@Y)


    def H(self, w):
        '''
        Description:
            Evaluates analytical Propagation Function at a scalar frequnecy value.
        Returns:
            H(w) Matrix
        '''

        GAM = self.gamma(w)
        H = expm(-GAM*self.ell) 

        return H #* np.exp(-TAUMIN*1j*w) 

    def Yc(self, w):
        '''
        Description:
            Evaluates analytical Characteristic Admittance at a scalar frequnecy value.
        Returns:
            Yc(w) Matrix
        '''

        Z, Y = self.shunt_and_series(w)
        return  inv(Z)@sqrtm(Z@Y)
    
    def __sample(self, loglim, nsamp, func):

        s = 1j*np.logspace(*loglim, nsamp).reshape(-1, 1)
        samp = np.empty((nsamp, *self.shape), dtype=complex)

        for sampi, w in zip(samp, s/1j): 
            sampi[:] = func(w[0])

        return s, samp

    
    def sample_Yc(self, loglim=(-1, 2), nsamp=100):
        '''
        Description:
            Evaluates analytical Characteristic Admittance model over log interval
        Returns:
            Tuple -> (s-domain, func-samples) with nsamp values
        '''

        return self.__sample(loglim, nsamp, self.Yc)
    
    def sample_H(self, loglim=(-1, 2), nsamp=100):
        '''
        Description:
            Evaluates analytical Propagation model over log interval
        Returns:
            Tuple -> (s-domain, func-samples) with nsamp values
        '''
        return self.__sample(loglim, nsamp, self.H)