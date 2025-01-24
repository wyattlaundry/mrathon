

from scipy.linalg import sqrtm, expm
from numpy.linalg import inv
import numpy as np

class PropMode:
    '''Data class to perform mode-level processing of data samples'''

    def __init__(self, basis, gams, ell, w) -> None:
        self.basis = basis 
        self.gams = gams
        self.ell = ell
        self.w = w.flatten() # Ensure flat

        self.tau = self.mindelay()

    def mindelay(self):
        '''
        Description:
            Calculates the minimum delay for this mode
        Returns:
            Tau: float
        '''

        # Beta and Frequnecys
        beta = self.gams.imag.T
        w    = self.w
        l    = self.ell

        # Find minimum phase delay over all freq
        return np.min(beta*l/w)
    
    def sample_H(self):
        '''
        Description:
            Samples this mode (H)
        Returns:
            Array of samples
        '''

        D = self.basis # Basis Matrix at each freq
        gams = self.gams
        ell = self.ell
        s = 1j*self.w
        tau = self.tau 

        # Time Shifted Propagation of Mode at each sample
        Hm = D.T*np.exp(s*tau-gams*ell)

        return Hm.T


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

    def gamma(self, w, Z=None, Y=None):
        '''
        Description:
            Evaluates analytical gamma at a scalar frequnecy value.
        Returns:
            Yc(w) Matrix
        '''
        Z = self.Z(w) if Z is None else Z
        Y = self.Y(w) if Y is None else Y
        return sqrtm(Z@Y)

    def Yc(self, w):
        '''
        Description:
            Evaluates analytical Characteristic Admittance at a scalar frequnecy value.
        Returns:
            Yc(w) Matrix
        '''

        Z, Y = self.Z(w), self.Y(w)
        GAM = self.gamma(w, Z, Y)

        return  inv(Z)@GAM
    
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
    
    def sample_gamma(self, loglim=(-1, 2), nsamp=100):
        '''
        Description:
            Evaluates analytical Gamma over log interval
        Returns:
            Tuple -> (s-domain, func-samples) with nsamp values
        '''
        return self.__sample(loglim, nsamp, self.gamma)
    
    def extractmodes(self, s, gamma_samples, ell, nummodes=3):    
        '''
        Description:
            Given gamma samples and respective, frequnecies 
            Decomposes Gamma into 3 eigenvalues and 3 basis matricies 
        Returns:
            (tuple) of PropMode object for each mode
        '''

        GAM = gamma_samples
        nsamp = GAM.shape[0]

        # Get eigenvalues of Gamma Matrix
        GAM_EIGS = np.zeros((nsamp, nummodes), dtype=complex)

        # Store Base Matrix Samples of each mode
        BaseMatricies = [
            np.empty_like(GAM) 
            for i in range(nummodes)
        ]

        # This is a lot of eigenvalue decomps
        for i in range(nsamp):

            # Eigendecomp of Gamma at this frequnecy
            eig, T = np.linalg.eig(GAM[i])
            Ti = np.linalg.inv(T) 

            # Sort the Eigenvalues and Eigenvectors
            srt = np.argsort(np.abs(eig))
            GAM_EIGS[i] = eig[srt]
            T, Ti = T[:,srt], Ti[srt] # Reorder T cols and Ti rows

            # Set the bases for each mode
            for mode in range(nummodes):
                u, v = T[:,[mode]], Ti[[mode]] # T col and Ti row
                BaseMatricies[mode][i] = u@v 

        # Mode Models
        w = s.imag
        Hmodes = [
            PropMode(BASE, EIG, ell, w)
            for BASE, EIG in zip(BaseMatricies, GAM_EIGS.T)
        ]

        # Return data model for each mode
        return Hmodes 