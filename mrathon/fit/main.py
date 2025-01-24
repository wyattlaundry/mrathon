
from typing import Any
import numpy as np
from scipy.linalg import block_diag, pinv
from numpy.linalg import eigvals

from ..data.models import VFModel

class VFFactory:
    '''
    Given data, an instance of this class will allow the user to fit data using Vector Fitting.
    This object can then be passed to VectorFit.IO to save the model, which can be loaded
    and used in a simulation.
    '''

    def __init__(self, s: np.ndarray, finput: np.ndarray, numpoles=5, tau=0, include_const=True) -> None:

        dim = len(finput.shape)

        self.ismatrix = dim==3

        # Determines if the +d is in model
        self.include_const = include_const

        # Vector Detected 
        if dim == 2:
            self.nomShape = (shp[1])
            f = finput 

        # Matricies Detected (Flatten to vector)
        elif dim == 3:
            shp = finput.shape
            self.nomShape = (shp[1], shp[2])
            f   = finput.reshape((shp[0], shp[1]*shp[2]), order='F') # Stack column vectors of matrix ontop eachother

        else:
            raise Exception


        # Function Information
        self.s   = s        # Vector of Domain Values
        self.f   = f        # Function Samples Over Domain 
        self.tau = tau     # Time Domain Delay Causaility

        self.fdim     = f.shape[1] # Dimension of Vector Function
        self.numsamp  = s.shape[0] # Number of samples in Function
        self.numpoles = numpoles # Number of Poles

        # Utility Vectors
        #self.ONE_G = np.ones((self.numsamp*self.fdim, self.fdim))  # Used in A_matrix 
        #self.ONE_G = np.ones((self.numsamp*self.fdim,0))  # Used in A_matrix 
        self.ONE_G = np.repeat(np.eye(self.fdim), self.numsamp, axis=0)#.reshape(-1, 1, order='F')
        
        self.ONE_Q = np.ones((self.numpoles,1)) # Used In newpoles function
        self.fD    = f.flatten('F').reshape(-1, 1)

        # Dummy Variables without Iterations
        
        self.d = np.zeros((self.fdim,1))
        #self.d = 0 
        self.R = np.ones((self.numpoles*self.fdim, 1)) # Offset and Numer Res
        self.Rh = np.ones((self.numpoles, 1)) # Denom Res (Averaged after LS solved)
 
        # Initialiation Routines
        self.__init__poles()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        '''Evaluated Fitted Model on any domain, not just samples'''

        # Domain to evaluated passed as argument
        s   = args[0]
        PSI, R = self.psi(s), self.R.reshape((-1, self.fdim), order='F')

        # Evaluate Function NOTE we do not need denominator if converged, as it should be 1.
        d = self.d if self.include_const else 0
        fhat  = PSI@R + d

        # Check if it is a matrix and reshape
        if self.ismatrix:
            fhat = fhat.reshape((-1, *self.nomShape), order='F')
        
        return fhat

    def __init__poles(self):
        '''
        Description:
            Creates the Initial Poles Vector. Can be overwritten after object creation.
            The poles are evenly spaced on a log domain based on min and max 
            values of the domain passed during initialization.
        '''

        # Linearly Space Poles
        smin = np.abs(self.s.min())
        smax = np.abs(self.s.max())
        beta = np.linspace(smin, smax, self.numpoles)

        # Initial Poles Evenly Spaced On Log Scale
        self.poles = (-0.01-1j)*beta.reshape(-1,1)

    def __diag_poles(self):
        '''
        Description:
            Helper Function Returning Diagonal Version of Poles
        '''
        return np.diag(self.poles.T[0])
    
    def __H(seld, A):
        '''
        Description:
            Helper Function For Hermitian Operation
        '''
        return A.T.conj()
    
    def __psuedo_inv(self, A):
        '''Convience Function for Pusedo Inverse'''
        return pinv(A)
        #return np.linalg.inv(self.__H(A)@A)@self.__H(A)

    def psi(self, s=None):
        '''
        Description:
            This is the non-linear portion of the system
            that is computed so that residues act linearly.
        Returns:
            The Recipricol Matrix based on the poles in the model.
            Dimension: (nsamp x npoles)
        '''
        s = self.s if s is None else s
        return np.reciprocal(s-self.poles.T)
    
    def linear_system_matrix(self):
        '''
        Description:
            The matrix that describes the linear
            system of residues, in regards
            to the rational function residue parameters
        Returns:
            Matrix of system
        '''

        # Recipricol Matrix and Diagonal Function Matrix (Summed over dimensions to get average, for denominator iteration)
        PSI, f = self.psi(), self.f

        # Numerator Residues
        PSI_DIM = block_diag(*(PSI for i in f.T))
  
        # Denominator REsidues
        A_DIM = np.vstack([np.diag(v)@PSI for v in f.T])

        # Total Matrix
        if self.include_const:
            return np.hstack([-A_DIM , PSI_DIM, self.ONE_G])
        else:
            return np.hstack([-A_DIM , PSI_DIM])
    
    def update_residues(self):
        '''
        Description: 
            With a set of known poles, this function 
            will determine the numerator and denominator 
            residues that best approximate the function f(s)
        Parameters:
            poles: Vector of Poles
        Returns:
            a, b, d, e: Rational Function Parameters
        '''

        # System Matrix and  Psuedo Inverse
        A  = self.linear_system_matrix()
        Ai = self.__psuedo_inv(A)

        # Solve for Residues for Given poles
        xhat = Ai@self.fD

        # Extract Numer & Denominator 
        Rh = xhat[:self.numpoles]

        if self.include_const:
            R  = xhat[-self.numpoles*self.fdim-self.fdim:-self.fdim] # Pole Residues
            d  = xhat[-self.fdim:].T        # Constant Numerator Offset
            self.Rh, self.R, self.d = Rh, R, d
        else:
            R  = xhat[-self.numpoles*self.fdim:] # Pole Residues
            self.Rh, self.R, self.d = Rh, R, 0

    def denom_roots(self):
        '''
        Description:
            Finds roots of denominator. (The poles of function)
        Returns:
            vector
        '''
         # Poles & Denominator Residues
        Q, ones = self.__diag_poles(),  self.ONE_Q

        # Eigenvalues of this matrix are the zeros of denom
        return eigvals(Q - ones@self.Rh.T).reshape(-1,1)

    def pole_validation(self):
        ''' 
        Description:
            Preserves causality in poles
        '''

        LHS = self.poles.real>0
        self.poles[LHS] = -self.poles[LHS].conj()


    def iterate(self, niter=5):
        ''' 
        Description:
            Improves model with niter iterations, updating poles and residues
        '''

        for i in range(niter):

            # Update Residues Based on Poles
            self.update_residues()

            # Get Candidate Poles
            self.poles = self.denom_roots()

            # Validation of Causiality
            self.pole_validation()
            
        
        # Final Stage Residue Determination
        self.update_residues()

        # Return Model Object instead of Fitting Object
        return self.asmodel()


    def residues(self):
        ''' 
        Description:
            Returns residues of the model's poles. If input was a matrix,
            matrix residues will be returned
        Returns:
            K x N x M matrix, K residues, NxM input
        '''
        return self.R.reshape((-1, *self.nomShape), order='F')
    
    def offset(self):
        ''' 
        Description:
            Returns offset (d) of the model
        Returns:
            N x M matrix,  NxM input
        '''
        if self.include_const:
            return self.d.reshape(self.nomShape, order='F')
        else:
            return np.zeros(self.nomShape)

    
    def asmodel(self):

        order    = self.numpoles
        #d        = self.d[0].item() if self.include_const else 0
        d        = self.offset()
        q        = self.poles[:,0]
        residues = self.residues()
        tau      = self.tau

        return VFModel(order, d, q, residues, tau)




