import numpy as np

from ..fit.main import VFModel

class Convolve:
    '''A convenience class that hold state variables and computes recursive convolution parameters'''

    def __init__(self, model: VFModel, dt: float) -> None:

        # VF Source Model
        self.model = model

        # Get all recursive funcs
        self.set_timestep(dt)
        
        # State Var Vector (ndim x npoles)
        shape = (model.residues.shape[-1], model.order)
        self.x = np.zeros(shape, dtype=complex).T

    def set_timestep(self, dt):
        '''
        Description:
            Set the timestep (dt) and update any dependent parameters
        '''

        # Save Time Step
        self.dt = dt 

        # Extract
        R, d = self.model.residues, self.model.d
        q = self.model.poles

        # Integration Prameters
        self.alpha = np.exp(q*dt)
        self.mu = -(1/q)*(1 + (1-self.alpha)/(q*dt))
        self.lam = (1/q)*(self.alpha + (1-self.alpha)/(q*dt))

        # Modified parameters
        weight =  self.alpha*self.mu + self.lam # Vector of weights for eahc pole
        self.C =  [w*r for w, r in  zip(weight , R)] # Weigh each matrix residue
        self.G = d + sum(m*r for m, r in  zip(self.mu, R)) # Weight each matrix residue

    # Sum poles and add the DC state
    def next(self, uk, uprev):
        '''
        Description:
            Calculates the next value of output via recursive convolution.
        Parameters:
            - uk: The current iteration of the input function
            - uprev: The previous iteration of the input function
        '''
        self.x = (self.x.T*self.alpha).T + uprev
        return sum(ci@xi for ci, xi in  zip(self.C, self.x))+ self.G@uk 
    
class ConvolveAC(Convolve):

    def next(self, uk, uprev):
        self.x = (self.x.T*self.alpha).T + uprev
        return sum(ci@xi for ci, xi in  zip(self.C, self.x))