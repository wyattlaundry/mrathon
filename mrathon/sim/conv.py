import numpy as np
from scipy.linalg import pinv

from mrathon.data.models import DeviceModel

from ..fit.main import VFModel
from abc import ABC, abstractmethod

def completelap(n):
    return n*np.eye(n)-1


class Simulator(ABC):
    '''Simulator Objects must have a method to set dt, as it may change and a universal application is ideal.'''

    @abstractmethod
    def set_dt(self, dt):
        pass

class Convolve(Simulator):
    '''A convenience class that hold state variables and computes recursive convolution parameters'''

    def __init__(self, model: VFModel, dt: float = None) -> None:

        # VF Source Model
        self.model = model

        # Get all recursive funcs
        if dt is not None:
            self.set_dt(dt)
        
        # State Var Vector (ndim x npoles)
        shape = (model.residues.shape[-1], model.order)
        self.x = np.zeros(shape, dtype=complex).T

    def set_dt(self, dt):
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
    '''
    'AC' Convolution Variation - discludes the DC impulse
    '''

    def next(self, uk, uprev):
        self.x = (self.x.T*self.alpha).T + uprev
        return sum(ci@xi for ci, xi in  zip(self.C, self.x))
    

class SimNode(Simulator):
    '''
    Sub-Simulation Object, administering the set of devices terminated at a given node. 
    Handles KCL and Admittance-Sourced History current
    '''

    def __init__(self, models: list[DeviceModel], dt=None) -> None:

        # Data Models
        self.models = models 
        self.num    = len(models)

        # AC Convolution Models of Each & DC Impulse of Each
        self.convs  = [ConvolveAC(m.shuntmodel) for m in models]

        # (TOTAL NUMBER OF CONDUCTORS)
        self.ncond = sum([m.ncond for m in models])
        
        # Complete Graph Laplacian 
        self.LAP   = completelap(self.ncond)

        # Set if passed
        if dt is not None:
            self.set_dt(dt)

    def set_dt(self, dt):
        '''
        Description:
            Set the timestep (dt) and update any dependent parameters
        '''

        self.dt = dt

        # Set dt for devices on node
        for conv in self.convs:
            conv.set_dt(dt)

        # This depends on dt 
        self.G      = np.vstack([c.G for c in self.convs])

        # Inverse Conversion Matrix to solve for I and V
        self.LHSi = pinv(np.hstack([self.LAP, -self.G]))

    def calc_IV(self):
        '''
        Description:
            Calculate the history current of the next iteration
        '''

        ishnt = 0
        iinc  = 0
        LAP   = self.LAP
        CONV  = self.LHSi

        # History Current
        IHIST = ishnt - LAP@iinc

        # Solve for I inj for each line and V of each phase
        IV = CONV@IHIST 
        #I = IV[:-1]
        #V = IV[-1]


class SimGraph(Simulator):
    '''
    Manages many nodes during the simulation, handling propagation and branch oversight.
    '''

    def __init__(self, nodes: list[SimNode], dt=None) -> None:

        # Data Models
        self.nodes = nodes
        self.num   = len(nodes)

        # Set if passed
        if dt is not None:
            self.set_dt(dt)


    def set_dt(self, dt):
        '''
        Description:
            Set the timestep (dt) and update any dependent parameters
        '''

        for node in self.nodes:
            node.set_dt(dt)



