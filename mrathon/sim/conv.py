from typing import Self
import numpy as np
from scipy.linalg import pinv

from mrathon.data.models import FDLineModel

from ..fit.main import VFModel
from abc import ABC, abstractmethod

def completelap(n):
    return (n*np.eye(n)-1)/(n-1)


class Simulator(ABC):
    '''Simulator Objects must have a method to set dt, as it may change and a universal application is ideal.'''

    @abstractmethod
    def set_dt(self, dt, niter):
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

    def set_dt(self, dt, niter):
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

    def next(self, uprev):
        self.x = (self.x.T*self.alpha).T + uprev
        return sum(ci@xi for ci, xi in  zip(self.C, self.x))
    

class SimDevice(Simulator):

    def __init__(self, vfunc, nphases=3) -> None:

        self.vfunc = vfunc
        self.nphases = nphases


    def set_dt(self, dt, niter):
        '''
        Description:
            Set the timestep (dt) and update any dependent parameters
        '''

        self.dt = dt
        self.niter = niter

        
        self.V = np.zeros((niter, self.nphases), dtype=complex)
        self.INJ = np.zeros((niter, self.nphases), dtype=complex)

    def step(self, n):

        self.V[n] = self.vfunc(n, self.dt)


class SimLine(Simulator):

    def __init__(self, fdline: FDLineModel, nphases) -> None:
        self.fdline = fdline
        self.nphases = nphases

        # Convolution Models
        self.Yconv = Convolve(self.fdline.Yc)
        self.Hconv = ConvolveAC(self.fdline.H)

    def set_dt(self, dt, niter):

        # Integer delay based on dt
        self.tau = int(self.fdline.H.tau/dt)
        
        # Configure Convolution Models
        self.Yconv.set_dt(dt, niter)
        self.Hconv.set_dt(dt, niter)

        # Clear Reflected Current History
        self.Ir = np.zeros((niter, self.nphases), dtype=complex)

    def set_dual(self, dual):
        '''
        Description:
            Assigns the opposing line terminal simulation model for reflection current access.
        '''
        self.dual: Self = dual 

    def injection(self, n, v, vprev):
        '''
        Description:
            Returns the injection into the node at iteration n
        '''
        # Get far end current
        i = self.dual.Ir[n-self.tau-1]

        # Calculate injection current
        return self.Yconv.next(v, vprev) - self.Hconv.next(i)

class SimNode(Simulator):
    '''
    Sub-Simulation Object, administering the set of devices terminated at a given node. 
    Handles KCL and Admittance-Sourced History current
    '''

    def __init__(self,device, nphases=3) -> None:

        # Data Models
        self.lines: list[SimLine] = []
        
        self.nphases = nphases
        self.device = device

    def add_line(self, line):
        self.lines.append(line)

    def set_device(self, device: SimDevice):
        self.device = device

    def set_dt(self, dt, niter):
        '''
        Description:
            Set the timestep (dt) and update any dependent parameters
        '''

       # Finalize Parameters
        self.nlines    = len(self.lines)

        self.dt = dt
        self.niter = niter 

        # LAplacian is the number of Lines + 1 for device
        self.LAP = completelap(self.nlines+1)

        # Set dt for devices on node
        for line in self.lines:
            line.set_dt(dt, niter)

        # Set dt for device
        self.device.set_dt(dt, niter)


    def step(self, n):
        '''
        Description:
            Calculate the history current of the next iteration
        '''

        # Calculates Voltage
        self.device.step(n)
        v, vprev = self.device.V[n], self.device.V[n-1]

        # Calculate Injection of lines and the governing device
        Iinj = [
            *(line.injection(n, v, vprev) for line in self.lines),
            self.device.INJ[n]
        ]

        # Calculate reflected current
        Ir = self.LAP@np.vstack(Iinj) # (lines x phase)

        # Set the reflection values for each line 
        for line, refl in zip(self.lines, Ir[::-1]):
            line.Ir[n] = refl

class SimGraph(Simulator):
    '''
    Manages many nodes during the simulation, handling propagation and branch oversight.
    '''

    def __init__(self, nodes: list[SimNode]) -> None:

        # Data Models
        self.nodes = nodes
        self.num   = len(nodes)

        self.n = 0


    def set_dt(self, dt, niter):
        '''
        Description:
            Set the timestep (dt) and update any dependent parameters
        '''
        self.dt = dt 
        self.niter = niter

        for node in self.nodes:
            node.set_dt(dt, niter)

        
    def connect(self, nodeA: SimNode, nodeB: SimNode, line):

        lineA = SimLine(line, nphases=nodeA.nphases)
        lineB = SimLine(line, nphases=nodeB.nphases)

        # Bind Line Models
        lineA.set_dual(lineB)
        lineB.set_dual(lineA)

        # Assign to nodes
        nodeA.add_line(lineA)
        nodeB.add_line(lineB)

    def step(self):

        # Need an incidence map
        for node in self.nodes:
            node.step(self.n)

        # Advance
        self.n += 1

    def restart(self):

        # Clears arrays and resets counters etc.
        self.set_dt(self.dt, self.niter)

        self.n = 0 

