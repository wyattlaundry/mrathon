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
            self.set_dt(dt, None)
        
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

    def __init__(self, vfunc, rth=1e-4, nphases=3) -> None:

        self.vfunc = vfunc
        self.nphases = nphases
        self.G = 1/rth

    def set_dt(self, dt, niter):
        '''
        Description:
            Set the timestep (dt) and update any dependent parameters
        '''

        self.dt = dt
        self.niter = niter

        
        self.V = np.zeros((niter, self.nphases), dtype=complex)
        self.INJ = np.zeros((niter, self.nphases), dtype=complex)

        # 'Into' gen/device
        self.Ir = np.zeros((niter, 3), dtype=complex)

    def injection(self, n):
        '''
        Description:
            Calls user-defined function to determine this iteration's nodal voltage and current injection
        '''

        Vint = self.vfunc(n, self.dt).astype(complex)

        # In steady state equivilent to G*delta V
        return self.G*Vint

class SimLine(Simulator):
    '''
    Numerical Implementation of F.D. Line Model. 
    Two instanaces are formed of this object and are linked 
    to exchange propagation history. Each dual instance is then assigned
    to a seperate SimNode instance.
    '''

    def __init__(self, fdline: FDLineModel, nphases) -> None:
        self.fdline  = fdline
        self.nphases = nphases

        # AC Convolve Used because 'G' term used explicitly in main voltage solve
        self.Yconvs = [
            ConvolveAC(mode)
            for mode in self.fdline.Yc
        ]

        # NOTE the -2H might come from the AC convolution variation
        # similar to how G is used in Yc conv
        self.Hconvs = [
            Convolve(mode)
            for mode in self.fdline.H
        ]

    def set_dt(self, dt, niter):

        # Integer delay based on dt
        self.inttaus = [
           int(mode.tau/dt)
            for mode in self.fdline.H
        ]

        # Fractional non-integer component of delay (float between 0 and 1)
        self.epsilons = [
            (mode.tau%dt)/dt
            for mode in self.fdline.H
        ]
        
        # Configure Convolution Models for each mode
        for mode in self.Yconvs:
            mode.set_dt(dt, niter)
        for mode in self.Hconvs:
            mode.set_dt(dt, niter)

        # Clear Reflected Current History
        self.Ir = np.zeros((niter, self.nphases), dtype=complex)

        # Store iteration propagated incident and shunt current
        self.IINCIDENT  = np.zeros( self.nphases, dtype=complex)
        self.ISHUNT = np.zeros( self.nphases, dtype=complex)

    def set_dual(self, dual):
        '''
        Description:
            Assigns the opposing line terminal simulation model for reflection current access.
        '''
        self.dual: Self = dual 

    def historical(self, n, vprev):
        '''
        Description:
            Returns the injection into the node at iteration (n)
            Calculating (Ishunt - Ifar)
        '''
        
        # Reset far current calculation
        self.IINCIDENT[:]  = 0
        self.ISHUNT[:]     = 0

        # Propagated Far Current Modes
        for Hmode, tau, eps in zip(self.Hconvs, self.inttaus, self.epsilons):

            # Get values for interpolation
            im0 = self.dual.Ir[n-tau] 
            im1 = self.dual.Ir[n-tau-1]
            im2 = self.dual.Ir[n-tau-2]
            
            # Interpolated values
            i     = im0 - (im0 - im1)*eps
            iprev = im1 - (im1 - im2)*eps

            # Add contribution to incidenct using interpolated
            self.IINCIDENT  += Hmode.next(i, iprev)

        # Shunt current modes (typically just 1)
        for Ymode in self.Yconvs:

            # Add contribution to shunt current
            self.ISHUNT  += Ymode.next(vprev)
        
        return  self.IINCIDENT - self.ISHUNT 

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

        # Save Args
        self.dt = dt
        self.niter = niter 

        # Finalize Parameters
        self.nlines = len(self.lines)

         # Voltage values storage
        self.V = np.zeros((niter, 3), dtype=complex)
        
        # Set dt for device
        self.device.set_dt(dt, niter)

        # Set dt for devices on node and Construct G Matrix
        I = np.eye(3, dtype=complex)
        G = I*self.device.G

        # Net Conductance of Lines
        for line in self.lines:

            # Configure
            line.set_dt(dt, niter)

            # Sum Each mode conductance for the line
            for mode in line.Yconvs:
                G += mode.G 

        # Constructing KCL
        #O = np.ones((1,self.nlines))
        #CONV = bmat([
        #    [O   , None, None , -G[0]],
        #    [None, O   , None , -G[1]],
        #    [None, None, O    , -G[2]],
        #]).toarray()
        #self.CONV = pinv(CONV)

        # This is the reduced form of the above
        self.N = self.nlines # number of 'reflected' current branches
        self.G = G

        # NOTE sign?
        self.Gchrg = np.linalg.inv(G.T@G + self.N*I)@G.T

    def step(self, n):
        '''
        Description:
            Calculate the history current of the next iteration
        '''

        # Calculate Injection of lines and the governing device
        HIST = self.device.injection(n)
        for line in self.lines:
            HIST += line.historical(n, self.V[n-1])

        # Solve for node V at this time step
        self.V[n] = self.Gchrg@HIST

        # Solve the net injection out of the node into line
        Iinj = (HIST + self.G@self.V[n])/self.N

        # Calculate reflected current for each line
        for line in self.lines:
            line.Ir[n] = Iinj - line.IINCIDENT
        

class SimGraph(Simulator):
    '''
    Manages network nodes during the simulation, handling iteration control and network connections.
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
        '''
        Description:
            Terminates F.D. line model between two provided nodes for simulation.
        '''

        lineA = SimLine(line, nphases=nodeA.nphases)
        lineB = SimLine(line, nphases=nodeB.nphases)

        # Bind Line Models
        lineA.set_dual(lineB)
        lineB.set_dual(lineA)

        # Assign to nodes
        nodeA.add_line(lineA)
        nodeB.add_line(lineB)

    def step(self):
        '''
        Description:
            Execute simulation timestep for all graph nodes
        '''

        # Need an incidence map
        for node in self.nodes:
            node.step(self.n)

        # Advance
        self.n += 1

    def restart(self):
        '''
        Description:
            Clears simulation history to prepare for new simulation of model.
        '''

        # Clears arrays and resets counters etc.
        self.set_dt(self.dt, self.niter)

        self.n = 0 

