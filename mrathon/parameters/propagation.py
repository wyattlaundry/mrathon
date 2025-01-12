from typing import Any
import numpy as np


class Propagation:

    def __init__(self, gamma, linelength) -> None:
        self.gamma = gamma
        self.ell   = linelength

    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        
        w = args[0]

        gam = self.gamma(w)
        L = self.ell

        return np.exp(-gam*L)
    
class Delay:

    # NOTE CRITICAL Minimum time delay aka MPS
    # While some frequnecies ARE fast, we want to ignore those that decay to
    # basically zero. So selecting fastest frequnecy from detectable frequnecies
    # MPS is the delay of largest freq + phase of H
    # Multiple Papers mention using only those with magnitude less than 100


    def __init__(self, H: Propagation, w, thresh=0.1) -> None:
        
        # Frequnecies
        self.w = w 

        # (PRE-MPS) Propagation Function samples
        self.H = H
        self.Hsamp = H(w)

        # Threshold for magnitude of response
        self.THRESH = thresh # 0.1#1e-6

        # Crit. Freq.
        self.OMEGA = self.omega()
        
        # True Delay
        self.tau = self.delay()

        

    def __call__(self, *args: Any, **kwds: Any) -> Any:

        w = args[0]
        
        # Shifted Prop. Function
        return self.H(w) * np.exp(1j*w*self.tau)

    def omega(self):
        '''
        Description:
            Find the nidex of the critical frequnecy that marks the threshold of detection.
        '''

        # Find index where magnitude drops below threshold
        detectable = np.abs(self.Hsamp)>self.THRESH
        OMEGA = self.w[np.argmin(detectable)]

        return OMEGA
        
    def delay(self):
        '''
        Description:
            Returns the time delay of the propagation function
        '''

        # Extract Parameters
        OMEGA = self.OMEGA
        ell = self.H.ell
        beta = self.H.gamma(OMEGA).imag

        # Propagation delay
        TAUMIN = beta*ell/OMEGA


        # Actual MPS
        tau = TAUMIN + np.angle(self.H(OMEGA))/OMEGA

        return tau[0]
    