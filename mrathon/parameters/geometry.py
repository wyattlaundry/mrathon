
from matplotlib.patches import Circle
import numpy as np
from numpy import pi
from scipy.stats import gmean
from matplotlib.axes import Axes


class LineGeometry:
    '''
    Line Geometry objects holds and calculates line spacings from coordinates and conductor radius.
    This can be passed to other MRATHON classes.
    '''

    def __init__(self, phase_coords, cond_coords=None, crad=0.01, transposed: bool=False) -> None:
        '''
        Parameters:
        - phase_coords: (nphases x 2) 
            A matrix of phase coordinates with X,Y as column in meters.
        - cond_coords: (ncond x 2) 
            A matrix of relative coordinates of conductors w.r.t. phase.
            This assumes that each bundle has identical geometry
        - crad: real float
            Conductor radius. Only supports 1 conductor radius size for now
        - transposed: bool
            Flag to indicate if phases are transposed over distance
        '''

        
        # If no conductor coordinates (cond_coords) passed, we assume there is 1 cable located directly at the phase coordinates
        if cond_coords is None:
            cond_coords = np.array([
                [0, 0], # Cable 1
            ])

        # Set Parameters
        self.phase_coords = phase_coords # Corrdinates x and y of phases (can have multiple cables at each phase)
        self.cond_coords  = cond_coords  # Relative coordinates of conductors w.r.t. phase
        self.crad         = crad         # Conductor Radii
        self.transposed   = transposed   # Transpose Flag

        ''' Phase Calculations '''

        # Relative Phase Distances and Image
        D  = self.relativedist(phase_coords)
        Dp = self.relativedist(phase_coords, image=True)

        ''' Per-Phase Conductor Calculations'''

        # Bundle GMR and Height GMR
        self.gmrad = self.GMR(self.cond_coords, crad)
        self.gmheight = gmean(Dp.diagonal())

        ''' Transposition Calculations'''

        # Set all to GMD if transposed
        if transposed:

            # Set GMD of distances and image
            D[:]  = self.GMD(D)
            Dp[:] = self.GMD(Dp)

            # Set GMR Heights (Roughly corresponsds to setting GMR of normal D matrix)
            np.fill_diagonal(Dp, self.gmheight)
            
        # Set bundle GMR as diagonal
        np.fill_diagonal(D, self.gmrad)

        # Set property
        self._D  = D
        self._Dp = Dp
    
    @property
    def D(self):
        return self._D
    
    @property
    def Dp(self):
        '''The Image Distance'''
        return self._Dp
    
    def relativedist(self, XY, image=False):
        '''
        Description:
            Calculates the relative distances from each coordinate
        '''
        x = XY[:,[0]]
        y = XY[:,[1]]

        dx = x.T - x
        dy = y.T + y if image else y.T - y

        return np.sqrt(dx**2 + dy**2)
    
    def GMD(self, D):
        '''
        Description:
            Given a distance matrix D, calculates GMD 
        '''
        idx = np.tril_indices_from(D, -1)
    
        return gmean(D[idx])


    def GMR(self, cond_coords, crad):
        '''
        Description:
            Given phase relative coordinates and conductor radius,
            this will calculate the GMR of the bundle
        '''

        D = self.relativedist(cond_coords)

        # Get all relative distances
        idx = np.tril_indices_from(D, -1)

        # Geometric Mean Radius
        gmr = gmean([crad, *D[idx]])

        return gmr

def plotgeom(ax: Axes, LG: LineGeometry, image=False, lims=True):

    # Labels
    ax.set_title('Line Geometry')
    ax.set_xlabel('X Coordinate (m)')
    ax.set_ylabel('Y Coordinate (m)')

    # Phase relative conductor coords
    cx, cy = LG.cond_coords.T.copy() 
    phasex, phasey = LG.phase_coords.T.copy()

    # Find  mean of conductor coordinates to get 'center' of bundle
    bx, by = np.mean(cx), np.mean(cy) 

    # Conjguate if Image
    if image: 
        cy     *= -1
        phasey *= -1

    # Plot each phase
    for px, py in zip(phasex, phasey):

        # Conductor Points
        ax.scatter(px + cx, py + cy, zorder=2)

        # Plot Circle indicating GMR
        ax.add_patch(
            Circle((px + bx, py + by), LG.gmrad, color='b', fill=False, ls='--')
        )

    # Scope
    if lims:
        hmax = np.max(cy)+ np.max(py)
        hmin = np.min(cy)+ np.min(py)
        pad = (hmax-hmin)*0.5
        if pad==0: pad = 1
        ax.set_ylim(hmin - pad, hmax + pad)

    # Scale Appropriatly
    ax.set_aspect('equal', adjustable='box')