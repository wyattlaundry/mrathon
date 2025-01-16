import numpy as np

from ..io.main import JSONModel


from abc import ABC, abstractproperty

class VFModel(JSONModel):
    '''
    A data-structure to hold VF model parameters from a VF function model.
    '''

    def __init__(self, order, d, poles, residues, tau=0) -> None:
        
        # Model Order/Pole Count
        self.order = order 
        self.tau   = tau

        # VF Parameters: Offset, Array of Poles, Tensor of Residues
        self.d: complex = d 
        self.poles: np.ndarray = poles 
        self.residues = residues
    
    def __str__(self) -> str:
        txt = f'Poles (Count: {self.order}):\n'
        for q in self.poles:
            txt += f'{q:.3f}\n'
        return txt
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def laplace(self, s):
        '''
        Evaluates VF model in laplace domain, given input vector s (complex)
        '''

        result = np.zeros((len(s), self.residues[0].shape))

        for pole, R in zip(self.poles, self.residues):
            result += R / (s - pole)
      
        PSI, R = self.psi(s), self.R.reshape((-1, self.fdim), order='F')

        # Evaluate Function NOTE we do not need denominator if converged, as it should be 1.
        fhat  = PSI@R + self.G 

        # Check if it is a matrix and reshape
        if self.ismatrix:
            fhat = fhat.reshape((-1, *self.nomShape), order='F')
    
    def asdict(self):

        Rdim     = self.residues[0].shape

        dictdata = {
            "order"    : self.order ,
            "tau"      : self.tau   , # Time delay
            "residue-dim": [*Rdim],
            "d"     : {
                "real": self.d.real, 
                "imag": self.d.imag
            },
            
            "poles" : [
                {
                    "q": {"real": q.real.item(), "imag": q.imag.item()},
                    "r": {
                        "real": [row.real.tolist() for row in rmat], 
                        "imag": [row.imag.tolist() for row in rmat]
                    }
                } for rmat, q in zip(self.residues, self.poles)
            ]
        }

        return dictdata
    
    @classmethod
    def fromdict(cls, data: dict):

        # Helper
        tocmplx = lambda z: z['real'] + 1j*z['imag']

        # Meta Parameters
        order = data['order']
        Rdim  = data['residue-dim']
        tau   = data['tau']

        # VF Offset 
        d = tocmplx(data['d'])

        # Vector of poles
        q        = np.empty(order   , dtype=complex)
        residues = np.empty((order, *Rdim), dtype=complex)

        # Format Poles and Residues
        for i, pole in enumerate(data['poles']):

            q[i]         = tocmplx(pole['q'])
            Rreal, Rimag = pole['r']['real'], pole['r']['imag']
            residues[i]  = np.array(Rreal) + 1j*np.array(Rimag)

        # Return Model Object
        return VFModel(order, d, q, residues, tau)



class FDLineModel(JSONModel):
    '''
    Data Structure of FD Line holding VF Admittance and Propagation Functions, as well as other line information
    '''

    def __init__(self, ell:int, ncond: int, H:VFModel, Yc: VFModel) -> None:
        self.ell = ell 
        self.ncond = ncond
        self.H = H
        self.Yc = Yc 

    def __str__(self) -> str:
        txt = f'Line Length: {self.ell/1000:.1f} [km]\n'
        txt += f'Conductors: {self.ncond} [#]\n'
        return txt
    
    def __repr__(self) -> str:
        return self.__str__()
    

    def asdict(self):

        Hdict = VFModel.asdict(self.H)
        Ydict = VFModel.asdict(self.Yc)

        return {
            "length"  : self.ell ,
            "ncond"   : self.ncond,
            "H-model" : Hdict,
            "Y-model" : Ydict
        }
    
    @classmethod
    def fromdict(cls, data):

        ell = data['length']
        ncond = data['ncond']

        H = VFModel.fromdict(data['H-model'])
        Y = VFModel.fromdict(data['Y-model'])

        # Return Model Object
        return FDLineModel(ell, ncond, H, Y)

