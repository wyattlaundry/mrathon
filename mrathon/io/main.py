


from abc import ABC, abstractmethod
import json


class JSONModel(ABC):

    def __init__(self) -> None:
        pass

    @classmethod
    @abstractmethod
    def fromdict(cls, data):
        pass

    @classmethod
    def read(cls, fname: str):
        '''Crates object from JSON structure in file'''
        with open(fname, 'r') as file:
            data = json.load(file)
            obj  = cls.fromdict(data)
            return obj
        
    @abstractmethod
    def asdict(self):
        pass
        
    def write(self, fname: str):
        '''Writes Dictionary JSON Form to specified file'''
        with open(fname, 'w') as file:
            data = self.asdict()
            json.dump(data, file, indent=2)

    
    
    

