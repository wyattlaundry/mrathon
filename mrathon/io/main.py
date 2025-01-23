


from abc import ABC, abstractmethod
import json
from typing import Type, TypeVar
T = TypeVar('T', bound='JSONModel')

class JSONModel(ABC):

    def __init__(self) -> None:
        pass

    @classmethod
    @abstractmethod
    def fromdict(cls, data):
        pass

    @classmethod
    def read(cls: Type[T], fname: str) -> T:
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

    
    
    

