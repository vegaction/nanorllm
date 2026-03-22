from abc import ABC

class BaseEnv(ABC):
    def reset(self, task):
        return 
    
    def step(self, action):
        return