from abc import ABC
from nanorllm.core.trajectory import Trajectory


class BaseAgent(ABC):
    def reset(self):
        return
    
    def update_from_env(self, observation, reward, done, info):
        raise NotImplementedError("")
    
    def update_from_model(self, reponse):
        raise NotImplementedError("")
    

    @property
    def messages(self):
        return []
    
    def trajectory(self):
        return Trajectory()