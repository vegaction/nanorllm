from nanorllm.envs.base import BaseEnv
from nanorllm.core.types import Action, RewardOutput




class MathEnv(BaseEnv):
    def __init__(self, reward_fn, max_turn):
        self.task = None
        self.turn_count = 0
        self.max_turn = max_turn
        self.reward_fn = reward_fn

    def reset(self, task):
        self.task = task
        observation = {"question": task['question']}
        info = {"task_id": task['task_id']}
        self.turn_count = 0 # 不能忘
        return observation, info
    
    def step(self, action):
        reward_output = self.reward_fn(self.task, action)
        self.turn_count += 1
        if reward_output.is_correct:
            done = True
            observation = {"feedback": "success"}
        else:
            if self.turn_count < self.max_turn:
                done = False
                observation = {"feedback": "Your previous answer is incorrect. Try again and put the final answer clearly."}
                
            else:
                done = True
                observation = {"feedback": "exceeds max turn"}

        info = {
                    "is_correct": reward_output.is_correct,
                    **reward_output.metadata,
                }
        return observation, reward_output.reward, done, info
    
   

