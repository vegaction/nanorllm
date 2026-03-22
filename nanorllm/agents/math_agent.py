from nanorllm.agents.base import BaseAgent
from nanorllm.core.trajectory import Trajectory, Step
from nanorllm.core.types import Action


class MathAgent(BaseAgent):
    def __init__(self, system_prompt):
        self.system_prompt = system_prompt
        self._messages = []
        self._trajectory = Trajectory()
        


    def reset(self):
        self._trajectory = Trajectory()
        self._messages = [{"role": "system", "content": self.system_prompt}]
        
        return
    
    def update_from_env(self, observation, reward, done, info):
        if self.trajectory.steps:
            last_step = self.trajectory.steps[-1]
            last_step.reward = reward
            last_step.done = done
            last_step.info = info
            
        termination_reason = info.get('termination_reason') if info else None

        if done or termination_reason is not None:
            self._trajectory.final_reward = reward
            self._trajectory.terminated = True
            if not termination_reason:
                self._trajectory.termination_reason = termination_reason
            else:                
                self._trajectory.termination_reason = 'done' if done else 'unknown'
            return

        user_message = self._format_observation(observation)
        self._messages.append({'role': "user", "content": user_message})
        if self._trajectory.task_id is None and info:
            self._trajectory.task_id = info.get('task_id')
        
        self._trajectory.steps.append(Step(observation=observation))


    def update_from_model(self, response):
        action = Action(response)

        prompt_messages = self._messages.copy() # 很重要，需要copy

        cur_step = self._trajectory.steps[-1]
        cur_step.prompt_messages = prompt_messages
        cur_step.model_response = response
        cur_step.action = action

        self.messages.append({'role': "assistant", "content": response})
        return action

    def _format_observation(self, observation):
        if isinstance(observation, dict):
            if 'feedback' in observation:
                return observation['feedback']
            if 'question' in observation:
                return observation['question']
        return observation        

    @property   
    def messages(self):
        return self._messages
    
    @property
    def trajectory(self):
        return self._trajectory
