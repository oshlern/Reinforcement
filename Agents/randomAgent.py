import numpy as np
from abstractAgent import Agent

class RandomChoose(Agent):
    def action(self, env, state, actions):
        return actions[np.random.randint(len(actions))]
