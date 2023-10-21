import numpy as np
from brain import brain
from GridWorld import GridWorld
class transfer(object):
    def __init__(self):
        pass

    def from_to(self, agent : brain, state, state_, default=.8):
        agent.set_qvalue(state_, default * agent.get_q_table()[state])
        return agent
    
    def from_to_dynamic(self, agent : brain, state, state_):
        agent.set_qvalue(state_, .8 * agent.get_q_table()[state])
        return agent
    
    def from_to_reverse(self, agent : brain, state, state_):
        value = 0 * agent.get_q_table()[state]
        value[0] = -60
        value[1] = -40
        value[2] = -40
        value[3] = -40
        value[4] = -60
        agent.set_qvalue(state_, value)
        return agent
