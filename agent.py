import numpy as np
from abc import ABCMeta,abstractmethod
from data_mgmt import DataThread
from environment import Environment
from chess_utils import map_side_to_int
import chess
from policy import GreedyPolicy

class Agent:
    """
    class modeling the Agent in the RL paradigm.
    In chess, this is a player (black or white)

    Attributes:
        color: BLACK or WHITE
        V: value function
        P: policy
        H: state history (eg. of an episode)
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def get_av_pairs(env): pass 

    def __init__(self, policy):
        """
        Constructor
        """
        self.policy=policy
        self.data_thread=DataThread()

    def play(self, env):
        """
        perform a greedy action, (play the best move according to the
        agent)
        """
        av=self.get_av_pairs(env)
        a=GreedyPolicy().choose_action(av)[0]
        r,s_n=env.perform_action(a)
        return a, r, s_n

    @abstractmethod
    def take_action(self, env, a=None): 
        """
        perform action -> changes the Environment
        """
        pass 

    def update_data(self,dm):
        """
        transfer data to a DataManager class so it can be processed
        """
        dm.update(self.data_thread,self.approximator)
        pass


