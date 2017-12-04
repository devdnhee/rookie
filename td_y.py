import chess
import numpy as np

class TDLambda:
    """
    Implementation of TD-lambda learning table method.

    Attributes:
        z: eligibility trace, dictionary with as (K,V) pair the
            (state,eligibility) where the eligibiliy is the weighing of the state
            for the episode, kind of a history of states
            should be reset after each episode!
        V: value table, dictionary with (K,V)=(state,value)
        l: lambda=0 -> TD-0
            lambda=1 -> every visit Monte Carlo
            => indicates how long states should be updated
        y: gamma=discount factor
    """

    TYPE='TABLE'

    def __init__(self, l=0.5, y=0.8, a=0.2):
        self.z=dict()
        self.V=dict()
        self.l=l
        self.y=y
        self.a=a

    def update(self,x,r,y):
        """
        TD-lambda update
        """
        if x not in self.V.keys():
            self.V[x]=0.001*np.random.randn()*0
        if y not in self.V.keys():
            self.V[y]=0.001*np.random.randn()*0
        delta=r+self.y*self.V[y]-self.V[x]
        for s in self.z.keys():
            self.z[s]=self.y*self.l*self.z[s]
            self.V[s]+=self.a*delta*self.z[s]
        self.z[x]=1
        self.V[x]+=self.a*delta
        #print 'New state value: ', self.V[x]

    def episodical_update(self):
        """
        to make code more transparent
        """
        self.reset()
        
    def reset(self):
        """
        reset z after each episode!
        """
        self.z=dict()

    def value(self,s):
        """
        value function V(s)
        """
        if s not in self.V.keys():
            self.V[s]=0.001*np.random.randn()*0
        return self.V[s]

    def set_modelfile(self,fn):
        pass

    def fit(self,device=None):
        """
        do nothing, to make code more transparent with approximation approaches
        """
        pass

