import numpy as np

class EpsilonGreedyPolicy:
    """
    Epsilon Greedy Policy

    Attributes:
        eps: epsilon
        decay_period: eps is changed every decay_period episodes
        decay_rate: the amount of change when eps changes
        counter: counts the amount of actions 
    """


    def __init__(self,eps=1.,decay_f=lambda n:(n+1)**(-2./3)):
        self.eps=eps
        self.n=0
        self.decay_f=decay_f
        if eps is None:
            self.eps=self.decay_f(self.n)

    def choose_action(self,av_pairs):
        """
        epsilon greedy choice of action in the RL paradigm
        """
        if np.random.uniform()<self.eps:
            rand=np.random.randint(low=0,high=len(av_pairs))
            return av_pairs[rand][0],av_pairs[rand][1]
        else:
            return GreedyPolicy().choose_action(av_pairs)

    def update(self):
        self.n+=1
        self.eps=max(0.05,self.decay_f(self.n))

class GreedyPolicy:
    """
    Greedy Policy wrapper
    """

    def __init__(self):
        pass

    def choose_action(self, av_pairs): 
        best_v=max([v for (a,v) in av_pairs])
        best_a=[a for (a,v) in av_pairs if v==best_v]
        rand=int(np.random.uniform(low=0,high=len(best_a)))
        return best_a[rand],best_v
