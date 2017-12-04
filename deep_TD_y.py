"""In this file, everything is organized for the plain TD-lambda method in a
deep learning framework. It contains code which is the framework for the
algorithms incorporating deeper search methods in the general TD(lambda)
algorithm, namely TD-Leaf(lambda) and TD-Stem(lambda)"""

import chess
import settings
import time
import sys
import multiprocessing as mp 
from chess_utils import map_side_to_int,rotate
from supervisor import Supervisor
from environment import Environment
import learn.ann as ann
from data_mgmt import DataManager
import tensorflow as tf
import numpy as np
from learn.preprocessing import faster_featurize
from policy import EpsilonGreedyPolicy,GreedyPolicy
from agent import Agent

class TDyEpisodeProcess(mp.Process):
    """
    An EpisodeProcess is a mp.Process which regulates an iteration of chess
    simulations (self play).

    Attributes:
        ep_task_q: mp.Queue with iterations to follow
        ep_task_lock: mp.Lock on ep_task_q
        task_q: mp.Queue holding the waiting line for episodes
        pol: Policy (Greedy or EpsilonGreedyPolicy)
        state: None except if the same starting state is used over and over
        mv_limit: max number of halfmoves in an episode
        q_lock: mp.Lock on task_q
        new_iter_cond: mp.Condition
        res_q: mp.Queue for getting results from the function approx network
        res_lock: mp.Lock for res_q
        self.eps_change_ev: mp.Event for when epsilon (of epsilon greedy)
            changes
    """

    def __init__(self,task_q,q_lock,pol,state,mv_limit,new_iter_cond,res_q,res_lock,ep_task_q,ep_task_lock,eps_change_ev,name):
        mp.Process.__init__(self,name=name)
        self.ep_task_q=ep_task_q
        self.ep_task_lock=ep_task_lock
        self.task_q=task_q
        self.pol=pol
        self.state=state
        self.mv_limit=mv_limit
        self.q_lock=q_lock
        self.new_iter_cond=new_iter_cond
        self.res_q=res_q
        self.res_lock=res_lock
        self.eps_change_ev=eps_change_ev

    def set_conn(self,conn):
        self.conn=conn

    def run(self):
        # need to generate a new random seed in every new process, otherwise
        # processes generate same random numbers!!!
        print 'I {} starting'.format(self.name)
        np.random.seed()
        ctr=0

        while True:
            # run an iteration, killing is controlled by the Supervisor
            self.new_iter_cond.acquire()
            self.new_iter_cond.wait()
            self.new_iter_cond.release()
            if self.eps_change_ev.is_set() and ctr!=0:
                self.pol.update()
                print ('I {} changed epsilon to{}'
                    .format(self.name,self.pol.eps))

            while True:
                # run episodes
                with self.q_lock:
                    next_task=self.task_q.get()
                if next_task is None:
                    # poison pill
                    self.task_q.task_done()
                    print '{} OUT, ran {} episodes'.format(self.name,ctr)
                    break 
                else:
                    res=self.run_episode()
                    with self.res_lock:
                        self.res_q.put(res)
                    self.task_q.task_done()
                    ctr+=1

    def run_episode(self):
        """ episodical run, return all necessary info for the learning
        algorithm"""
        env=Environment(state=self.state)
        ag=TDyAgent(pol,self.conn,self.ep_task_q,self.ep_task_lock)
        mv_cnt=1
        rewards=[0,0]
        while mv_cnt<self.mv_limit and not env.is_terminal():
            index=int(not env.get_turn())
            s,a,r,sn=ag.take_action(env)
            rewards[index]+=r
            mv_cnt+=1

        return (rewards, mv_cnt, ag.data_thread) 

class TDyAgent(Agent):
    """
    Agent for TD lambda with function approximation
    """

    def __init__(self, policy,conn,ep_task_q,ep_task_lock):
        Agent.__init__(self, policy)
        self.name=mp.current_process().name
        self.conn=conn
        self.ep_task_q=ep_task_q
        self.ep_task_lock=ep_task_lock
    
    def get_av_pairs(self, env):
        """
        get action value (AV) pairs corresponding with Environment 
        """
        as_pairs=env.get_as_pairs()
        as_pairs.append((None,env.current_state))
        # need to take into account that it's a zero sum game
        # invert value if black
        S=[t[1] for t in as_pairs]
        N=len(S)
        S=np.array([self.data_thread.put_and_get(s) for s in S])
        S=np.reshape(S,(S.shape[0],S.shape[-1]))

        with self.ep_task_lock:
            self.ep_task_q.put((self.name,S))
        v=self.conn.recv()

        if v is None:
            for p in mp.active_children():
                print p.name
                if p.name[:3]=='Epi':
                    p.terminate()
        v=map_side_to_int(env.get_turn())*v

        try:
            av=[(as_pairs[i][0],v[i,0]) for i in xrange(N)]
        except:
            env.draw()
            print as_pairs
            print S.shape
            print N
            print v.shape
            import time
            time.sleep(10)

        return av

    def take_action(self, env, a=None): 
        """
        perform action -> changes the Environment
        """
        s=env.current_state
        if a is None:
            av=self.get_av_pairs(env)
            _,v=av.pop()
            a,_=self.policy.choose_action(av)
        # r reward from side to move perspective
        r,s_n=env.perform_action(a)
        # r_w reward from whites perspective
        r_w=(-1 if env.get_turn() else 1)*r # invert value if white 
        
        self.data_thread.append((s,r_w,v)) 
        if Environment.terminal_state(s_n):
            self.data_thread.append((s_n,r_w,r_w))
            self.data_thread.set_outcome(env.outcome())
        return s, a, r, s_n


class TDyPlayAgent(Agent):

    def __init__(self,approx):
        Agent.__init__(self, GreedyPolicy())
        self.approx=approx
    
    def get_av_pairs(self, env):
        """
        get action value (AV) pairs corresponding with Environment 
        """
        as_pairs=env.get_as_pairs()
        # need to take into account that it's a zero sum game
        # invert value if black
        S=[t[1] for t in as_pairs]
        N=len(S)
        S=np.array([faster_featurize(s) for s in S])
        S=np.reshape(S,(S.shape[0],S.shape[-1]))

        v=self.approx.value(S)
        v=map_side_to_int(env.get_turn())*v

        av=[(as_pairs[i][0],v[i,0]) for i in xrange(N)]
        return av

    def take_action(self, env, a=None): 
        """
        perform action -> changes the Environment
        """
        s=env.current_state
        if a is None:
            av=self.get_av_pairs(env)
            a,v=self.policy.choose_action(av)
        # r reward from side to move perspective
        r,s_n=env.perform_action(a)
        return s, a, r, s_n

class TDyDataManager(DataManager):
    """class implementing the algorithm on the data"""

    def __init__(self,l=0.5,y=0.7,max_keep=30):
        DataManager.__init__(self)
        self.LAMBDA=np.array([l**i for i in xrange(300)])
        self.GAMMA=np.array([y**i for i in xrange(300)])
        self.max_keep=settings.params['MK']

    def update(self, data_thread):
        S,R,V=data_thread.get_data()
        S=S[-self.max_keep:]
        R=R[-self.max_keep:]
        V=V[-self.max_keep:]

        # Data augmentation
        Sr=[rotate(S[i]) for i in xrange(len(S))]        

        X=np.array([data_thread.put_and_get(s) for s in S])
        X=np.reshape(X,(X.shape[0],X.shape[-1]))
        Xr=np.array([data_thread.put_and_get(s) for s in Sr])
        Xr=np.reshape(Xr,(Xr.shape[0],Xr.shape[-1]))
        
        O=data_thread.get_outcome()*np.ones(X.shape[0])
        Or=data_thread.get_outcome()*np.ones(Xr.shape[0])

        T=len(R)-1
        G=np.zeros((T,T))
        R=np.array(R)
        for k in xrange(T):
            for n in range(1,T-k+1):
                # n-step return
                G[k,n-1]=np.dot(self.GAMMA[:n],R[k:k+n])+self.GAMMA[n]*V[k+n]
           
        # lambda return
        G_lambda=[(1-self.LAMBDA[0])*np.dot(self.LAMBDA[:T-k],G[k,:T-k])+self.LAMBDA[T-k]*R[-1] for k in xrange(T)]
        G_lambda.append(R[-1])

        V_target=np.array(G_lambda).reshape((T+1,1))
        V_target_r=-V_target

        self.internal_counter+=1
        V_target=np.concatenate((V_target,V_target_r))
        X=np.concatenate((X,Xr))
        O=np.concatenate((O,Or))
        S=S+Sr

        if self.X is None:
            self.X=X
            self.Y=V_target
            self.O=O
            self.S+=S
        else:
            to_keep=min(O.shape[0],self.internal_contribution)
            p=np.random.permutation(O.shape[0])
            X=X[p]
            V_target=V_target[p]
            O=O[p]
            S=[S[i] for i in p]
            X=X[:to_keep]
            V_target=V_target[:to_keep]
            O=O[:to_keep]
            S=S[:to_keep]
            self.X=np.concatenate((self.X,X))
            self.Y=np.concatenate((self.Y,V_target))
            self.O=np.concatenate((self.O,O)) 
            self.S+=S

        if self.Y.shape[0]>self.internal_buffer:
            self.shuffle()
        

class TDySupervisor(Supervisor):

    def __init__(self,policy,store_period=10000,sample_period=1000,mv_limit=100,y=1,l=0.7):
        Supervisor.__init__(self,policy,store_period=store_period,sample_period=sample_period,mv_limit=mv_limit)
        self.dm=TDyDataManager(y=y,l=l)

    def get_name(self):
        return 'DeepTDy'

    def create_proc(self,task_q,q_lock,state,new_iter_cond,res_q,res_lock,ep_task_q,ep_task_lock,eps_change_ev,name):
        return TDyEpisodeProcess(task_q,q_lock,self.pol,state,self.mv_limit,new_iter_cond,res_q,res_lock,ep_task_q,ep_task_lock,eps_change_ev,name) 

