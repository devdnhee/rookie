"""
This module contains all classes used in the TDLeaf(lambda) framework. The
algorithm is discussed in the master thesis chapter 4. The main difference is
how the data is transferred to the DataManager, namely by incorporating
value function from leaf nodes in the search tree
"""

import chess
import cPickle as cp
import settings
from chess_utils import rotate
import search
from supervisor import Supervisor
import sys
import multiprocessing as mp 
from chess_utils import map_side_to_int
from supervisor import Supervisor
from environment import Environment,load_DS
import learn.ann as ann
from data_mgmt import DataManager
import tensorflow as tf
import numpy as np
from learn.preprocessing import faster_featurize,change_PL
from policy import EpsilonGreedyPolicy,GreedyPolicy
from agent import Agent
import deep_TD_y as tdy
import learn.cnn as cnn

USE_DSET=False

class TDLeafEpisodeProcess(tdy.TDyEpisodeProcess):
    """
    subclass of tdy.TDyEpisodeProcess for TD-Leaf(lambda) 
    """
    def __init__(self,depth,task_q,q_lock,pol,state,mv_limit,new_iter_cond,res_q,res_lock,ep_task_q,ep_task_lock,eps_change_ev,name):
        tdy.TDyEpisodeProcess.__init__(self,task_q,q_lock,pol,state,mv_limit
            ,new_iter_cond,res_q,res_lock,ep_task_q,ep_task_lock,eps_change_ev,name)
        self.depth=depth

    def run_episode(self):
        env=Environment(state=self.state,draw_r=0,move_r=0,mate_r=10,dset=settings.params['USE_DSET'])
        ag=TDLeafAgent(self.pol,self.conn,self.ep_task_q,self.ep_task_lock,depth=self.depth)
        mv_cnt=1
        rewards=[0,0]
        while mv_cnt<self.mv_limit and not env.is_terminal():
            index=int(not env.get_turn())
            s,a,r,sn=ag.take_action(env)
            rewards[index]+=r
            mv_cnt+=1
        return (rewards, mv_cnt, ag.data_thread) 


class TDLeafAgent(tdy.TDyAgent):

    def __init__(self,policy,conn,ep_task_q,ep_task_lock,depth=4):
        tdy.TDyAgent.__init__(self,policy,conn,ep_task_q,ep_task_lock)
        self.depth=depth

    def get_av_pairs(self,env):
        a,v,l= search.alphabeta_batch_hist_leaf(self.V,faster_featurize,env,list(env.hist),self.depth,-float('inf'),float('inf'))
        ao,o=search.alphabeta_outcome(None,None,env.current_state,settings.params['OC_DEPTH'],-float('inf'),float('inf'))
        if o>0:
            a=ao
        return a,v,l

    def V(self,S):
        """value function"""
        with self.ep_task_lock:
            self.ep_task_q.put((self.name,S))
        v=self.conn.recv()
        return v

    def take_action(self,env,a=None): 
        s=env.current_state
        as_pairs=env.get_as_pairs()
        e=np.random.rand()
        if e<self.policy.eps:
            rand=np.random.randint(len(as_pairs))  
            a=as_pairs[rand][0]
            sn=as_pairs[rand][1]
            _,v,leaf=search.alphabeta_batch_hist_leaf(self.V,faster_featurize,Environment(sn),list(env.hist),self.depth-1,-float('inf'),float('inf'))
        else:
            a,v,leaf=self.get_av_pairs(env)

        r,s_n=env.perform_action(a)
        r_w=(-1 if env.get_turn() else 1)*r # invert value if white 
        
        self.data_thread.append((leaf,r,v)) 
        if Environment.terminal_state(s_n):
            self.data_thread.append((s_n,r,r))
            self.data_thread.set_outcome(env.outcome())
        return s, a, r, s_n


class TDLeafPlayAgent(tdy.TDyPlayAgent):

    def __init__(self,approx,depth=4):
        tdy.TDyPlayAgent.__init__(self,approx)
        self.depth=depth

    def get_av_pairs(self,env):
        return [search.alphabeta_batch_hist(self.approx.value,faster_featurize,env,list(env.hist),self.depth,-float('inf'),float('inf'))]

    def take_action(self,env,a=None): 
        ao,o=search.alphabeta_outcome(None,None,env.current_state,settings.params['OC_DEPTH'],-float('inf'),float('inf'))
        if o>0:
            env.perform_action(ao) 
        else:   
            a,v=search.alphabeta_batch_hist(self.approx.value,faster_featurize,env,list(env.hist),self.depth,-float('inf'),float('inf'))
            env.perform_action(a)

class TDLeafSupervisor(Supervisor):

    def __init__(self,policy,store_period=10000,sample_period=1000,mv_limit=100,depth=4,y=1,l=0.8):
        Supervisor.__init__(self,policy,store_period=store_period,sample_period=sample_period,mv_limit=mv_limit)
        self.dm=tdy.TDyDataManager(y=y,l=l)
        self.depth=depth
        self.gamma=y
        self.lambd=l
        if 'depth' not in self.meta.keys():
            self.meta['depth']=[(0,depth)]
            self.meta['gamma']=[(0,y)]
            self.meta['lambda']=[(0,l)]
        else:
            self.meta['depth'].append((self.meta['episodes'],depth))
            self.meta['gamma'].append((self.meta['episodes'],y))
            self.meta['lambda'].append((self.meta['episodes'],l))

    def get_name(self):
        return 'TDLeaf'

    def retrieve(self,fn):
        with open(fn,'rb') as f:
            self.meta=cp.load(f)
        self.meta['depth'].append((self.meta['episodes'],self.depth))
        self.meta['gamma'].append((self.meta['episodes'],self.gamma))
        self.meta['lambda'].append((self.meta['episodes'],self.lambd))

    def create_proc(self,task_q,q_lock,state,new_iter_cond,res_q,res_lock,ep_task_q,ep_task_lock,eps_change_ev,name):
        return TDLeafEpisodeProcess(self.depth,task_q,q_lock,self.pol,state,self.mv_limit,new_iter_cond,res_q,res_lock,ep_task_q,ep_task_lock,eps_change_ev,name) 

