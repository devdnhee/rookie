import chess
import search
from supervisor import Supervisor
import sys
import multiprocessing as mp 
from chess_utils import map_side_to_int
from supervisor import Supervisor
from environment import Environment
import learn.ann as ann
from data_mgmt import DataManager
import tensorflow as tf
import numpy as np
from learn.preprocessing import faster_featurize
from policy import EpsilonGreedyPolicy,GreedyPolicy
from agent import Agent
import deep_TD_y as tdy

class TDStemEpisodeProcess(tdy.TDyEpisodeProcess):
    def __init__(self,depth,task_q,q_lock,pol,state,mv_limit,new_iter_cond,res_q,res_lock,ep_task_q,ep_task_lock,eps_change_ev,name):
        tdy.TDyEpisodeProcess.__init__(self,task_q,q_lock,pol,state,mv_limit
            ,new_iter_cond,res_q,res_lock,ep_task_q,ep_task_lock,eps_change_ev,name)
        self.depth=depth

    def run_episode(self):
        env=Environment(state=self.state,draw_r=-1,move_r=-0.001,mate_r=1)
        ag=TDStemAgent(pol,self.conn,self.ep_task_q,self.ep_task_lock,depth=self.depth)
        #print '{} running episode'.format(ag.name)
        mv_cnt=1
        rewards=[0,0]
        while mv_cnt<self.mv_limit and not env.is_terminal():
            #env.draw()
            index=int(not env.get_turn())
            s,a,r,sn=ag.take_action(env)
            rewards[index]+=r
            #print s,a,r,sn
            #import time
            #time.sleep(1)
            mv_cnt+=1
        return (rewards, mv_cnt, ag.data_thread) 


class TDStemAgent(tdy.TDyAgent):

    def __init__(self,policy,conn,ep_task_q,ep_task_lock,depth=4):
        tdy.TDyAgent.__init__(self,policy,conn,ep_task_q,ep_task_lock)
        self.depth=depth

    def get_av_pairs(self,env):
        return search.alphabeta_batch_hist(self.V,faster_featurize,env,list(env.hist),self.depth,-float('inf'),float('inf'))

    def V(self,S):
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
            _,v=search.alphabeta_batch_hist(self.V,faster_featurize,Environment(sn),list(env.hist),self.depth-1,-float('inf'),float('inf'))
        else:
            a,v=self.get_av_pairs(env)

        r,s_n=env.perform_action(a)
        # r_w reward from whites perspective
        r_w=(-1 if env.get_turn() else 1)*r # invert value if white 
        #print r_w
        #print r
        
        self.data_thread.append((s,r_w,v)) 
        if Environment.terminal_state(s_n):
            #print 'Reward for white: ',r_w
            self.data_thread.append((s_n,r_w,r_w))
            self.data_thread.set_update()
            if env.result()=='1-0' or env.result()=='0-1':
                #env.draw()
                self.data_thread.set_win()
        return s, a, r, s_n


class TDStemPlayAgent(tdy.TDyPlayAgent):

    def __init__(self,approx,depth=4):
        tdy.TDyPlayAgent.__init__(self,approx)
        self.depth=depth

    def get_av_pairs(self,env):
        return [search.alphabeta_batch_hist(self.approx.value,faster_featurize,env,list(env.hist),self.depth,-float('inf'),float('inf'))]

    def take_action(self,env,a=None): 
        a,v=search.alphabeta_batch_hist(self.approx.value,faster_featurize,env,list(env.hist),self.depth,-float('inf'),float('inf'))
        env.perform_action(a) 


class TDStemSupervisor(Supervisor):

    def __init__(self,policy,store_period=10000,sample_period=1000,mv_limit=100,depth=4,y=1,l=0.8):
        Supervisor.__init__(self,policy,store_period=store_period,sample_period=sample_period,mv_limit=mv_limit)
        self.dm=tdy.TDyDataManager(y=y,l=l)
        self.depth=depth

    def get_name(self):
        return 'TDStem'

    def create_proc(self,task_q,q_lock,state,new_iter_cond,res_q,res_lock,ep_task_q,ep_task_lock,eps_change_ev,name):
        return TDStemEpisodeProcess(self.depth,task_q,q_lock,self.pol,state,self.mv_limit,new_iter_cond,res_q,res_lock,ep_task_q,ep_task_lock,eps_change_ev,name) 

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('name',help='name of the model, will be stored in Models directory')
    parser.add_argument('-s','--state',default=None,help='initial position, random when not, default: random')
    parser.add_argument('-m','--move-count',default=10,help='max number of moves in episode',type=int)
    parser.add_argument('-N',default=50,help='number of episodes every iteration',type=int)
    parser.add_argument('-I',default=20,help='number of iterations',type=int)
    parser.add_argument('-d','--depth',default=3,help='depth',type=int)
    parser.add_argument('-o','--old-model',help='old .sv file, for continuing learning a model')
    parser.add_argument('-c','--ckpt',help='old ckpt, for continued learning')
    parser.add_argument('-M',default='32 6 32',help='hidden units per layer')
    parser.add_argument('-f','--eps-factor',default=0.75,help='epsilon decay',type=float)
    parser.add_argument('-y','--gamma',default=1,help='discount factor',type=float)
    parser.add_argument('--lambd',default=0.7,help='lambda',type=float)
    parser.add_argument('--eps-start',default=0,help='epsilon decay',type=int)
    args=parser.parse_args()

    if args.old_model==None:
        D=faster_featurize('8/6k1/2R5/8/3K4/8/8/8 w - -').shape[1]
        print D
        M=[int(m) for m in args.M.split()]
        kwargs={'D': D, 'M': M, 'learning_rate':1e-4}
        graph_f=ann.build_graph
        pol=EpsilonGreedyPolicy(eps=1.0,decay_f=lambda n:(n+1)**(-args.eps_factor))
        pol.n=args.eps_start
        sv=TDStemSupervisor(pol,mv_limit=args.move_count,depth=args.depth,y=args.gamma,l=args.lambd)
        sv.run(args.I,args.N,graph_f,kwargs,state=args.state,name=args.name)
    else: 
        assert args.ckpt is not None
        pol=EpsilonGreedyPolicy(eps=0.0,decay_f=lambda n:(n+1)**(-args.eps_factor))
        pol.n=args.eps_start
        sv=TDStemSupervisor(pol,mv_limit=args.move_count,depth=args.depth,y=args.gamma,l=args.lambd) 
        sv.retrieve(args.old_model) 
        sv.continue_run(args.I,args.ckpt,name=args.name,N=args.N)
