import numpy as np
import settings
from environment import Environment
import deep_TD_y as tdy
import os,inspect
from agent import Agent
from policy import GreedyPolicy
import chess.gaviota as gav
import chess
import chess.uci as uci
import td_stem as tdstem
import tensorflow as tf
from approximator import Approximator
import tablebases 
from chess_utils import map_side_to_int

currentdir=os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
gavdir=os.path.join(currentdir,'Gaviota')

class OptimalAgent(Agent):
    """Agent which plays solved endgames perfectly with the help of a Gaviota
    TB. (hence can only be used when TB are installed)"""

    def __init__(self,policy=GreedyPolicy()):
        Agent.__init__(self,policy)
        self.approx=OptimalApproximator()
        
    def get_av_pairs(self,env):
        """getting action value pairs from a state"""
        as_pairs=env.get_as_pairs()
        S=np.array([t[1] for t in as_pairs])
        A=[t[0] for t in as_pairs]
        # invert score, because S from other side's perspective
        V=self.approx.value(S,inv=True)
        return zip(A,np.asarray(V))

    def take_action(self,env,a=None): 
        s=env.current_state
        if a is None:
            av=self.get_av_pairs(env)
            a,v=self.policy.choose_action(av)
        # r reward from side to move perspective
        r,s_n=env.perform_action(a)
        # r_w reward from whites perspective
        return s, a, r, s_n

    def get_best_moves(self,env):
        av=self.get_av_pairs(env)
        v=max([t[1] for t in av])
        a=[t[0] for t in av if t[1]==v]
        return a

def J(tb,s):
    """sort of value function for the optimal agent, so minimax algorithms can
    be used"""
    if s is None:
        return 0
    b=chess.Board.from_epd(s)[0]
    if b.is_checkmate():
        return -2
    else:
        return 1./tb.probe_dtm(b) if tb.probe_dtm(b)!=0 and tb.probe_dtm(b) is not None else 0

class OptimalApproximator():
    """not really an approximator. More a ground truth."""

    def __init__(self):
        self.tb=gav.open_tablebases(gavdir)

    def value(self,S,inv=False):
        return np.array([J(self.tb,S[i]) if not inv else -J(self.tb,S[i]) for i
            in xrange(S.shape[0])])

    def dtm(self,s):
        return np.abs(self.tb.probe_dtm(chess.Board.from_epd(s)[0]))

def evaluate(start_state,agent):
    """yielding an score to evaluate an agent compared to the ground truth."""
    env=Environment()
    env.reset(state=start_state)

    optAgent=OptimalAgent()

    mv_cnt=0.
    loss=0
    while not env.is_terminal() or mv_cnt>=50:
        env.draw()
        av=optAgent.get_av_pairs(env) 
        d_av=dict(av)
        print d_av
        a,_,_=agent.play(env)
        a_max=optAgent.policy.choose_action(av)[0] 
        print a,a_max
        print d_av[a_max]-d_av[a]
        loss+=d_av[a_max]-d_av[a]
        mv_cnt+=1

    return loss/mv_cnt

def recursive_eval(env,board,A,agents,start_turn,S):
    """yielding an score to evaluate an agent compared to the ground truth.
    Based on the metrics discussed in the thesis document."""
    assert board.epd()==env.current_state
    S.append(env.current_state)
    th_wdl=map_side_to_int(start_turn == env.get_turn())*tablebases.TB.probe_wdl(board) #1->winning, 0->drawing, -1->losing
    th_dtm=np.abs(tablebases.TB.probe_dtm(board))
    if board.is_game_over(claim_draw=True):
        if board.result()=='1-0':
            if start_turn:
                wdl=1
            else:
                wdl=-1
        elif board.result()=='0-1':
            if start_turn:
                wdl=-1
            else:
                wdl=1
        else:
            wdl=0
        return wdl,0, wdl,0,env.current_state 
    else:
        a,_,_=agents[env.get_turn()].play(env)
        board.push_uci(a)
        _,_, wdl, dtm,s=recursive_eval(env,board,A,agents,start_turn,S)
        A.append((th_wdl,th_dtm,wdl,dtm+1,s))
        return th_wdl,th_dtm,wdl,dtm+1,env.current_state

def recursive_eval_sim(agent,N=10,w=False):
    """simulation of recursive_eval on N samples.
    params:
        N: number of episodes to simulate
        w: side of the leaned model (white or black)
    """

    fn='games.txt'

    eval_dict=dict()
    eval_dict['lhs']=[]
    eval_dict['wc']=[]
    eval_dict['dc']=[]
    eval_dict['we']=[]
    A=[]

    with open(fn,'w') as f:
        # also storing which states were occurred for observing the games
        # afterwards
        all_s=[]
        for i in xrange(N):
            agents=dict()
            env=Environment(settings.params['DSET'][i])
            start_turn=env.get_turn()
            board=chess.Board.from_epd(env.current_state)[0]
            if w:
                agents[True]=agent
                agents[False]=OptimalAgent()
                start_turn=True
            else:
                agents[not board.turn]=OptimalAgent()
                agents[board.turn]=agent
            print board.turn

            env.draw()
            S=[]
            th_wdl,th_dtm,wdl,dtm,_=recursive_eval(env,board,A,agents,start_turn,S)
            print 'TB WDL: {}\tTB DTM: {}\t'.format(th_wdl,th_dtm)
            env.draw()
            print 'WDL: {}\tLENGTH: {}\n\n'.format(wdl,dtm)
            if th_wdl==-1:
                if th_dtm!=0:
                    eval_dict['lhs'].append(-float(dtm)/th_dtm)
            elif th_wdl==1:
                eval_dict['wc'].append(int(wdl==1))
                if wdl==1:
                    eval_dict['we'].append(float(th_dtm)/dtm)
            else:
                eval_dict['dc'].append(int(wdl==0))

            if dtm>th_dtm and th_dtm<=3 and wdl!=1 or th_wdl==-1 and (wdl!=-1 or dtm>th_dtm):  
                res=('TB WDL: {}\tTB DTM: {}\tWDL: {}\tLENGTH: {}\n\n'
                    .format(th_wdl,th_dtm,wdl,dtm))
                f.write(res)
                f.write(str(env.hist))
                a=OptimalAgent()
                for b in S:
                    f.write(str(a.get_av_pairs(Environment(b))))
                    f.write(str(chess.Board.from_epd(b)[0])+'\n\n')

            all_s+=S

        return A,eval_dict,all_s 

def evaluation(env,agent):
    '''
    metrics can be calculated after this
    '''
    agents=dict()

    start_turn=env.get_turn()

    board=chess.Board.from_epd(env.current_state)[0]
    agents[not board.turn]=OptimalAgent()
    agents[board.turn]=agent

    th_wdl=tablebases.TB.probe_wdl(board) #1->winning, 0->drawing, -1->losing
    th_dtm=tablebases.TB.probe_dtm(board)

    mv_cnt=0
    while not board.is_game_over(claim_draw=True):
        a,_,_=agents[env.get_turn()].play(env)
        board.push_uci(a)
        mv_cnt+=1

        if board.result()=='1-0':
            if start_turn:
                wdl=1
            else:
                wdl=-1
        elif board.result()=='0-1':
            if start_turn:
                wdl=-1
            else:
                wdl=-1
        else:
            wdl=0

    return th_wdl, th_dtm, wdl, mv_cnt

def model_play(m1,m2,play_file):
    # script playing models against each other. (slooow)
    import cPickle as cp
    with open(play_file,'rb') as f:
        states=cp.load(f)

    scores=[0,0]

    for s in states:
        for i in xrange(2):
            if i==0:
                M=[m1,m2]
            else:
                M=[m2,m1]

            e=Environment(s)
            board=chess.Board.from_epd(e.current_state)[0]

            while not board.is_game_over(claim_draw=True):
                print board,'\n\n'

                with tf.Session() as sess:
                    saver=tf.train.import_meta_graph(M[int(e.get_turn())]+'.meta')
                    saver.restore(sess,M[int(e.get_turn())])
                    approx=Approximator(sess)
                    agent=tdstem.TDStemPlayAgent(approx,depth=3)
                    
                    a,_,_=agent.play(e)
                    board.push_uci(a)

            if board.result()=='1-0':
                if i==0: scores[0]+=1
                else: scores[1]+=1
            elif board.result()=='0-1':
                if i==0: scores[1]+=1
                else: scores[0]+=1

    print scores


    while not board.is_game_over(claim_draw=True):
        a,_,_=agents[env.get_turn()].play(env)
        board.push_uci(a)
        mv_cnt+=1

        if board.result()=='1-0':
            if start_turn:
                wdl=1
            else:
                wdl=-1
        elif board.result()=='0-1':
            if start_turn:
                wdl=-1
            else:
                wdl=-1
        else:
            wdl=0

def evaluation_simulation(agent,N=10):
    eval_dict=dict()
    eval_dict['lhs']=[]
    eval_dict['wc']=[]
    eval_dict['dc']=[]
    eval_dict['we']=[]

    for _ in xrange(N):
        env=Environment(dset=False)
        env.draw()
        th_wdl,th_dtm,wdl,dtm=evaluation(env,agent)
        print 'TB WDL: {}\tTB DTM: {}\t'.format(th_wdl,th_dtm)
        env.draw()
        print 'WDL: {}\tLENGTH: {}\n\n'.format(wdl,dtm)
        if th_wdl==-1:
            eval_dict['lhs'].append(-float(mv_cnt)/th_dtm)
        elif th_wdl==1:
            eval_dict['wc'].append(int(wdl==1))
            if wdl==1:
                eval_dict['we'].append(float(th_dtm)/dtm)
        else:
            eval_dict['dc'].append(int(wdl==0))

    return eval_dict

def wc(t):
    """WIN CONVERSION RATE"""
    return int(t[2]==1) if t[0]==1 else None

def we(t):
    """WIN EFFICIENCY"""
    return float(t[1])/t[3] if t[0]==1 and t[2]==1 else None

def lhs(t):
    """LOSS HOLDING SCORE"""
    return float(t[3])/t[1] if t[0]==-1 and t[1]!=0 else None

def dc(t):
    """DRAW CONVERSION RATE"""
    return int(t[2]==0) if t[0]==0 else None

if __name__=='__main__':
    '''
    model_fn='Models/good_models/TDStem_krk/TDLeaf_td_leaf_cont__12_05-0_0312331-900'
    with tf.Session() as sess:
        saver=tf.train.import_meta_graph(model_fn+'.meta')
        saver.restore(sess,model_fn)
        approx=Approximator(sess)
        agent=tdstem.TDStemPlayAgent(approx,depth=3)
        print recursive_eval_sim(agent,N=10)
    '''
    import settings
    settings.params['PL']=list('KQRBNPkqrnbp')
    model_play('Models/3P/TDS4/network','Models/3P/TDL4/network','dataset/test.epd')
    
