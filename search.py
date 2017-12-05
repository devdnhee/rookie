"""Implementation of game search algorithms (minimax and alpha beta search)"""

import chess
import random
from environment import Environment
import numpy as np
from chess_utils import map_side_to_int,get_piece,uci_to_squares,rotate
import chess.gaviota as gav

def minimax(V,F,env,depth):
    """
    native minimax without optimizations
    params:
        V: value function
        F: methode to transform data into features
        env: environment (chess position)
        depth: depth of search
    returns:
        max_a: best action
        max_score: score of this action
    """
    as_pairs=env.get_as_pairs()
    if depth==0 or len(as_pairs)==0:
        return None,map_side_to_int(env.get_turn())*V(F(env.current_state))
    else:
        max_a=None
        max_score=None
        for (a,s) in as_pairs:
            env=Environment(state=s)
            score=-minimax(V,F,env,depth-1)[1]
            if score>max_score:
                max_score=score
                max_a=a
        return max_a,max_score

def alphabeta_native(V,F,env,depth,alpha,beta):
    """
    minimax with alpha beta pruning
    params:
        V: value function
        F: methode to transform data into features
        env: environment (chess position)
        depth: depth of search
        alpha
        beta
    returns:
        max_a: best action
        max_score: score of this action
    """
    as_pairs=env.get_as_pairs()
    if depth==0 or len(as_pairs)==0:
        return None,map_side_to_int(env.get_turn())*V(F(env.current_state))
    else:
        act=None
        for (a,s) in as_pairs: 
            env=Environment(s)
            score=-alphabeta_native(V,F,env,depth-1,-beta,-alpha)[1]
            if score>=beta:
                return a,beta
            elif score>alpha:
                alpha=score
                act=a
        return act, alpha 

def alphabeta_batch(V,F,env,depth,alpha,beta):
    """
    alpha beta pruning on a batch of positions 
    params:
        V: value function
        F: methode to transform data into features
        env: batch of environments (chess positions)
        depth: depth of search
        alpha
        beta
    returns:
        max_a: best action
        max_score: score of this action
    """
    if depth<1:
        S=F(env.current_state)
        return None, map_side_to_int(env.get_turn())*V(S)
    as_pairs=env.get_as_pairs()
    if len(as_pairs)==0:
        return None,map_side_to_int(env.get_turn())*V(F(env.current_state))
    if depth==1:
        S=np.array([F(s) for (a,s) in as_pairs])
        S=np.reshape(S,(S.shape[0],S.shape[-1]))
        values=map_side_to_int(env.get_turn())*V(S)
        index=np.argmax(values)
        return as_pairs[index][0],values[index,0]
    else:
        act=None
        for (a,s) in as_pairs: 
            env=Environment(s)
            score=-alphabeta_batch(V,F,env,depth-1,-beta,-alpha)[1]
            if score>=beta:
                return a,beta
            elif score>alpha:
                alpha=score
                act=a
        return act, alpha 

def alphabeta_dtm(sp,a,s,depth,alpha,beta):
    """
    alpha beta pruning on a ground truth dtm 
    params:
        sp: previous state
        a: action
        s: current state
        depth: depth of search
        alpha
        beta
    returns:
        max_a: best action
        max_score: score of this action
    """
    if depth==0:
        ep=Environment(sp)
        return None,-map_side_to_int(ep.get_turn())*ep.action_outcome(a)
    elif depth==1:
        e=Environment(s)
        as_pairs=e.get_as_pairs()
        outcomes=[-0.5*map_side_to_int(e.get_turn())*e.action_outcome(an) for
                 (an,sn) in as_pairs]
        max_o=max(outcomes)
        rand=outcomes.index(max_o)
        return as_pairs[rand][0],max_o
    else:
        best_an=None
        e=Environment(s)
        as_pairs=e.get_as_pairs()
        for (an,sn) in as_pairs:
            score=-0.5*alphabeta_dtm(s,an,sn,depth-1,-beta,-alpha)[1]
            if score>=beta:
                return an, beta
            elif score>alpha:
                alpha=score
                best_an=an
        return best_an,alpha

def alphabeta_outcome(sp,a,s,depth,alpha,beta):
    """
    alpha beta pruning on a ground truth outcome 
    params:
        sp: previous state
        a: action
        s: current state
        depth: depth of search
        alpha
        beta
    returns:
        max_a: best action
        max_score: score of this action
    """
    if depth<1:
        env=Environment(sp)
        env.perform_action(a)
        o=map_side_to_int(env.get_turn())*env.int_outcome()
        #print o
        return None,o
    env=Environment(s)
    as_pairs=env.get_as_pairs()
    if len(as_pairs)==0:
        env=Environment(sp)
        env.perform_action(a)
        o=map_side_to_int(env.get_turn())*env.int_outcome()
        #print o
        return None,o
    if depth==1:
        outcomes=[0.5*map_side_to_int(env.get_turn())*env.action_outcome(a) for
                  (a,sn) in as_pairs]
        best=np.argmax(np.array(outcomes))
        best_o=outcomes[best]
        return as_pairs[best][0],best_o
    act=None
    for (a,sn) in as_pairs:
        score=-0.5*alphabeta_outcome(s,a,sn,depth-1,-beta,-alpha)[1]
        if score>=beta:
            return a, beta
        elif score>alpha:
            alpha=score
            act=a
    return act,alpha

def alphabeta_batch_hist(V,F,env,hist,depth,alpha,beta):
    """alpha_beta_batch with added memory (dynamic programming) 
    params:
        hist: history of observed states
    """
    if depth<1:
        S=F(env.current_state)
        return None, map_side_to_int(env.get_turn())*V(S)
    as_pairs=env.get_as_pairs()
    if len(as_pairs)==0:
        return None,map_side_to_int(env.get_turn())*V(F(env.current_state))
    # avoid repetition
    as_pairs=[(a,s) for (a,s) in as_pairs if s not in hist]
    if len(as_pairs)==0:
        as_pairs=env.get_as_pairs()
    if depth==1:
        S=np.array([F(s) for (a,s) in as_pairs if s ])
        S=np.reshape(S,(S.shape[0],S.shape[-1]))
        values=map_side_to_int(env.get_turn())*V(S)
        index=np.argmax(values)
        return as_pairs[index][0],values[index,0]
    else:
        act=None
        for (a,s) in as_pairs: 
            env=Environment(s)
            score=-alphabeta_batch_hist(V,F,env,hist+[s],depth-1,-beta,-alpha)[1]
            if score>=beta:
                return a,beta
            elif score>alpha:
                alpha=score
                act=a
        return act, alpha 

def alphabeta_batch_hist_leaf(V,F,env,hist,depth,alpha,beta):
    if depth<1:
        S=F(env.current_state)
        return None, map_side_to_int(env.get_turn())*V(S),env.current_state
    as_pairs=env.get_as_pairs()
    if len(as_pairs)==0:
        return None,map_side_to_int(env.get_turn())*V(F(env.current_state)),env.current_state
    # avoid repetition
    as_pairs=[(a,s) for (a,s) in as_pairs if s not in hist]
    if len(as_pairs)==0:
        as_pairs=env.get_as_pairs()
    if depth==1:
        S=np.array([F(s) for (a,s) in as_pairs if s ])
        S=np.reshape(S,(S.shape[0],S.shape[-1]))
        values=map_side_to_int(env.get_turn())*V(S)
        index=np.argmax(values)
        #env.draw()
        #a0,s0=minimax(V,F,env,1)
        #assert np.abs(values[index,0]-s0)<0.0001
        return as_pairs[index][0],values[index,0],as_pairs[index][1]
    else:
        act=None
        best_leaf=None
        for (a,s) in as_pairs: 
            env=Environment(s)
            _,score,leaf=alphabeta_batch_hist_leaf(V,F,env,hist,depth-1,-beta,-alpha)
            score=-score
            if score>=beta:
                return a,beta,leaf
            elif score>alpha:
                alpha=score
                act=a
                best_leaf=leaf
        return act, alpha,best_leaf 

def trans_add_entry(table,s,d,sc,mv):
    table['state']={'depth':d,'score':sc,'move':mv}

pieces=['P','R','B','N','Q','K','p','r','b','n','q','k']
PIECE_MAP=dict()
for i in xrange(len(pieces)):
    PIECE_MAP[pieces[i]]=i

def zobrist_array():
    # deprecated: not part of chess v22
    import chess
    #return chess.POLYGLOT_RANDOM_ARRAY
    return []

ZOB=zobrist_array()

def zobrist(s):
   '''
   TODO: ep and castle
   '''
   b =(s.split()[0]).split('/')
   z=None
   indexes=[]
   for i in xrange(len(b)):
       f=0
       for j in xrange(len(b[i])):
           if ord('0')<=ord(b[i][j])<=ord('9'):
               f+=int(b[i][j])
           else:
               p=b[i][j]
               ind=PIECE_MAP[p]*64+i*8+f
               indexes.append(ind)
               if z is None:
                   z=ZOB[ind]
               else:
                   z^=ZOB[ind]
               f+=1
   if s.split()[1]=='w':
       z^=ZOB[-1]
   print indexes
   return z

def new_zobrist(z,s,a):
    '''
    different kind of xoring for different kind of moves
    '''
    (sq1,sq2)=uci_to_squares(a)
    #print a
    #print sq1,sq2
    Environment(s).draw()
    p1=get_piece(s,sq1)
    p2=get_piece(s,sq2)
    #print p1,p2

    ind1=PIECE_MAP[p1]*64+sq1[0]*8+sq1[1]
    ind2=PIECE_MAP[p1]*64+sq2[0]*8+sq2[1]
    indexes=[ind1,ind2]
    if p2 is not None:
        rem=PIECE_MAP[p2]*64+sq2[0]*8+sq2[1]
        indexes.append(rem)

    z2=z^ZOB[-1]^ZOB[ind1]^ZOB[ind2]
    if p2 is not None:
        z2^=ZOB[rem]
    #print indexes
    return z2

def alphabeta_zobtrans(V,F,trans,env,z,depth,alpha,beta):
    """some doodling around with a self written zobrist hash function, did not
    perform as good as with the python hash function for dictionaries"""
    as_pairs=env.get_as_pairs()
    st=env.current_state
    if len(as_pairs)==0:
        return None,map_side_to_int(env.get_turn())*V(F(env.current_state))
    if z in trans:
        if trans[z]['depth']>=depth:
            return trans[z]['move'],trans[z]['score']
        else:
            "change order of lookup in favour of pv"
            ind=[a for (a,s) in as_pairs].index(trans[z]['move'])
            as_pairs[0], as_pairs[ind]=as_pairs[ind],as_pairs[0]
    if depth==1:
        S=np.array([F(s) for (a,s) in as_pairs])
        S=np.reshape(S,(S.shape[0],S.shape[-1]))
        values=map_side_to_int(env.get_turn())*V(S)
        index=np.argmax(values)
        #a0,s0=minimax(V,F,env,1)
        #assert np.abs(values[index,0]-s0)<0.0001
        trans_add_entry(trans,z,depth,values[index,0],as_pairs[index][0])
        return as_pairs[index][0],values[index,0]
    else:
        act=None
        for (a,s) in as_pairs: 
            zn=new_zobrist(z,st,a)
            env=Environment(s)
            score=-alphabeta_zobtrans(V,F,trans,env,zn,depth-1,-beta,-alpha)[1]
            if score>=beta:
                return a,beta
            elif score>alpha:
                alpha=score
                act=a
        trans_add_entry(trans,z,depth,alpha,act)
        return act, alpha 

def alphabeta_trans(V,F,trans,env,depth,alpha,beta):
    as_pairs=env.get_as_pairs()
    if len(as_pairs)==0:
        return None,map_side_to_int(env.get_turn())*V(F(env.current_state))
    if env.current_state in trans:
        s=env.current_state
        if trans[s]['depth']>=depth:
            return trans[s]['move'],trans[s]['score']
        else:
            "change order of lookup in favour of pv"
            ind=[a for (a,s) in as_pairs].index(trans[s]['move'])
            as_pairs[0], as_pairs[ind]=as_pairs[ind],as_pairs[0]
    if depth==1:
        S=np.array([F(s) for (a,s) in as_pairs])
        S=np.reshape(S,(S.shape[0],S.shape[-1]))
        values=map_side_to_int(env.get_turn())*V(S)
        index=np.argmax(values)
        #a0,s0=minimax(V,F,env,1)
        #assert np.abs(values[index,0]-s0)<0.0001
        trans_add_entry(trans,env.current_state,depth,values[index,0],as_pairs[index][0])
        return as_pairs[index][0],values[index,0]
    else:
        act=None
        for (a,s) in as_pairs: 
            env=Environment(s)
            score=-alphabeta_trans(V,F,trans,env,depth-1,-beta,-alpha)[1]
            if score>=beta:
                trans_add_entry(trans,env.current_state,depth,beta,a)
                return a,beta
            elif score>alpha:
                alpha=score
                act=a
        trans_add_entry(trans,env.current_state,depth,alpha,act)
        return act, alpha 

def test():
    from approximator import Approximator 
    import time
    import tensorflow as tf
    from learn.preprocessing import faster_featurize
    env=Environment(draw_r=-1,move_r=0.001)
    env.reset() 
    model_fn='Models/DeepTDy_m8_krk_3-4_cont__07_05/DeepTDy_m8_krk_3-4_cont__07_05-0_0173614-0'
    with tf.Session() as sess:
        saver=tf.train.import_meta_graph(model_fn+'.meta')
        saver.restore(sess,model_fn)
        approx=Approximator(sess)
        V=approx.value
        F=faster_featurize
        flag=False
        mv_cnt=0
        time1=0
        time2=0
        trans=dict()
        while not flag:
            env.draw()
            print env.hist
            print '\n'

            if env.is_terminal():
                print env.result()
                flag=True

            else:
                start=time.time()
                a,score=alphabeta_batch(V,F,env,3,-float('inf'),float('inf'))
                end=time.time()
                a2,score2,leaf=alphabeta_batch_hist_leaf(V,F,env,list(env.hist.keys()),3,-float('inf'),float('inf'))
                end2=time.time()
                #assert np.abs(score-score2)<0.001
                env.perform_action(a2)
                time1+=end-start
                time2+=end2-end
                mv_cnt+=1
                print('\nLeaf:')
                Environment(state=leaf).draw()

        print('AB-Minimax Batch: {}\tAB-Minimax hist:{}'.format(time1/mv_cnt,time2/mv_cnt))

def speed_test():
    from approximator import Approximator 
    import time
    import tensorflow as tf
    from learn.preprocessing import faster_featurize
    import settings
    from environment import load_DS

    settings.init()
    load_DS('dataset/krk.epd')
    settings.params['PL']=list('KRkr')
    model_fn='Models/stem_leaf/TDLeaf/TDLeaf_stem_or_leaf_7__03_07/TDLeaf_stem_or_leaf_7__03_07-1_13299-0'
    with tf.Session() as sess:
        saver=tf.train.import_meta_graph(model_fn+'.meta')
        saver.restore(sess,model_fn)
        approx=Approximator(sess)
        V=approx.value
        F=faster_featurize

        avg_time1=0
        avg_time2=0
        avg_time3=0

        for _ in xrange(20):
            env=Environment()

            flag=False
            mv_cnt=0
            time1=0
            time2=0
            time3=0
            while not flag:
                if env.is_terminal():
                    flag=True

                else:
                    start=time.time()
                    a,score=alphabeta_native(V,F,env,3,-float('inf'),float('inf'))
                    end=time.time()
                    a2,score2=alphabeta_batch_hist(V,F,env,list(env.hist.keys()),3,-float('inf'),float('inf'))
                    end2=time.time()
                    a3,score=alphabeta_batch(V,F,env,3,-float('inf'),float('inf'))
                    end3=time.time()
                    env.perform_action(a)
                    time1+=end-start
                    time2+=end2-end
                    time3+=end3-end2
                    mv_cnt+=1

            avg_time1+=time1/mv_cnt
            avg_time2+=time2/mv_cnt
            avg_time3+=time3/mv_cnt

        print avg_time1/100, avg_time2/100, avg_time3/100

def test_outcome():
    s0p='5K2/7k/8/2Q5/8/8/8/8 w - -'
    u='c5h5'
    s0='5K2/7k/8/7Q/8/8/8/8 b - -'
    e0=Environment(s0)
    e0.draw()
    print alphabeta_outcome(s0p,u,s0,0,-float('inf'),float('inf'))

    s1=s0p
    e1=Environment(s1)
    e1.draw()
    print alphabeta_outcome(None,None,s1,1,-float('inf'),float('inf'))

    s2='7k/1Q3K2/8/8/8/8/8/8 b - -'
    e2=Environment(s2)
    e2.draw()
    print alphabeta_outcome(None,None,s2,2,-float('inf'),float('inf'))

    s3='8/8/8/5K1k/8/Q7/8/8 b - -'
    e3=Environment(s3)
    e3.draw()
    print alphabeta_outcome(None,None,s3,2,-float('inf'),float('inf'))

    s4='8/8/8/Q4K1k/8/8/8/8 w - -'
    e4=Environment(s4)
    e4.draw()
    print alphabeta_outcome(None,None,s4,3,-float('inf'),float('inf'))
    e4.perform_action('f5f6')
    e4.draw()
    print alphabeta_outcome(None,None,e4.current_state,2,-float('inf'),float('inf'))

if __name__=='__main__':
    speed_test()
