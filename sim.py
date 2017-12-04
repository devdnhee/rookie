"""script running a simulation

for use:
    python sim.py -h
"""

import chess
import settings
import search
from supervisor import Supervisor
import sys
import multiprocessing as mp 
from chess_utils import map_side_to_int
from supervisor import Supervisor
from environment import Environment,load_DS
import learn.ann as ann
import learn.cnn as cnn
from data_mgmt import DataManager
import tensorflow as tf
import numpy as np
from learn.preprocessing import faster_featurize,change_PL
from policy import EpsilonGreedyPolicy,GreedyPolicy
from agent import Agent
import deep_TD_y as tdy
import td_leaf as tdleaf
import td_stem as tdstem

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
    parser.add_argument('--alpha',default=1e-4,help='learning rate',type=float)
    parser.add_argument('-f','--eps-factor',default=0.75,help='epsilon decay',type=float)
    parser.add_argument('-y','--gamma',default=1,help='discount factor',type=float)
    parser.add_argument('--lambd',default=0.7,help='lambda',type=float)
    parser.add_argument('--eps-start',default=0,help='epsilon decay',type=int)
    parser.add_argument('--dset',action='store_true')
    parser.add_argument('--cnn',action='store_true') 
    parser.add_argument('--ds-file',default=None)
    parser.add_argument('--piece-conf',default='KRkr')
    parser.add_argument('--mode',default='stem')
    parser.add_argument('-R',default=2,type=int)
    parser.add_argument('--mk',type=int)
    parser.add_argument('--ocd',default=5,type=int) 
    parser.add_argument('--cnn-f',action='store_true') 
    args=parser.parse_args()

    settings.init()
    settings.params['USE_DSET']=args.dset
    settings.params['OC_DEPTH']=args.ocd
    settings.params['MK']=args.mk
    change_PL(list(args.piece_conf))
    settings.params['RAND']=args.R
    print settings.params['PL']
    if settings.params['USE_DSET']:
        load_DS(args.ds_file)
    if args.eps_factor==0:
        decay_f=lambda n: 1-0.01*n
    else:
        decay_f=lambda n:(n+1)**(-args.eps_factor)

    if args.old_model==None:
        D=faster_featurize('8/6k1/8/8/3K4/8/8/8 w - -').shape[1]
        print D
        M=[int(m) for m in args.M.split()]
        if args.cnn:
            c0=len(args.piece_conf)*2
            if args.cnn_f: 
                kwargs={'D': D, 'M':M ,
                        'learning_rate':args.alpha,'F':[(1,1),(2,2),(3,3),(3,3)],
                        'c0':c0,'C':[(c0,),(c0,),(c0,),(2*c0,)]
                       }
            else:
                kwargs={'D': D, 'M':M ,
                        'learning_rate':args.alpha,'F':[(1,1)],'c0':c0,'C':[(c0,)]}
            graph_f=cnn.build_graph
        else: 
            kwargs={'D': D, 'M': M, 'learning_rate':args.alpha}
            graph_f=ann.build_graph
        pol=EpsilonGreedyPolicy(eps=1.0,decay_f=decay_f)
        pol.n=args.eps_start
        if args.mode=='leaf':
            sv=tdleaf.TDLeafSupervisor(pol,mv_limit=args.move_count,depth=args.depth,y=args.gamma,l=args.lambd)
        else:
            sv=tdstem.TDStemSupervisor(pol,mv_limit=args.move_count,depth=args.depth,y=args.gamma,l=args.lambd)
        sv.run(args.I,args.N,graph_f,kwargs,state=args.state,name=args.name)
    else: 
        assert args.ckpt is not None
        pol=EpsilonGreedyPolicy(eps=0.0,decay_f=decay_f)
        pol.n=args.eps_start
        if args.mode=='leaf':
            sv=tdleaf.TDLeafSupervisor(pol,mv_limit=args.move_count,depth=args.depth,y=args.gamma,l=args.lambd)
        else:
            sv=tdstem.TDStemSupervisor(pol,mv_limit=args.move_count,depth=args.depth,y=args.gamma,l=args.lambd)
        sv.retrieve(args.old_model) 
        sv.continue_run(args.I,args.ckpt,name=args.name,N=args.N)
