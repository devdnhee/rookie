"""script of a simulation with an optimal agent to evaluate models and store
the outcome.

use:
    python optimal_play.py -h

"""

import optimal as opt
from approximator import Approximator
import cPickle as cp
import td_stem as tdstem
import td_leaf as tdleaf
import settings
import tensorflow as tf
from environment import load_DS

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('o',help='output file')
    parser.add_argument('c',help='old ckpt with model to play with')
    parser.add_argument('-N',default=100,help='number of episodes to play',type=int)
    parser.add_argument('-p',help='piece cfg')
    parser.add_argument('-D',help='dset file')
    parser.add_argument('-R',default=10,type=int,help='number of random moves to play before registration')
    parser.add_argument('-d',default=3,type=int,help='depth')
    parser.add_argument('-w',action='store_true')

    args=parser.parse_args()

    settings.init()
    settings.params['USE_DSET']=True
    settings.params['PL']=args.p
    load_DS(args.D)
    settings.params['RAND']=args.R
    settings.params['OC_DEPTH']=args.d

    model_fn=args.c
    with tf.Session() as sess:
        saver=tf.train.import_meta_graph(model_fn+'.meta')
        saver.restore(sess,model_fn)
        approx=Approximator(sess)
        agent=tdstem.TDStemPlayAgent(approx,depth=3)
        A,evaldict,all_s=opt.recursive_eval_sim(agent,N=args.N,w=args.w)
        with open(args.o,'wb') as f:
            cp.dump((A,evaldict,all_s),f)
