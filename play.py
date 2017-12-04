"""
script making possible to play against an engine
use: python play.py -h
"""

from __future__ import absolute_import
import time
import numpy as np
import tensorflow as tf
import sys
import cPickle
import chess
import time
from chess import pgn
import os
import chess.pgn as pgn
from environment import Environment
import deep_TD_y as tdy 
from policy import GreedyPolicy
from learn.preprocessing import faster_featurize
import learn.ann as ann
from approximator import Approximator
import learn
import optimal as opt
import search
import td_leaf as tdleaf
import td_stem as tdstem
import settings

def play(model_fn, color, start_board=None, sim=False, depth=3):

    env=Environment(start_board)
    pol=GreedyPolicy()
    with tf.Session() as sess:
        if model_fn is not None: 
            saver=tf.train.import_meta_graph(model_fn+'.meta')
            saver.restore(sess,model_fn)

        approx=Approximator(sess)
        a=[None,None]
        if sim:
            a[int(color)]=tdstem.TDStemPlayAgent(approx,depth=depth)
            a[int(not color)]=opt.OptimalAgent()
            
        else:
            a[int(not color)]=tdstem.TDStemPlayAgent(approx,depth=depth)
            a[int(color)]=tdstem.TDStemPlayAgent(approx,depth=depth)

        oa=opt.OptimalAgent()

        flag=False
        
        name=str(raw_input("What's your name? "))
        print "Let's play a game, %s!" %(str(name))

        while not flag:
            time.sleep(2)

            env.draw() 
            print 'DTM: {}'.format(np.abs(oa.approx.tb.probe_dtm(chess.Board.from_epd(env.current_state)[0])))

            if env.is_game_over():
                print env.result()
                flag=True

            else:
                print 'Evaluation: {}'.format(a[int(color)].get_av_pairs(env))
                print 'Optimal moves: {}'.format(oa.get_best_moves(env))
                start=time.time()

                if env.get_turn()==color:
                    if sim:
                        a[int(color)].play(env)
                    else:
                        suc=False
                        while not suc:
                            m=str(raw_input('YOUR MOVE: '))
                            try:
                                env.perform_action(m)
                                suc=True
                            except:
                                raise ValueError

                else:
                    a[int(not color)].play(env)

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('-m','--model',default='Models/krk/TDS/7/network',help="""Path
                        to the used approximation function (tensorflow model)""")
    parser.add_argument('-s','--simulation',action='store_true',help="""
                       setting this flag turns on simulation mode, where the
                        model is playing against the optimal player""")
    parser.add_argument('-d','--depth',type=int,default=3,help="""
                       the depth of the minimax search of the engine""")
    parser.add_argument('-c','--color',default='w',help="""
                       side you choose to play with. In simulation mode this is
                        the side of the model""")
    parser.add_argument('-p','--position',default='8/5k2/3KR3/8/8/8/8/8 w - -',help="""
                       the starting position (epd string)""")
    parser.add_argument('--cfg',default='KRkr',help="""
                       indicating the specific problem, important to know the
                        number of input planes to the network""")
    args=parser.parse_args()

    settings.init()
    settings.params['PL']=list(args.cfg)
    color=(args.color=='w')

    play(model_fn=args.model,color=color,start_board=args.position,sim=args.simulation,depth=args.depth)
