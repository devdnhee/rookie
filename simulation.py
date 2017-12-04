import sys
from subprocess import call
import os
from environment import Environment
import learn.ann as ann

def sim1():
    env=Environment(draw_r=-1,move_r=-0.001)
    env.reset(state='7k/8/8/4K3/8/8/1R6/8 w - -')
    s='7k/8/8/4K3/8/8/1R6/8 w - -'
    f=faster_featurize(s)
    #f=featurize(chess.Board.from_epd(s)[0])
    D=f.shape[1]
    pol=EpsilonGreedyPolicy(eps=1,decay_rate=0.00,decay_period=100)

    M=[64,128,64]
    M2=[64]
    c0=8
    F=[(1,1),(2,2),(3,3),(3,3)]
    C=[(8,),(8,),(12,),(16,)]

    g1=cnn.build_graph(F,C,c0,M2,D,device='/gpu:1',learning_rate=3e-7)
    g2=ann.build_graph(M,D,device='/gpu:1',learning_rate=1e-4)


    #approx=DeepTDLambda(l=0.5,y=0.7,epochs=1000,name='cnn',batch_sz=125,save_period=100,max_buffer=100000)
    #sv1=Supervisor(env,pol,approx,'cnn.sv',mv_limit=33,store_period=100,sample_period=10)
    #sv1.run(10,500,g1,device='/gpu:1',state='7k/8/8/4K3/8/8/1R6/8 w - -')

    approx=DeepTDLambda(l=0.5,y=0.7,name='ann',epochs=1000,batch_sz=125,save_period=100,max_buffer=100000)
    sv2=Supervisor(env,pol,approx,'ann.sv',mv_limit=33,store_period=100,sample_period=10)
    sv2.run(10,500,g2,device='/gpu:1',state='7k/8/8/4K3/8/8/1R6/8 w - -')

def run_models():
    models=[[64],[64,128,64],[128],[32],[32,64,128,64,32]]
    for m in models:
        n='m8in3_{}_{}'.format(len(m),m[0])
        string=' '.join([str(i) for i in models])
        os.system('python deep_TD_y.py -s \'7k/8/5K2/8/2R5/8/8/8 w - -\' -M'
                +string+' -I 10 -N 250 -n '+n)

if __name__=='__main__':
    run_models()
        
