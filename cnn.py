
import cPickle as pickle
import numpy as np
from util import init_weigths
import tensorflow as tf
import ann


def build_graph(F,C,c0,M,D,device='/cpu:0',non_lin=tf.nn.relu,learning_rate=10e-5,momentum=0.9):

	g=tf.Graph()

	with g.as_default():
	    # placeholders
	    X_pl=tf.placeholder(tf.float32,[None,None],name='x-input')
	    tf.add_to_collection('X_pl',X_pl)
	    Y_pl=tf.placeholder(tf.float32,[None,None],name='y-input')
	    tf.add_to_collection('Y_pl',Y_pl)

	    # Variables
        # Convolution
        Wconv=[init_weights((F[0]+(c0,)+C[0]),name='Wconv0')]
        bconv=[0*init_weights((C[0]),name='bconv0')]
	    g.add_to_collection('vars',Wconv[0])
	    g.add_to_collection('vars',bconv[0])
        for i in xrange(len(F)-1):
            Wconv.append(init_weights((F[i+1]+C[i]+C[i+1]),name='Wconv'+str(i+1)))
            bconv.append(0*init_weights((C[i+1],name='bconv'+str(i+1))))
            g.add_to_collection('vars',W[-1])
            g.add_to_collection('vars',b[-1])

        # fully connected
	    W=[init_weigths([D,M[0]],name='W1')]
	    b=[init_weigths([M[0],1],name='b1')]
	    g.add_to_collection('vars',W[0])
	    g.add_to_collection('vars',b[0])
	    for i in xrange(len(M)-1):
            W.append(init_weigths([M[i],M[i+1]],name='W'+str(i+2)))
            b.append(init_weigths([M[i+1],1],name='b'+str(i+2)))
            g.add_to_collection('vars',W[-1])
            g.add_to_collection('vars',b[-1])
	    W.append(init_weigths([M[-1],1],name='W'+str(len(M)+1)))
	    b.append(init_weigths([1,1],name='b'+str(len(M)+1)))
	    g.add_to_collection('vars',W[-1])
	    g.add_to_collection('vars',b[-1])

        # 
	    
	    # Forward propagation ANN 
	    with tf.name_scope('ANN'):
            #TODO: INPUT Z Z=X_pl
            for i in xrange(len(W)-1):
                with tf.name_scope("ANNLayer_"+str(i+1)):
                Z=non_lin(tf.matmul(Z,W[i])+tf.transpose(b[i]))
            with tf.name_scope('output_node'):
                forward_ann=tf.matmul(Z,W[-1])+tf.transpose(b[-1])
	    
	    tf.add_to_collection('forward_ann',forward_ann)

	    # Normalization
	    mu, sigma2=tf.nn.moments(Y_pl,axes=[0])
	    sigma=tf.sqrt(sigma2)
	    tf.add_to_collection('sigma',sigma)

	    with tf.name_scope('NORMALIZATION'):
		Y_norm=tf.div(Y_pl-mu,sigma)
		
	    # Value function
	    with tf.name_scope('VALUE'):
		#value=sigma*forward+mu
		value=forward
	    tf.add_to_collection('value',value)
		
	    # Loss function
	    with tf.name_scope('LOSS'):
		loss=tf.reduce_mean(tf.square(Y_pl-forward))
	    tf.add_to_collection('loss',loss)
		
	    # Error
	    with tf.name_scope('ERROR'):
		error=sigma*tf.abs(Y_pl-forward)
		avg_error=tf.reduce_mean(error)           
	    tf.add_to_collection('avg_error',avg_error)
	    tf.add_to_collection('error',error)

	    with tf.name_scope('TRAINING'):
		train_op=tf.train.MomentumOptimizer(learning_rate=learning_rate,momentum=momentum).minimize(loss)
	    tf.add_to_collection('train_op',train_op)

    
    
