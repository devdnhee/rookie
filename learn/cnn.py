import cPickle as pickle
import numpy as np
from util import init_weigths, channelize, unison_shuffle
import tensorflow as tf

def build_graph(**kwargs):
    """
    building a CNN from scratch with tensorflow.
    Arguments:
        F: filter dimensions between the layers
        C: list of i/o dimensions 
        c0: input  dimensions 
        device
        non_lin
        learn_rate
        momentum
    """

    M=kwargs['M']
    #D=kwargs['D']
    F=kwargs['F']
    C=kwargs['C']
    D=kwargs['D']
    c0=kwargs['c0']
    device=kwargs.get('device','/cpu:0')
    non_lin=kwargs.get('non_lin',tf.nn.relu)
    learning_rate=kwargs.get('learning_rate',0.00001)
    momentum=kwargs.get('momentum',0.9)
    D=8*8*C[-1][0]+(D%(8*8*c0))

    with tf.device(device):

	g=tf.Graph()

	with g.as_default():
	    # placeholders
	    X_pl=tf.placeholder(tf.float32,[None,None],name='x-input')
	    tf.add_to_collection('X_pl',X_pl)
	    Y_pl=tf.placeholder(tf.float32,[None,None],name='y-input')
	    tf.add_to_collection('Y_pl',Y_pl)

            
            N=tf.shape(X_pl)[0]

	    # Variables FCN
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


            # Variables CNN
            Wconv=[init_weigths((F[0]+(c0,)+C[0]),name='Wconv0')]
            bconv=[0*init_weigths((C[0]),name='bconv0')]
	    g.add_to_collection('vars',Wconv[0])
	    g.add_to_collection('vars',bconv[0])
            for i in xrange(len(F)-1):
                Wconv.append(init_weigths((F[i+1]+C[i]+C[i+1]),name='Wconv'+str(i+1)))
                bconv.append(0*init_weigths((C[i+1]),name='bconv'+str(i+1)))
                g.add_to_collection('vars',Wconv[-1])
                g.add_to_collection('vars',bconv[-1])

            print [v.name for v in g.get_collection('vars')]

            # Channelization
            Xim, Xni=channelize(X_pl,(8,8,c0))
                
            # Forward propagation cnn
            with tf.name_scope('CNN'):
                Z=Xim
                for i in xrange(len(Wconv)):
                    with tf.name_scope("Convlayer_"+str(i+1)):
                        Z=tf.nn.conv2d(Z,Wconv[i],strides=[1,1,1,1],padding='SAME')
                        Z=tf.nn.bias_add(Z,bconv[i])


            s=tf.shape(Wconv[-1])[3]
            Xf=tf.reshape(Z,[N,8*8*s])
            Xflat=tf.concat([Xf,Xni],1)

            g.add_to_collection('s',s)
            g.add_to_collection('Z',Z)
            g.add_to_collection('Xf',Xf)
            g.add_to_collection('Xflat',Xflat)


	    # Forward propagation neural network 
	    with tf.name_scope('ANN'):
		Z=Xflat
		for i in xrange(len(W)-1):
		    with tf.name_scope("layer_"+str(i+1)):
			Z=non_lin(tf.matmul(Z,W[i])+tf.transpose(b[i]))
		with tf.name_scope('output_node'):
		    forward=tf.matmul(Z,W[-1])+tf.transpose(b[-1])
	    
	    tf.add_to_collection('forward',forward)

	    # Normalization
	    mu, sigma2=tf.nn.moments(Y_pl,axes=[0])
	    sigma=tf.sqrt(sigma2)
	    tf.add_to_collection('sigma',sigma)

	    with tf.name_scope('NORMALIZATION'):
		Y_norm=tf.div(Y_pl-mu,sigma)
		
	    # Value function
	    with tf.name_scope('VALUE'):
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

    return g 

def test():
    import preprocessing as pp
    X,Y,Yd,B=pp.read_data('krk_data_20000_balanced_8.cpkl')
    ind_nd=np.where(Yd==0)[0]
    Ydtm=Y[ind_nd]
    Xdtm=np.zeros((Ydtm.shape[0],X.shape[1]))
    Bdtm=[]
    for i in xrange(len(ind_nd)):
        Xdtm[i,:]=X[ind_nd[i]]
        Bdtm.append(B[ind_nd[i]])
    N=Xdtm.shape[0]
    print "Ydtm:{}\tXdtm{}".format(Ydtm.shape,Xdtm.shape)

    """
    TODO: delete this for loop
    """
    print "data read without splitting"
    for i in xrange(5):
        print B[i], Y[i] 

    N=len(Y)
    Y=np.array(Y).reshape((N,1))
    Yd=np.array(Yd).reshape((N,1))

    D=X.shape[1]
    print D
    M=[128]
    c0=8
    F=[(1,1),(3,3),(3,3)]
    C=[(8,),(16,),(32,)]

    import learn
    # first time use:
    graph=build_graph(F,C,c0,M,D)
    with tf.Session(graph=graph) as sess:
        learn.fit(sess,Xdtm,Ydtm,Bdtm,init=True)
        

    # after checkpoints:
    #saver=tf.train.import_meta_graph('models/my-model-4_455-800.meta')
    #sess=learn.restore_session(saver,'my-model-4_455-800')
    #learn.fit(sess,Xdtm,Ydtm,Bdtm,init=False,saver=saver)



if __name__=='__main__':
    test()
