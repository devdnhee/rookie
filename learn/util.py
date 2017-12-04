import numpy as np
from tensorflow.python.ops.variables import Variable
import tensorflow as tf


"""
code provided by tensorflow for summaries about several quantities
"""
def variable_summaries(var, name):
    with tf.name_scope('summaries'):
        mean=tf.reduce_mean(var)
        tf.scalar_summary('mean/'+name,mean)
        with tf.name_scope('stddev'):
            stddev=tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.scalar_summary('stddev/'+name,stddev)
        tf.scalar_summary('max/'+name,tf.reduce_max(var))
        tf.scalar_summary('min/'+name,tf.reduce_min(var))
        tf.histogram_summary(name,var)

"""
convert data to channels for the image board features, and split with non image
features
"""
def channelize(X,sh):
    with tf.name_scope('Channelization'):
        imfeat=np.prod(list(sh))
        N=tf.shape(X)[0]
        D=tf.shape(X)[1]
        Xim=tf.slice(X,[0,0],[N,imfeat])
        Xni=tf.slice(X,[0,imfeat],[N,D-imfeat])
        Xim=tf.reshape(Xim,(N,)+sh)
        return Xim, Xni

"""
initialize the weights for a tensor
"""
def init_weigths(shape,name):
    return tf.Variable(0.1*np.sqrt(2.0/shape[0])*tf.random_normal(shape),dtype=tf.float32,name=name)

"""
shuffle three arrays at according to each other, for data shuffling
"""
def unison_shuffle(x,y,b,pr=False):
    p=np.random.permutation(x.shape[0])
    b_p=[b[i] for i in p]
    if pr:
        print p
    return x[p],y[p],b_p

def unison_shuffle_n(A):
    p=np.random.permutation(A[0].shape[0])
    return [x[p] for x in A] 
