"""
Module with network implementations for learning
"""

import tensorflow as tf
from util import unison_shuffle
import numpy as np
import os

"""
useful variables
"""
HERE=os.path.abspath(os.path.dirname(__file__))
MODEL_DIR=os.path.join(HERE,'models')
TB_PATH='tensorboard'
META='.meta'

def create_modelname(err):
    """
    way to incorporate the score of a model into a string for a filename
    """
    s=str(err)
    return str(int(err))+'_'+s[2:min([9,len(s)])]

def model_path(fn):
    return os.path.join(MODEL_DIR,fn)

def meta_path(fn):
    return os.path.join(MODEL_DIR,fn+META)

def restore_session(saver,model):
    """
    Params:
        saver: tf.Saver(), to rebuild old session with tensorflow graph and
        operations.
    """
    sess=tf.Session()
    saver.restore(sess,os.path.join(MODEL_DIR,model))
    return sess 

def data_preparation(X,Y,B,split=0.8): 
    """
    transform data into training and test set 
    params:
        X: features
        Y: values
        B: chess boards (states)
    return:
        X_train,Y_train,B_train,X_test,Y_test,B_test
    """
    # split the data in test and training set
    split=int(split*X.shape[0])
    Y=np.array(Y).reshape((X.shape[0],1))
    X_train=X[:split,:]
    X_test=X[split:,:]
    Y_train=Y[:split,:]
    Y_test=Y[split:,:]
    B_train=B[:split]
    B_test=B[split:]
    return X_train,Y_train,B_train,X_test,Y_test,B_test

def fit(sess,X_train,Y_train,B_train,X_test,Y_test,B_test,fp='models/my-model',init=False,split=0.8,epochs=10000,batch_sz=250,save_period=10,device='/cpu:0',saver=None):
    """
    General method for networks in module so they can be fitted to the data.
    Done with Batch SGD with Momentum
    General use:
        if no checkpoint file is available, give in a session (with graph)
        if ckpt file is available, it is sufficient to give filename in the
        models folder, so learning can be continued.
    """


    with sess.graph.as_default():
        X_pl=tf.get_collection('X_pl')[0]
        Y_pl=tf.get_collection('Y_pl')[0]


        if saver is None: 
            saver=tf.train.Saver()
        
        N=X_train.shape[0]
        best_loss=5000000
        best_it=0
        batches=int(N/batch_sz)
        best_f=''

        if init:
            sess.run(tf.initialize_all_variables())

        for i in xrange(epochs):
            x,y,b=unison_shuffle(X_train,Y_train,B_train)
            
            X_pl=tf.get_collection('X_pl')[0]

            for j in xrange(batches):
                s=(j+1)*batch_sz if j+1<batches else N
                xbatch=x[j*batch_sz:s]
                ybatch=y[j*batch_sz:s]
                feed_dict={X_pl: xbatch, Y_pl: ybatch}
                train_op=tf.get_collection('train_op')[0]
                sess.run(train_op,feed_dict)

            if i%save_period==0:
                train_feed={X_pl: X_train,Y_pl: Y_train}
                test_feed={X_pl: X_test,Y_pl: Y_test}
                error=tf.get_collection('error')[0]
                avg_err=tf.get_collection('avg_error')[0]

                train_eval=sess.run([error,avg_err],train_feed)
                test_eval=sess.run([error,avg_err],test_feed)
                print i, "I: CKPT: Training: {}\tTest:{}".format(train_eval[1],test_eval[1])

                if np.abs(best_loss-test_eval[1])<0.0001:
                    break

                if test_eval[1]<best_loss:
                    best_loss=test_eval[1]
                    best_it=i
                    fn=create_modelname(best_loss)                    
                    best_f=saver.save(sess,fp+'-'+fn,global_step=i)

                else:
                    break

                if i-best_it>10*save_period:
                    print "Final test score: ",best_loss
                    break


        print best_f 
        return best_f.split('/')[-1]
