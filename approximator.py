import numpy as np
from learn.preprocessing import faster_featurize
import ctypes
import learn.ann as ann
import learn
from learn.util import unison_shuffle
import tensorflow as tf
import multiprocessing as mp 

class Approximator():
    """
    wrapper for a tensorflow session containing the function approximater
    network
    """

    @staticmethod
    def V(S,model_fn):
        """Value function on a serialized tf model"""
        with tf.Session() as sess:
            saver=tf.train.import_meta_graph(model_fn+'.meta')
            saver.restore(sess,model_fn)
            approx=Approximator(sess)
            x=np.array([faster_featurize(s) for s in S])
            print x.shape
            x=np.reshape(x,(x.shape[0],x.shape[-1]))
            print x.shape
            v=approx.value(x)
        return v

    def __init__(self,session):
        self.sess=session

    def value(self,X):
        """Value function with a batch of chess positions as input"""
        with self.sess.graph.as_default():
            X_pl=tf.get_collection('X_pl')[0]
            Y_pl=tf.get_collection('Y_pl')[0]
            value_f=tf.get_collection('forward')[0]
            feed_dict={X_pl:X}
            V=self.sess.run(value_f,feed_dict=feed_dict)
        return V

    def fit(self,X,Y,X_train,Y_train,saver,fp='models/my-model',epochs=1000):
        """fitting observed data (mini-batch SGD)"""
        X,Y,_=unison_shuffle(X,Y,np.zeros(Y.shape))
        mf=learn.fit(self.sess,X,Y,np.zeros(Y.shape),X_train,Y_train,np.zeros(Y_train.shape),fp=fp,saver=saver,epochs=epochs)
        print 'I: File stored in {}'.format(mf)

        
class NetworkProcess(mp.Process):
    """
    a process taking 1 CPU for asynchronous performability which controls
    the network training and evaluation operations with the observed
    simulations. Builds up a network from scratch when initialized.

    Attributes:
        graph: tf.Graph -> NN
        fit_q: mp.Queue -> Approximator.fit tasks
        graph_f: Approximator.value function
        kwargs: network params for setup
        init_cond: mp.Condition() -> synchronizing network initialization
        fp: filepath to where model will be stored
        stop_play: mp.Event -> iteration of episodes has finished
        conns: shared mp dictionary for mp.Connection -> K: process name, V:
            mp.Connection
            used for sharing data over processes
        ep_task_q: mp.Queue -> EpisodeProcesses requesting function
            approximations
        ep_task_lock: synchronizing approximator access with mp.Lock
        fit_done: mp.Condition -> fitting is done
        play_finished: mp.Condition -> play is finished
        internal_flag: a hotfix mechanism to ensure independent train en
            validation sets for training
        X_train: training set for next fit iteration
        Y_train: function approximations for X_train
    """

    def __init__(self,fit_q,graph_f,kwargs,init_cond,fp,ep_task_q,ep_task_lock,stop_play):
        mp.Process.__init__(self)

        self.graph=graph_f(**kwargs)
        self.fit_cond=mp.Condition()
        self.fit_done=mp.Condition()
        self.fit_q=fit_q
        self.init_cond=init_cond
        self.play_finished=mp.Condition()
        self.fp=fp
        self.stop_play=stop_play
        self.conns=dict()

        self.ep_task_q=ep_task_q
        self.ep_task_lock=ep_task_lock
        
        self.internal_flag=False
        self.X_train=None
        self.Y_train=None

    def register_conn(self,proc_name,con):
        """setting up a connection for data transfer"""
        self.conns[proc_name]=con

    def fit_notify(self):
        """synchronization method"""
        self.fit_cond.acquire()
        self.fit_cond.notify_all()
        self.fit_cond.release()

    def fit_wait(self):
        """synchronization method"""
        self.fit_done.acquire()
        self.fit_done.wait()
        self.fit_done.release()

    def play_f_wait(self):
        """synchronization method"""
        self.play_finished.acquire()
        self.play_finished.wait()
        self.play_finished.release()

    def run(self):
        # random seed ensuring reproducibality
        np.random.seed()

        with tf.Session(graph=self.graph,config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
            approx=Approximator(sess)
            with sess.as_default():
                saver=tf.train.Saver()
                sess.run(tf.initialize_all_variables())

                forward=tf.get_collection('forward')[0]
                X_pl=tf.get_collection('X_pl')[0]
                error=tf.get_collection('loss')

                self.init_cond.acquire()
                self.init_cond.notify_all()
                self.init_cond.release()

                # getting example batch for checking functionality during run
                X_s=faster_featurize('6k1/8/8/5K2/8/8/8/8 w - -')            

                while True:
                    # waiting for an external signal to terminate

                    print 'BEFORE VALUE ITERATION: ',approx.value(X_s) 
                    while not self.stop_play.is_set():
                        # fetch tasks and calculate approximations
                        try:
                            (proc_name,X)=self.ep_task_q.get(timeout=0.01)
                        except:
                            continue
                        Y=approx.value(X)
                        self.conns[proc_name].send(Y)
                    
                    # Synchronization with finished EpisodeProcesses -> start
                    # fitting
                    self.play_finished.acquire()
                    self.play_finished.notify_all()
                    self.play_finished.release()
                    self.fit_cond.acquire()
                    self.fit_cond.wait()

                    (X,Y)=self.fit_q.get()
                    if self.internal_flag:
                        # retrieved to learn model
                        approx.fit(X,Y,self.X_train,self.Y_train,saver,fp=self.fp)
                        print 'AFTER VALUE ITERATION: ',approx.value(X_s) 
                        print saver.last_checkpoints[-1]
                        self.internal_flag=False
                    else:
                        # retrieved for validation
                        self.X_train=X
                        self.Y_train=Y
                        self.internal_flag=True

                    # more synchronization before new iteration
                    self.fit_cond.release()
                    self.stop_play.clear()
                    self.fit_done.acquire()
                    self.fit_done.notify_all()
                    self.fit_done.release()
        
class NetworkContProcess(mp.Process):
    """copy of NetworkProcess, where we build up on an existing model, which
    offers much more versability, thanks to this mp.Process we can dynamically
    change variables after we have already trained a model

    Attributes:
        prev_model: file path of previous model
     """

    def __init__(self,fit_q,prev_model,init_cond,fp,ep_task_q,ep_task_lock,stop_play,alpha=None):
        mp.Process.__init__(self)

        self.prev_model=prev_model

        self.fit_cond=mp.Condition()
        self.fit_done=mp.Condition()
        self.fit_q=fit_q
        self.init_cond=init_cond
        self.play_finished=mp.Condition()
        self.fp=fp
        self.stop_play=stop_play
        self.conns=dict()

        self.ep_task_q=ep_task_q
        self.ep_task_lock=ep_task_lock
        self.alpha=alpha

        self.internal_flag=False
        self.X_train=None
        self.Y_train=None

    def register_conn(self,proc_name,con):
        self.conns[proc_name]=con


    def fit_notify(self):
        self.fit_cond.acquire()
        self.fit_cond.notify_all()
        self.fit_cond.release()

    def fit_wait(self):
        self.fit_done.acquire()
        self.fit_done.wait()
        self.fit_done.release()

    def play_f_wait(self):
        self.play_finished.acquire()
        self.play_finished.wait()
        self.play_finished.release()

    def run(self):
        np.random.seed()
        saver=tf.train.import_meta_graph(self.prev_model+'.meta')
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)) as sess:
            saver.restore(sess,self.prev_model)
            approx=Approximator(sess)
            with sess.as_default():
                forward=tf.get_collection('forward')[0]
                X_pl=tf.get_collection('X_pl')[0]
                error=tf.get_collection('loss')

                self.init_cond.acquire()
                self.init_cond.notify_all()
                self.init_cond.release()
                X_s=faster_featurize('6k1/8/8/5K2/8/8/8/8 w - -')            
                while True:
                    # waiting for an external signal to terminate

                    print 'BEFORE VALUE ITERATION: ',approx.value(X_s) 
                    while not self.stop_play.is_set():
                        try:
                            (proc_name,X)=self.ep_task_q.get(timeout=0.01)
                        except:
                            continue
                        Y=approx.value(X)
                        self.conns[proc_name].send(Y)
                    
                    self.play_finished.acquire()
                    self.play_finished.notify_all()
                    self.play_finished.release()

                    self.fit_cond.acquire()
                    self.fit_cond.wait()

                    (X,Y)=self.fit_q.get()
                    if self.internal_flag:
                        approx.fit(X,Y,self.X_train,self.Y_train,saver,fp=self.fp)
                        print 'AFTER VALUE ITERATION: ',approx.value(X_s) 
                        print saver.last_checkpoints[-1]
                        self.internal_flag=False
                    else:
                        self.X_train=X
                        self.Y_train=Y
                        self.internal_flag=True

                    self.fit_cond.release()

                    self.stop_play.clear()

                    self.fit_done.acquire()
                    self.fit_done.notify_all()
                    self.fit_done.release()
 

if __name__=='__main__':
    import settings
    settings.init()
    settings.params['PL']=list('KRkr')
    model='Models/stem_leaf/TDStem/TDStem_stem_or_leaf_7__28_06/TDStem_stem_or_leaf_7__28_06-1_23116-0'
    S=['3k2R1/8/3K4/8/8/8/8/8 b - -','2k5/8/3K4/8/8/8/1R6/8 w -,-','2k5/8/3K4/8/8/8/8/8 w -,-']
    print Approximator.V(S,model)
