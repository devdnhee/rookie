import chess
import ctypes
import multiprocessing as mp
import multiprocessing.sharedctypes
import cPickle as cp
from approximator import Approximator,NetworkProcess,NetworkContProcess
import time
import tensorflow as tf
from learn.preprocessing import faster_featurize
import learn.ann as ann
import cPickle
#import matplotlib.pyplot as plt
import numpy as np
from agent import Agent
from environment import Environment
import os
from chessmoveswrapper import ChessMovesEngine
from abc import ABCMeta,abstractmethod

class Supervisor:
    """
    class controlling the actual learning. All relevant info is stored here,
    also about the learning curves to analyze later. Controls all
    asynchronicities. The supervisor manages learning simulations etc.

    Attributes:
        store_period: episodes before storing
        sample_period= sampling for data information
        pol=policy
        approx=approx
        dm: the DataManager
        meta: metadata to store with it
        
        fn= filename
        agents= the agents
        mv_limit=mv_limit
    """

    __metaclass__=ABCMeta
    
    STORE_DIR='Models'
    
    @staticmethod
    def get_path(fn):
        """
        utility function
        """
        return os.path.join(Supervisor.STORE_DIR,fn)

    def __init__(self,policy,store_period=10000,sample_period=1000,mv_limit=100):
        self.meta=dict()
        self.store_period=store_period
        self.sample_period=sample_period
        self.pol=policy
        self.dm=None
        self.meta['avg_rew']=[0,0]
        self.meta['avg_len']=[]
        self.meta['r_lists']=[[],[]]
        self.meta['l_list']=[]
        self.meta['w_list']=[]
        self.meta['episodes']=0
        self.mv_limit=mv_limit
        self.meta['elapsed_time']=0
        self.meta['outcomes']=[]
        self.meta['sim_time']=0
        self.meta['N']=[]
        self.meta['mps']=[]
        self.meta['eps']=[]
        self.eps=[]

        self.episode_f=None

    def store(self,fn):
        with open(fn,'wb') as f:
            cp.dump(self.meta,f)

    def retrieve(self,fn):
        with open(fn,'rb') as f:
            self.meta=cp.load(f)

    @abstractmethod
    def get_name(self):
        pass

    """
    @abstractmethod
    def run_episode(self,state=None):
        pass
    """

    def run(self,I,N,graph_f,kwargs,name='',state=None,fn=None,prev_model=None):
        """ the core method of the Supervisor. Runs an entire simulation.
         params:
             I: number of iterations
             N: number of episodes every iteration
             graph_f: how to build a tensorflow graph
             kwargs: arguments for graph_f
             state: starting position of the episodes
             fn: where to store the final model
        """

        self.meta['kwargs']=kwargs
        self.meta['state']=state

        X=faster_featurize('6k1/8/8/5K2/8/8/8/8 w - -')
        
        pool_size=mp.cpu_count()-2
        self.meta['cpus']=pool_size

        mn=self.get_name()+'_'+name+'_'+time.strftime('_%d_%m',time.gmtime(time.time()))
        directory=os.path.join(Supervisor.STORE_DIR,mn)
        os.mkdir(directory)
        fp=os.path.join(directory,'model')

        tim=time.time()

        # Creation of synchronization variables
        fit_q=mp.Queue()
        init_cond=mp.Condition()
        new_iter_cond=mp.Condition()
        stop_play=mp.Event()
        eps_change_ev=mp.Event()
        
        task_q=mp.JoinableQueue()
        q_lock=mp.RLock()
        res_q=mp.Queue()
        res_lock=mp.RLock()
        ep_task_q=mp.Queue()
        ep_task_lock=mp.RLock()
        print 'ep task queue {} created'.format(id(ep_task_q))

        n_process=NetworkProcess(fit_q,graph_f,kwargs,init_cond,os.path.join(directory,mn),ep_task_q,ep_task_lock,stop_play)
        ep_proc=[]
        for i in xrange(pool_size):
            n='EpisodeProcess-'+str(i)
            a,b=mp.Pipe()
            ep_proc.append(self.create_proc(task_q,q_lock,state,new_iter_cond,res_q,res_lock,ep_task_q,ep_task_lock,eps_change_ev,name=n))
            ep_proc[i].set_conn(a)
            n_process.register_conn(n,b)

        n_process.start()
        for proc in ep_proc:
            proc.start()

        init_cond.acquire()
        init_cond.wait()
        init_cond.release()
        
        print 'Number of cpus used: {}'.format(pool_size)
    
        for i in xrange(2*I):
            print 'Iteration {}'.format(i)
            #print 'BEFORE SELF PLAY: ',sess.run(forward,{X_pl:X})
            eps_change_ev.set()

            W=0
            ctr=0
            while W<50:
                ctr+=1

                start=time.time()

                for _ in xrange(N): task_q.put(1)
                for _ in xrange(pool_size): task_q.put(None)

                #print 'All processes: {}'.format(mp.active_children())
                new_iter_cond.acquire()
                new_iter_cond.notify_all()
                #print 'I: {} notifying all threads: new iteration'.format(mp.current_process().name)
                new_iter_cond.release() 

                task_q.join()

                end=time.time()
 
                # TODO here: assimilate data
                data=[]
                while not res_q.qsize()<=0:
                    it=res_q.get()
                    data.append(it)

                rewards=[t[0] for t in data]
                lengths=[t[1] for t in data]
                data=[t[2] for t in data]
                
                #[rewards,lengths,data]=map(list,zip(*data))
                r_w=sum([r[0] for r in rewards])/float(N)
                r_b=sum([r[1] for r in rewards])/float(N)
                avg_len=sum(lengths)/float(N)
                outcomes=sum([dt.get_outcome() for dt in data if not np.isnan(dt.get_outcome())]) 
                wins=sum([np.abs(dt.get_outcome()) for dt in data if not np.isnan(dt.get_outcome())]) 
                avg_outcome=outcomes/float(N)
                win_rate=wins/float(N)
                #print wins
                W+=wins

                #print avg_len, win_rate, r_w, r_b
                mps=sum(lengths)/(end-start)
                
                starttime=time.time()
                for dt in data:
                    self.dm.update(dt)
                avg_time=(time.time()-starttime)/float(len(data))
                print 'Glambda calculation TIME avg: {}\t size data:{}'.format(avg_time,self.dm.Y.shape)

                self.meta['sim_time']+=(end-start)
                self.meta['r_lists'][0].append(r_w)
                self.meta['r_lists'][1].append(r_b)
                self.meta['w_list'].append(wins)
                self.meta['avg_len'].append(avg_len)
                self.meta['mps'].append(mps)
                self.meta['eps'].append(self.pol.eps)
                self.meta['episodes']+=N
                self.meta['outcomes'].append(outcomes)
                self.meta['N'].append(N)

                print('I: {} episodes, {} wins in iteration {},MPS: {}'
                      .format(N*ctr,wins,i,mps))
                eps_change_ev.clear()

            self.pol.update()
            mps=sum(self.meta['mps'][-ctr:])/ctr
            win_rate=sum(self.meta['w_list'][-ctr:])/ctr
            outcome_rate=sum(self.meta['outcomes'][-ctr:])/ctr
            avg_len=sum(self.meta['avg_len'][-ctr:])/ctr
            print('I: MPS: {}\tWIN RATE: {}\tOUTCOME RATE: {}\tAVG LENGTH: {}'
                  .format(mps,win_rate,outcome_rate,avg_len))
            print 'I: fitting data after {} episodes'.format(ctr*N)

            #print 'ID stop_play main {}'.format(id(stop_play))
            stop_play.set()
            n_process.play_f_wait()

            self.dm.write_out_windata('data_visualized.txt')

            self.dm.clean()
            X,Y=self.dm.get_data()
            #X,Y=self.dm.get_balanced_data()
            print 'Dimensions Data: ',X.shape, Y.shape
            
            fit_q.put((X,Y))
            
            # Synchronization between NetworkProcess & mainProcess
            n_process.fit_notify()
            n_process.fit_wait()

            self.pol.eps=max([0.05,self.pol.eps-0.05])
            print 'New Epsilon: {}'.format(self.pol.eps)

        for p in ep_proc:
            p.terminate()
        n_process.terminate()
        self.meta['elapsed_time']+=time.time()-tim
        print 'I: NetworkProcess terminated'
        self.store(os.path.join(directory,name+'_meta.sv'))
        print 'I: metadata stored in {}'.format(os.path.join(directory,name+'meta.sv'))

    def meta_update(self,rewards,mv_cnt,win,update):
        self.meta['episodes']+=1
        self.meta['updates']+=int(update)

        # running mean
        self.meta['avg_rew'][0]=(1-1./(self.meta['episodes']))*self.meta['avg_rew'][0]+((rewards[0])/mv_cnt)/(self.meta['episodes'])
        self.meta['avg_rew'][1]=(1-1./(self.meta['episodes']))*self.meta['avg_rew'][1]+((rewards[1])/mv_cnt)/(self.meta['episodes'])
        self.meta['avg_len']=(1-1./(self.meta['episodes']))*self.meta['avg_len']+(1.*mv_cnt)/(self.meta['episodes'])
        self.meta['win_rate']=(1-1./(self.meta['episodes']))*self.meta['win_rate']+(1.*win)/(self.meta['episodes'])

    def continue_run(self,I,prev_model,name,N=1000,alpha=None):
        """
        same as run, but now for NetworkContProcess
        """
        state=self.meta['state']
 
        pool_size=mp.cpu_count()-2
        self.meta['cpus']=pool_size

        self.pol.eps=0.1

        mn=self.get_name()+'_'+name+'_'+time.strftime('_%d_%m',time.gmtime(time.time()))
        directory=os.path.join(Supervisor.STORE_DIR,mn)
        os.mkdir(directory)

        #test_v=sess.run(forward,feed_dict={X_pl:X})
        tim=time.time()

        fit_q=mp.Queue()
        init_cond=mp.Condition()
        new_iter_cond=mp.Condition()
        stop_play=mp.Event()
        eps_change_ev=mp.Event()
        
        task_q=mp.JoinableQueue()
        q_lock=mp.RLock()
        res_q=mp.Queue()
        res_lock=mp.RLock()
        ep_task_q=mp.Queue()
        ep_task_lock=mp.RLock()
        print 'ep task queue {} created'.format(id(ep_task_q))

        #self.pol.eps=0
        #self.pol.n=40

        n_process=NetworkContProcess(fit_q,prev_model,init_cond,os.path.join(directory,mn),ep_task_q,ep_task_lock,stop_play,alpha=alpha)
        ep_proc=[]
        for i in xrange(pool_size):
            n='EpisodeProcess-'+str(i)
            a,b=mp.Pipe()
            ep_proc.append(self.create_proc(task_q,q_lock,state,new_iter_cond,res_q,res_lock,ep_task_q,ep_task_lock,eps_change_ev,name=n))
            ep_proc[i].set_conn(a)
            n_process.register_conn(n,b)

        n_process.start()
        for proc in ep_proc:
            proc.start()

        init_cond.acquire()
        init_cond.wait()
        init_cond.release()
        
        print 'Number of cpus used: {}'.format(pool_size)
    
        for i in xrange(2*I):
            print 'Iteration {}'.format(i)
            #print 'BEFORE SELF PLAY: ',sess.run(forward,{X_pl:X})
            eps_change_ev.set()

            W=0
            ctr=0
            while W<50:
                ctr+=1

                start=time.time()

                for _ in xrange(N): task_q.put(1)
                for _ in xrange(pool_size): task_q.put(None)

                #print 'All processes: {}'.format(mp.active_children())
                new_iter_cond.acquire()
                new_iter_cond.notify_all()
                #print 'I: {} notifying all threads: new iteration'.format(mp.current_process().name)
                new_iter_cond.release() 

                task_q.join()

                end=time.time()
 
                # TODO here: assimilate data
                data=[]
                while not res_q.qsize()<=0:
                    it=res_q.get()
                    data.append(it)

                rewards=[t[0] for t in data]
                lengths=[t[1] for t in data]
                data=[t[2] for t in data]
                
                #[rewards,lengths,data]=map(list,zip(*data))
                r_w=sum([r[0] for r in rewards])/float(N)
                r_b=sum([r[1] for r in rewards])/float(N)
                avg_len=sum(lengths)/float(N)
                outcomes=sum([dt.get_outcome() for dt in data if not np.isnan(dt.get_outcome())]) 
                wins=sum([np.abs(dt.get_outcome()) for dt in data if not np.isnan(dt.get_outcome())]) 
                avg_outcome=outcomes/float(N)
                win_rate=wins/float(N)
                #print wins
                W+=wins

                #print avg_len, win_rate, r_w, r_b
                mps=sum(lengths)/(end-start)
                
                starttime=time.time()
                for dt in data:
                    self.dm.update(dt)
                avg_time=(time.time()-starttime)/float(len(data))
                print 'Glambda calculation TIME avg: {}\t size data:{}'.format(avg_time,self.dm.Y.shape)

                self.meta['sim_time']+=(end-start)
                self.meta['r_lists'][0].append(r_w)
                self.meta['r_lists'][1].append(r_b)
                self.meta['w_list'].append(wins)
                self.meta['avg_len'].append(avg_len)
                self.meta['mps'].append(mps)
                self.meta['eps'].append(self.pol.eps)
                self.meta['episodes']+=N
                self.meta['outcomes'].append(outcomes)
                self.meta['N'].append(N)

                print('I: {} episodes, {} wins in iteration {},MPS: {}'
                      .format(N*ctr,wins,i,mps))
                eps_change_ev.clear()

            self.pol.update()
            mps=sum(self.meta['mps'][-ctr:])/ctr
            win_rate=sum(self.meta['w_list'][-ctr:])/ctr
            outcome_rate=sum(self.meta['outcomes'][-ctr:])/ctr
            avg_len=sum(self.meta['avg_len'][-ctr:])/ctr
            print('I: MPS: {}\tWIN RATE: {}\tOUTCOME RATE: {}\tAVG LENGTH: {}'
                  .format(mps,win_rate,outcome_rate,avg_len))
            print 'I: fitting data after {} episodes'.format(ctr*N)

            #print 'ID stop_play main {}'.format(id(stop_play))
            stop_play.set()
            n_process.play_f_wait()

            self.dm.clean()
            X,Y=self.dm.get_data()
            #X,Y=self.dm.get_balanced_data()
            print 'Dimensions Data: ',X.shape, Y.shape
            
            fit_q.put((X,Y))
            
            # Synchronization between NetworkProcess & mainProcess
            n_process.fit_notify()
            n_process.fit_wait()

            self.pol.eps=max([0.05,self.pol.eps-0.05])
            print 'New Epsilon: {}'.format(self.pol.eps)

            #self.dm.write_out_windata('data_visualized.txt')
            #print 'Visualization of data stored in'.format('data_visualization.txt')

        for p in ep_proc:
            p.terminate()
        n_process.terminate()
        self.meta['elapsed_time']+=time.time()-tim
        print 'I: NetworkProcess terminated'
        self.store(os.path.join(directory,name+'_meta.sv'))
        print 'I: metadata stored in {}'.format(os.path.join(directory,name+'meta.sv'))
        #self.dm.store(os.path.join(directory,name+'_data.dm')) 
        #print 'Data stored in '.format(os.path.join(directory,name+'_data.dm'))
        #self.dm.write_out_windata('data_visualized.txt')
        #print 'Visualization of data stored in'.format('data_visualization.txt')


