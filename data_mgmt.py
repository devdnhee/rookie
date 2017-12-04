from abc import ABCMeta,abstractmethod
from learn.util import unison_shuffle_n
from learn.preprocessing import faster_featurize
import numpy as np
import cPickle as cp
import multiprocessing as mp 
import chess

class DataManager:
    """
    An abstract class used as a buffer object to guarantee transfer between
    processes of data used for training, testing and validation. Data is
    managed in this class in order to try to achieve as much independence
    between the seen data as possible to reduce overfitting. The use of the
    DataManager is thus to store and throw away data. Some samples have to be
    thrown away to improve performance and ensure independence between samples
    (see thesis)
    The way this is done is by diffusing samples from previous iterations into
    new samples which will be used for learning. This is the purpose of the ?_q
    attributes, they function as some form of memory to retrieve samples from
    to generate datasets.

    Attributes:
        X: list of processed chessboards (into features)
        Y: list of observed values for X 
        O: list of observed outcome of X
        S: list of observed chessboards, in epd strings

        X_q: list of X
        Y_q: list of Y
        O_q: list of O
        S_q: list of S

        max_buffer: max size
        internal_counter: 
    """

    __metaclass__=ABCMeta

    KEEP=[1-0.1*i for i in xrange(5)]

    def __init__(self,max_buffer=10000):
        self.X=None
        self.Y=None
        self.O=None
        self.S=[]

        self.X_q=[]
        self.Y_q=[]
        self.O_q=[]
        self.S_q=[]

        self.max_buffer=max_buffer
        self.internal_counter=0
        self.internal_buffer=5*max_buffer
        self.internal_contribution=300
        
    def write_out(self,fp):
        """method to see the data (for testing purposes)"""
        string=''
        for i in xrange(500):
            b=chess.Board.from_epd(self.S[-i])[0] 
            string+=str(b)+'\t'+str(self.Y[-i])+'\t'+str(self.O[-i])
            string+='\n\n'
        with open(fp,'w') as f:
            f.write(string)
                
    def write_out_windata(self,fp):
        """method to see the data (for testing purposes)"""
        string=''
        i=1
        j=0
        while j<10 and i<=self.O.shape[0]:
            if not np.isnan(self.O[-i]) and np.abs(self.O[-i])==1:
                b=chess.Board.from_epd(self.S[-i])[0] 
                string+=str(b)+'\t'+str(self.Y[-i])+'\t'+str(self.O[-i])
                string+='\n\n'
                j+=1
            i+=1
        with open(fp,'w') as f:
            f.write(string)

    @abstractmethod
    def update(self,data_thread):
        """update DataManager with newly seen samples from a DataThread"""
        pass
    
    def shuffle(self):
        """shuffle data for randomness"""
        N=self.X.shape[0]
        p=np.random.permutation(N)

        self.X=self.X[p]
        self.Y=self.Y[p]
        self.O=self.O[p]
        self.X=self.X[-min([N,self.max_buffer]):,:]
        self.Y=self.Y[-min([N,self.max_buffer]):,:]
        self.O=self.O[-min([N,self.max_buffer]):]

        self.internal_contribution=self.max_buffer/self.internal_counter+1
        print 'AVG Contribution game: {}'.format(self.internal_contribution)

    def clean(self):
        """clean after iteration and store into buffer, more recently seen data
        is stored in higher numbers"""

        N=self.X.shape[0]
        
        x,y,o=self.get_outcome_data()

        p=np.random.permutation(N)

        self.X=self.X[p]
        self.Y=self.Y[p]
        self.O=self.O[p]
        self.X=self.X[-min([N,self.max_buffer]):,:]
        self.Y=self.Y[-min([N,self.max_buffer]):,:]
        self.O=self.O[-min([N,self.max_buffer]):]
        self.X=np.concatenate((self.X,x))
        self.Y=np.concatenate((self.Y,y))
        self.O=np.concatenate((self.O,o))

        self.X_q.insert(0,self.X)
        self.Y_q.insert(0,self.Y)
        self.O_q.insert(0,self.O)

        if len(self.X_q)>=5:
            self.X_q.pop()
            self.Y_q.pop()
            self.O_q.pop()

        self.X=None
        self.Y=None
        self.O=None

        self.internal_counter=0
        self.internal_contribution=300

    def store(self,fn):
        with open(fn,'wb') as f:
            cp.dump((self.X,self.Y,self.O),f)

    def retrieve(self,fn):
        with open(fn,'rb') as f:
            t=cp.load(f)
            self.X=t[0]
            self.Y=t[1]
            self.O=t[2]

    def get_data(self):
        X_data=[]
        Y_data=[]
        for i in xrange(len(self.X_q)):
            s=self.X_q[i].shape[0]
            p=np.random.permutation(s)
            Xp=self.X_q[i][p]
            X_data.append(Xp[:int(s*self.KEEP[i]),:])
            Yp=self.Y_q[i][p]
            Y_data.append(Yp[:int(s*self.KEEP[i]),:])
        X_data=np.concatenate([x for x in X_data])
        Y_data=np.concatenate([y for y in Y_data])
        return X_data,Y_data

    def get_balanced_data(self):
        """ balance the data, this is important, as for example in the
        beginning the model will perform very badly, not much successes are
        seen relative to the overall data. By imputing more successes this info
        is learned faster.
        """
        win_ind=np.where(np.abs(self.O)==1)
        draw_ind=(self.O==0)
        rest_ind=np.where(np.isnan(self.O))

        x=self.X[win_ind[0]] 
        y=self.Y[win_ind[0]]
        n=x.shape[0]
        p1=np.random.permutation(n)
        x,y=x[p1],y[p1]
        x,y=x[:n/2],y[:n/2]

        xu,yu=self.X[draw_ind[0]],self.Y[draw_ind[0]]
        m=min([7*n,xu.shape[0]])
        p2=np.random.permutation(xu.shape[0])
        xu,yu=xu[p2],yu[p2]
        xu,yu=xu[:m/2],yu[:m/2]

        xr,yr=self.X[rest_ind[0]],self.Y[rest_ind[0]]
        l=min([10*n,xr.shape[0]])
        p3=np.random.permutation(xr.shape[0])
        xr,yr=xr[p3],yr[p3]
        xr,yr=xr[:l/2],yr[:l/2]

        x,y=np.concatenate((x,xu,xr)),np.concatenate((y,yu,yr))

        return x,y

    def get_outcome_data(self):
        win_ind=np.where(np.abs(self.O)==1)
        x=self.X[win_ind[0]]
        y=self.Y[win_ind[0]]
        o=self.O[win_ind[0]]
        n=x.shape[0]
        p1=np.random.permutation(n)
        x,y,o=x[p1],y[p1],o[p1]
        x,y,o=x[:n/2],y[:n/2],o[:n/2]
        return x,y,o

class DataThread:
    """Every EpisodeProcess keeps one of these, to gather data in an organized
    way to transfer further to the DataManager"""

    def __init__(self):
        self.s=[]
        self.r=[]
        self.v=[]
        self.o=np.NaN
        self.f=dict()

    def append(self,(s,r,v)):
        self.s.append(s)
        self.r.append(r)
        self.v.append(v)

    def set_outcome(self,outcome):
        self.o=outcome

    def get_outcome(self):
        return self.o

    def empty(self):
        self.s=[]
        self.r=[]
        self.v=[]

    def get_data(self):
        return self.s, self.r, self.v

    def put(self,s,F):
        self.f[s]=F

    def put_and_get(self,s):
        if s not in self.f.keys():
            self.f[s]=faster_featurize(s)
        return self.f[s]

    def get_feat(self,s):
        return self.f[s]
