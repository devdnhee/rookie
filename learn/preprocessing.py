import os,sys,inspect
currentdir =os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 
projdir=os.path.dirname(parentdir)
sys.path.insert(0,projdir) 
import chess_utils as util
from chessmoveswrapper import ChessMovesEngine

import numpy as np
import chess.gaviota as gav
import chess
import chess.uci as uci
import chess.pgn as pgn
import logging
import cPickle as cp
import settings

gavdir=os.path.join(projdir,'gaviota')

def change_PL(pl):
    settings.params['PL']=pl

def create_epd_set(pgnfile,writefile):
    """
    method to generate a set of epd strings fom a pgn dataset.
    """
    dset=open(pgnfile)
    stop=False
    epd_set=[]
    i=0
    while not stop and i<500000:
        i+=1
        try:
            game=pgn.read_game(dset)
            if game is None:
                break
            node=game
            while not node.is_end():
                node=node.variation(0)
                b=node.board().epd()
                epd_set.append(b)
        except Exception as e:
            print e
            break
        if i%1000==0:
            print(i)
    print epd_set[:100]
    with open(writefile,'w') as f:
        cp.dump(epd_set,f)

def erase_doubles(epdset):
    with open(epdset,'rb') as f:
        dset=cp.load(f)
    print('size before uniquefying: {}',len(dset))
    dset=list(set(dset))
    print('size after uniquefying: {}',len(dset))
    with open('epdset_unique','wb') as f:
        cp.dump(dset,f)

def filter_ds(epdset_f,output,filter_f,*args):
    """filter a ds to a specific filter function"""
    with open(epdset_f,'rb') as f:
        ds=cp.load(f)
    print len(ds)
    print len(ds)
    filtered_ds=[epd for epd in ds if filter_f(epd,*args)]
    print len(filtered_ds)
    with open(output,'wb') as f:
        cp.dump(filtered_ds,f)

def filter_size(epd,sz):
    return sum([int(c.isalpha()) for c in epd])<=sz+1

def filter_cfg(epd,cfgs):
    epd=epd.split()[0]
    a=[np.prod(np.array([int(epd.count(p)==cfg.count(p)) for p in
                         cfg]))*int(filter_size(epd,len(cfg)-1)) for cfg in cfgs]
    return  sum(a)>0 

def filter_split(epdset_f):
    with open(epdset_f,'rb') as f:
        dset=cp.load(f)
    name=epdset_f.split()[0]
    ds1=dset[:len(dset)/2]
    ds2=dset[len(dset)/2:]
    with open(name+'1.epd','wb') as f:
        cp.dump(ds1,f)
    with open(name+'2.epd','wb') as f:
        cp.dump(ds2,f)

def filter_join(epdset_f1,epdset_f2):
    with open(epdset_f1,'rb') as f:
        dset=cp.load(f)
    with open(epdset_f2,'rb') as f:
        dset2=cp.load(f)
    ds=dset+dset2
    with open(epdset_f1,'wb') as f:
        cp.dump(ds,f)

def create_testset(fn,o,N=2000):
    with open(fn,'rb') as f:
        ds=cp.load(f)
    p=np.random.permutation(len(ds))
    ds_shuffle=[ds[i] for i in p][:min([N,len(ds)])]
    with open(o,'wb') as f:
        cp.dump(ds_shuffle,f)

def get_data(N=2000):
    bl=generate_chess_boards(N,balanced=True)
    tb=gav.open_tablebases(gavdir)
    t=[(float(tb.probe_dtm(b) if b.turn else -tb.probe_dtm(b))
        ,featurize(b),draw_indicator(tb,b),b) for b in bl if tb.probe_dtm(b) is not None] 
    Y,X,Y_draw,B=zip(*t)
    X=np.concatenate([x for x in X])
    Y=np.array(Y)
    Y_draw=np.array(Y_draw)
    Y=whiten_dtm(Y_draw,Y)
    return (X,Y,Y_draw,B)

def write_data(path,N=3000):
    data=get_data(N)
    with open(path,'w') as f:
        cp.dump(data,f)

def read_data(path):
    return cp.load(open(path,'r'))

def set_up_engine(path=os.path.join(projdir,'stockfish7')):
    engine=uci.popen_engine(path)
    engine.uci()
    return engine

def get_score(engine,board,ih):
    engine.position(board)
    engine.go()
    score=ih.info["score"][1]
    cp=score.cp
    mate=score.mate
    if cp is None:
        toret= 2*(int(mate>0)-0.5)*100*(1+1/(np.abs(mate)/10.0))
    else: toret=cp/float(100)
    return toret

"""
function indicating if position is draw or not
"""
def draw_indicator(tb, board):
    dtm=tb.probe_dtm(board)
    if dtm==0 and not board.is_checkmate():
        return 1
    else:
        return 0
    
"""
whitening dtm data for draw positions
"""
def whiten_dtm(draw,dtm):
    sigma=np.std(dtm[draw==0])
    mu=np.mean(dtm[draw==0])
    for i in xrange(len(dtm)):
        if draw[i]==1:
            dtm[i]=sigma*np.random.randn()+mu
    return dtm

"""
Generate random data (boards) for KRK problem. 
"""
def generate_chess_boards(N=20000,both_col=True,balanced=True):
    board_list=[]
    epd_list=[]
    tb=gav.open_tablebases(gavdir)
    if not balanced:
        for _ in xrange(N):
            rand=np.random.uniform()
            turn=int(np.round(np.random.uniform())) if both_col else 0
            side=[chess.WHITE,chess.BLACK][turn]
            if rand<0.48:
                b=util.random_board_initialization(util.piece_configurations['KRk'],turn=turn)
                b.turn=side
            elif 0.48 <= rand <= 0.96: 
                b=util.random_board_initialization(util.piece_configurations['KRk'],turn=turn)
                b.turn=side
            else:
                b=util.random_board_initialization(util.piece_configurations['Kk'],turn=turn)
                b.turn=side
            if b.epd() not in epd_list:
                epd_list.append(b.epd())
                board_list.append(b)
    else:
        i=0
        limit=N/30
        
        hist=dict()
        while i<N:
            rand=np.random.uniform()
            turn=int(np.round(np.random.uniform())) if both_col else 0
            side=[chess.WHITE,chess.BLACK][turn]
            if rand<0.48:
                b=util.random_board_initialization(util.piece_configurations['KRk'],turn=turn)
                b.turn=side
            elif 0.48 <= rand <= 0.96: 
                b=util.random_board_initialization(util.piece_configurations['KRk'],turn=turn)
                b.turn=side
            else:
                b=util.random_board_initialization(util.piece_configurations['Kk'],turn=turn)
                b.turn=side
            if b.epd() not in epd_list:
                dtm=tb.probe_dtm(b)
                if dtm not in hist.keys():
                    hist[dtm]=0
                if hist[dtm]<limit:
                    epd_list.append(b.epd())
                    board_list.append(b)
                    hist[dtm]=hist[dtm]+1
                    i+=1
            
        print hist
            
    return board_list

"""
transform the chess board in to a feature representation used for machine
learning models.

Returns
-------
the feature vector, numpy s)hape (1,D) (column vector)
"""
def featurize(board):
    # function to call to get a nice representation of the feature
    global feat_func

    # x are the features of the board
    x=[]

    """
    # material configuration
    start_config=[8,2,2,2,1,1]
    for p in range(1,7):
        for t in xrange(2):
            pl=board.pieces(p,bool(t))
            x.append(len(pl))
            for _ in xrange(start_config[p-1]):
                if len(pl)>0:
                    sq=pl.pop()
                    x.append(sq%8)
                    x.append(sq/8)
                else:
                    # as the piece doesn't exist, we want the NN to not make
                    # wrong conclusions, this is done best by yielding in a
                    # gaussian random number (later, the integers will be
                    # processed into a gaussian distribution anyway)
                    x+=[0,8*np.random.randn()]
    """
    
    pieces_look=[chess.KING,chess.ROOK]
    inp_ch_piece=2
    channels=np.zeros((inp_ch_piece*2*len(pieces_look),8,8))
    for t in xrange(2):
        for p in xrange(len(pieces_look)):
            pl=board.pieces(pieces_look[p],bool(t))
            for sq in pl:
                assert 4*t+2*p+1<8
                channels[4*t+2*p,sq/8,sq%8]=1.0
                at_ss=board.attacks(sq)
                for at_sq in at_ss:
                    # extra check needed, due to a bug somewhere in chess
                    # library
                    if not at_sq/8>=8:
                        channels[4*t+2*p+1,at_sq/8,at_sq%8]=1.0

    #channel_printer(channels)

    # add whose turn it is as a feature
    x.append(int(board.turn))

    # some game decisive features
    x+=[int(board.is_checkmate()),int(board.is_stalemate()),int(board.can_claim_draw()),int(board.is_insufficient_material())]

    x=np.array(x,ndmin=2)
    ch=channels.reshape((1,4*64*len(pieces_look)))
    X=np.concatenate((x,ch),axis=1)
    return X

def faster_featurize(epd,pieces_look=settings.params['PL']):
    """transformation of a board to chess features. with pieces_look we can
    limit the total amount of bitboards"""
    pieces_look=pieces_look

    engine=ChessMovesEngine(epd)
    inp_ch_piece=2
    channels=np.zeros((inp_ch_piece*len(settings.params['PL']),8,8))

    pieces=dict()
    a=(epd.split(' ')[0]).split('/')
    assert len(a)==8
    for i in xrange(len(a)):
        f=0
        for j in xrange(len(a[i])):
            if ord('0')<=ord(a[i][j])<=ord('9'):
                f+=int(a[i][j])
            else:
                p=a[i][j]
                channels[2*settings.params['PL'].index(p),i,f]=1
                pieces[(i,f)]=p 
                f+=1
    
    for _ in xrange(2):
        engine._compute_moves()
        moves=engine.get_moves_and_states() 
        for m in moves.keys():
            ((r1,c1),(r2,c2))=util.uci_to_squares(m)
            p=pieces[(r1,c1)]
            channels[2*settings.params['PL'].index(p)+1,r2,c2]=1
        engine.change_turn()

    checkmate=engine.get_result!='*' and engine.get_result!= '1/2-1/2'
    draw=engine.get_result=='1/2-1/2'

    a=[int(engine.turn()=='w'),int(checkmate),int(draw)]

    X=np.array(a,ndmin=2)
    ch=channels.reshape((1,inp_ch_piece*64*len(settings.params['PL'])))
    X=np.concatenate((X,ch),axis=1)
    return X

def perf_featurize():
    X,Y,Y_draw,B=get_data(10000)
    S=[b.epd() for b in B]
    import time
    start=time.time()
    X=np.array([featurize(chess.Board.from_epd(s)[0]) for s in S])
    X=X.reshape((X.shape[0],X.shape[2]))
    duration=time.time()-start
    print 'FEATURIZE TIME: {}'.format(duration/10000.)

def channel_printer(channels):
    s=""
    for i in xrange(channels.shape[0]):
        for r in xrange(channels.shape[1]):
            s+='\n'
            for c in xrange(channels.shape[2]):
                s+=str(int(channels[i,r,c]))+" "
        s+='\n\n'
    print s

def channel_printer_latex(channels):
    s=""
    for i in xrange(channels.shape[0]):
        s+='$\\begin{smallmatrix}'
        for r in xrange(channels.shape[1]):
            for c in xrange(channels.shape[2]):
                s+=str(int(channels[i,r,c]))
                if c<channels.shape[2]-1:
                    s+='&'
            if r<channels.shape[1]-1:
                s+='\\\\'
        s+='\\end{smallmatrix}$'
        s+='\n\n'
    print s

"""
X is 1 board feature
"""
def to_channels(X):
    X=X.T
    n=X.shape[0]/64
    channels=X[3:]
    return np.reshape(channels,(n,8,8)) 

def defeaturize(X):
    pass 
    
def print_features(epd,pieces_look=['K','R','k','r']):
    f=faster_featurize(epd,pieces_look=pieces_look)
    ch=to_channels(f)
    channel_printer(ch)

def print_features_latex(epd):
    f=faster_featurize(epd)
    ch=to_channels(f)
    channel_printer_latex(ch)

def featurize_col(val,col):
    column=val[:,col]
    s=np.unique(val[:,col])
    if s.shape[0]<=2:
        return binarize(column)
    else: return realize(column)

"""
make a binary representation of a feature
"""
def binarize(column):
    return column.astype(int)

"""
make a real representation of a feature
"""
def realize(column):
    b=column.astype(float)
    mu=np.mean(b)
    sigma=np.std(b)
    if sigma==0: sigma=1
    return (b-mu)/sigma 

"""
make a one hot encoding for a category
"""
def categorize(val,col):
    cat=set(column)
    A=None
    for s in cat:
        a=np.array([int(v==s) for v in column])
        if A is None:
            A=a
        else:
            A=np.column_stack((A,a))
    return A

def fast_test():
    epd='2k5/8/3K4/8/8/3R4/8/8 w - -'
    faster_featurize(epd)

if __name__=='__main__':
    #print generate_chess_boards)
    #write_data(path='krk_data_20000_balanced_8.cpkl',N=20000)
    #x,y,y_d,b=read_data('krk_data_20000_balanced_8.cpkl')
    #fast_test()
    settings.params['PL']='KRkr'
    print_features_latex('1k6/8/K2R4/8/8/8/8/8 w - -')
    #                   ,pieces_look=['K','Q','R','B','N','P','k','q','r','b','n','p'])
    #create_epd_set('dataset/fics2.pgn','dataset/fics2.epd')
    #erase_doubles('dataset/fics.epd')
    #filter_ds('dataset/5p_fics.epd','dataset/kqk_fics.epd',filter_cfg,['KQk','Kkq'])
    #filter_split('filtered.epd')
    #create_testset('dataset/3p_fics.epd','dataset/test.epd',N=20)
