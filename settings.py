"""Settings (over whole directory) If I had known the power of this file on
beforehand I would have implemented it from the beginning with many parameters
"""

import cPickle as cp
params={'PL':['K','Q','k','q'],'DSETFILE':'','DSET':[],'S':0,'USE_DSET':False,
        'RAND':10,'MK':50,'OC_DEPTH':5}

def init():
    global params 
    params['PL']=['K','Q','R','B','N','P','k','q','r','b','n','p']
    #params['PL']=['K','R','k','r']
    params['DSETFILE']='dataset/3p.epd'
    with open(params['DSETFILE'],'rb') as f:
        params['DSET']=cp.load(f)
        params['DSET']=[epd for epd in params['DSET'] if sum([int(c.isalpha()) for c in epd])<=6]
    params['S']=len(params['DSET'])
    print 'SIZE starpos: ', params['S']
