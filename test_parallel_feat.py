import multiprocessing as mp
import time
from learn.preprocessing import faster_featurize

S=['2k5/8/3K4/8/8/3R4/8/8 w - -','8/8/1k6/8/5K2/3R4/8/8 w - -']

s=time.time()
p=mp.Pool(5)
r=p.map(faster_featurize,S)
e=time.time()-s
print 'Parallelized: ', e

s=time.time()
r=[faster_featurize(st) for st in S]
e=time.time()-s
print 'Serialized: ', e 

