import numpy as np
import settings
import tensorflow as tf
from approximator import Approximator
import td_stem as tdstem
import td_leaf as tdleaf
from optimal import recursive_eval_sim,lhs,we,wc,dc 
import cPickle as cp
import matplotlib.pyplot as plt
from environment import load_DS
import argparse

if name=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--meta',default=None)
    parser.add_argument('--meta',default=None)
    
