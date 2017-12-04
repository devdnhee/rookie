"""comparing different trained models with each other, generates plots"""

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


def comparison_stem_leaf():
    settings.init()
    settings.params['USE_DSET']=True
    settings.params['PL']='KRkr'
    load_DS('dataset/krk.epd')
    
    with open('Models/stem_leaf/TDLeaf/TDLeaf_stem_or_leaf_7__03_07/stem_or_leaf_7_meta.sv','rb') as f:
        leaf=cp.load(f)
    with open('Models/stem_leaf/TDStem/TDStem_stem_or_leaf_7__28_06/stem_or_leaf_7_meta.sv','rb') as f:
        stem=cp.load(f)
    print leaf.keys()
    print stem.keys()
    N_l=leaf['N'][0]
    N_s=stem['N'][0]
    #print N_l, N_s
    w_l=leaf['w_list']
    r_l=leaf['r_lists']
    l_l=leaf['avg_len']
    w_s=stem['w_list']
    r_s=stem['r_lists']
    l_s=stem['avg_len']

    mps_s=np.mean(np.array(stem['mps']))
    mps_l=np.mean(np.array(leaf['mps']))
    ntot_s=stem['episodes']
    ntot_l=leaf['episodes']
    el_s=stem['elapsed_time']
    el_l=leaf['elapsed_time']

    print mps_s, mps_l, ntot_s, ntot_l, el_s, el_l

    ep_s2=[0]
    rate_s=[]
    cumsum=0
    for i in xrange(len(w_s)):
        if i<53:
            cumsum+=5000
            rate_s.append(5000.)
        elif i<73:
            cumsum+=500
            rate_s.append(500.)
        else:
            cumsum+=250
            rate_s.append(250.)
        ep_s2.append(cumsum)
    wr_s=np.array([0]+w_s)/np.array([1]+rate_s)
    rrw_s=5000*np.array(r_s[0])/np.array(rate_s)
    rrb_s=5000*np.array(r_s[1])/np.array(rate_s)

    ep_l2=[0]
    rate_l=[]
    cumsum=0
    for i in xrange(len(w_l)):
        if i<63:
            cumsum+=5000
            rate_l.append(5000.)
        elif i<83:
            cumsum+=500
            rate_l.append(500.)
        else:
            cumsum+=250
            rate_l.append(250.)
        ep_l2.append(cumsum)
    wr_l=np.array([0]+w_l)/np.array([1]+rate_l)
    rrw_l=5000*np.array(r_l[0])/np.array(rate_l)
    rrb_l=5000*np.array(r_l[1])/np.array(rate_l)

    plt.figure(1)
    plt.subplot(111)
    line_stem, =plt.plot(ep_s2,wr_s,label='TD-Stem'+r'$(\lambda)$')
    for i in [100000,170000,264000,275000,283200,291000]:
        plt.axvline(x=i,color='#99ccff')
    line_leaf, =plt.plot(ep_l2,wr_l,label='TD-Leaf'+r'$(\lambda)$')
    for i in [120000,220000,315000,325250,333000,341000]:
        plt.axvline(x=i,color='#ffc266')
    plt.xlabel(r'$N$')
    plt.ylabel('winning rate')
    plt.legend(handles=[line_leaf,line_stem])
    plt.xlim(0,max(ep_l2))
    plt.ylim(0,1)
    #plt.title('krk endgame learning curve')

    plt.show()

    mps_s=np.mean(np.array(stem['mps']))
    mps_l=np.mean(np.array(leaf['mps']))
    ntot_s=stem['episodes']
    ntot_l=leaf['episodes']
    el_s=stem['elapsed_time']
    el_l=leaf['elapsed_time']

    print mps_s, mps_l, ntot_s, ntot_l, el_s, el_l

    model_fn='Models/stem_leaf/TDStem/TDStem_stem_or_leaf_7__28_06/TDStem_stem_or_leaf_7__28_06-1_23116-0'
    with open('Models/stem_leaf/TDStem/sim','rb') as f:
        A,evaldict,S=cp.load(f)
    
    wc_s=np.mean(np.array(evaldict['wc']))
    we_s=np.mean(np.array(evaldict['we']))
    lhs_s=np.mean(np.array(evaldict['lhs']))
    #t=stem['']
    print wc_s, we_s, lhs_s

    S=[t[-1] for t in A]
    dtm=[t[1] for t in A]
    wdl=[t[0] for t in A]
    v=Approximator.V(S,model_fn)
    hist_wcs=33*[0]
    hist_wes=33*[0]
    hist_lhss=33*[0]
    avg_vs=33*[0]
    std_vs=33*[0]
    #print A
    for i in xrange(len(hist_wcs)):
        hist_wcs[i]=np.mean(np.array([wc(t) for t in A if wc(t) is not None and 
                    t[1]==i]))
        hist_wes[i]=np.mean(np.array([we(t) for t in A if we(t) is not None and 
                    t[1]==i]))
        hist_lhss[i]=np.mean(np.array([lhs(t) for t in A if lhs(t) is not None and 
                    t[1]==i]))
        avg_vs[i]=np.mean(np.array([v[j] for j in xrange(v.shape[0]) if
                                   dtm[j]==i and wdl[j]==1]))
        std_vs[i]=np.std(np.array([v[j] for j in xrange(v.shape[0]) if
                                   dtm[j]==i and wdl[j]==1]))
        
    model_fn='Models/stem_leaf/TDLeaf/TDLeaf_stem_or_leaf_7__03_07/TDLeaf_stem_or_leaf_7__03_07-1_13299-0'
    with open('Models/stem_leaf/TDLeaf/sim','rb') as f:
        A,evaldict,S=cp.load(f)
    wc_l=np.mean(np.array(evaldict['wc']))
    we_l=np.mean(np.array(evaldict['we']))
    lhs_l=np.mean(np.array(evaldict['lhs']))
    #t=stem['']
    print wc_l, we_l, lhs_l

    S=[t[-1] for t in A]
    dtm=[t[1] for t in A]
    wdl=[t[0] for t in A]
    v=Approximator.V(S,model_fn)
    hist_wcl=33*[0]
    hist_wel=33*[0]
    hist_lhsl=33*[0]
    avg_vl=33*[0]
    std_vl=33*[0]
    #print A
    for i in xrange(len(hist_wcl)):
        hist_wcl[i]=np.mean(np.array([wc(t) for t in A if wc(t) is not None and 
                    t[1]==i+1]))
        hist_wel[i]=np.mean(np.array([we(t) for t in A if we(t) is not None and 
                    t[1]==i+1]))
        hist_lhsl[i]=np.mean(np.array([lhs(t) for t in A if lhs(t) is not None and 
                    t[1]==i+1]))
        avg_vl[i]=np.mean(np.array([v[j] for j in xrange(v.shape[0]) if
                                   dtm[j]==i+1 and wdl[j]==1]))
        std_vl[i]=np.std(np.array([v[j] for j in xrange(v.shape[0]) if
                                   dtm[j]==i+1 and wdl[j]==1]))
    
    x=np.array(range(1,len(hist_wcs)+1))

    plt.figure(2)
    plt.subplot(111)
    b1=plt.bar(x-1./6, hist_wcs,width=1./3,align='center',label='TD-Stem'+r'$(\lambda)$ ')
    b2=plt.bar(x+1./6, hist_wcl,width=1./3,align='center',label='TD-Leaf'+r'$(\lambda)$ ')
    #plt.title('krk endgame win conversion rate')
    plt.legend(handles=[b1,b2])
    plt.xlabel('DTM')
    plt.ylabel('WCR')
    plt.show()

    plt.figure(3)
    plt.subplot(111)
    b1=plt.bar(x-1./6, hist_wes,width=1./3,align='center',label='TD-Stem'+r'$(\lambda)$ ')
    b2=plt.bar(x+1./6, hist_wel,width=1./3,align='center',label='TD-Leaf'+r'$(\lambda)$ ')
    #plt.title('krk endgame win efficiency')
    plt.legend(handles=[b1,b2])
    plt.xlabel('DTM')
    plt.ylabel('WE')
    plt.show()

    plt.figure(4)
    plt.subplot(111)
    b1=plt.bar(x-1./6, hist_lhss,width=1./3,align='center',label='TD-Stem'+r'$(\lambda)$ ')
    b2=plt.bar(x+1./6, hist_lhsl,width=1./3,align='center',label='TD-Leaf'+r'$(\lambda)$ ')
    #plt.title('krk endgame loss holding score')
    plt.legend(handles=[b1,b2])
    plt.xlabel('DTM')
    plt.ylabel('LHS')
    plt.show()

    plt.figure(5)
    plt.subplot(111)
    b1,=plt.plot(x,avg_vs,label='TD-Stem'+r'$(\lambda)$ ')
    b2,=plt.plot(x,avg_vl,label='TD-Leaf'+r'$(\lambda)$ ')
    s1,=plt.plot(x,np.array(avg_vs)+2*np.array(std_vs),color='#99ccff')
    s2,=plt.plot(x,np.array(avg_vs)-2*np.array(std_vs),color='#99ccff')
    s3,=plt.plot(x,np.array(avg_vl)+2*np.array(std_vl),color='#ffc266')
    s4,=plt.plot(x,np.array(avg_vl)-2*np.array(std_vl),color='#ffc266')
    #plt.title('krk endgame value function')
    plt.legend(handles=[b1,b2])
    plt.xlabel('DTM')
    plt.ylabel(r'$V$')
    plt.show()


def comparison_stem_leaf_kqk():

    model_fn_leaf='Models/KQK/TDL/network'
    with open('Models/KQK/TDL_BAD/sim','rb') as f:
        A,evaldict,S=cp.load(f)
    wc_l=np.mean(np.array(evaldict['wc']))
    we_l=np.mean(np.array(evaldict['we']))
    lhs_l=np.mean(np.array(evaldict['lhs']))
    #t=stem[''] 
    print wc_l, we_l, lhs_l
    model_fn_leaf='Models/KQK/TDS/network'
    with open('Models/KQK/TDS_BAD/sim','rb') as f:
        A,evaldict,S=cp.load(f)
    wc_s=np.mean(np.array(evaldict['wc']))
    we_s=np.mean(np.array(evaldict['we']))
    lhs_s=np.mean(np.array(evaldict['lhs']))
    #t=stem[''] 
    print wc_s, we_s, lhs_s
    with open('Models/KQK/TDL_BAD/meta','rb') as f:
        leaf=cp.load(f)
    with open('Models/KQK/TDS_BAD/meta','rb') as f:
        stem=cp.load(f)
    mps_s=np.mean(np.array(stem['mps']))
    mps_l=np.mean(np.array(leaf['mps']))
    ntot_s=stem['episodes']
    ntot_l=leaf['episodes']
    el_s=stem['elapsed_time']
    el_l=leaf['elapsed_time']
    print mps_s, mps_l, ntot_s, ntot_l, el_s, el_l



    import tablebases
    with open('Models/KQK/TDL/meta','rb') as f:
        leaf=cp.load(f)
    with open('Models/KQK/TDS/meta','rb') as f:
        stem=cp.load(f)

    settings.init()
    load_DS('dataset/kqk_fics.epd')
    settings.params['PL']='KQkq'
    settings.params['USE_DSET']=True

    N_l=np.array([0]+leaf['N'],dtype=float) 
    N_s=np.array([0]+stem['N'],dtype=float) 
    eps_l=leaf['eps'] 
    eps_s=stem['eps']
    w_l=np.array([0]+leaf['w_list'])/N_l
    w_s=np.array([0]+stem['w_list'])/N_s
    e_l=np.cumsum(N_l)
    e_s=np.cumsum(N_s)

    stages_s=[t[0] for t in stem['lambda']]
    stages_l=[t[0] for t in leaf['lambda']]
    print leaf['lambda']

    l_l=leaf['avg_len'] 
    l_s=stem['avg_len']

    plt.figure(1)
    plt.subplot(111)
    line_stem, =plt.plot(e_s,w_s,label='TD-Stem'+r'$(\lambda)$')
    line_leaf, =plt.plot(e_l,w_l,label='TD-Leaf'+r'$(\lambda)$')
    for i in stages_s:
        plt.axvline(x=i,color='#99ccff')
    for i in stages_l:
        plt.axvline(x=i,color='#ffc266')
    plt.xlabel(r'$N$')
    plt.ylabel('winning rate')
    plt.legend(handles=[line_leaf,line_stem])
    plt.xlim(0,max(max(e_l),max(e_s)))
    plt.ylim(0,1)
    plt.show()

    mps_s=np.mean(np.array(stem['mps']))
    mps_l=np.mean(np.array(leaf['mps']))
    ntot_s=stem['episodes']
    ntot_l=leaf['episodes']
    el_s=stem['elapsed_time']
    el_l=leaf['elapsed_time']

    print mps_s, mps_l, ntot_s, ntot_l, el_s, el_l

    model_fn_stem='Models/KQK/TDS/network'
    with open('Models/KQK/TDS/sim2','rb') as f:
        A,evaldict,S=cp.load(f)

    wc_s=np.mean(np.array(evaldict['wc']))
    we_s=np.mean(np.array(evaldict['we']))
    lhs_s=np.mean(np.array(evaldict['lhs']))
    #t=stem['']
    print wc_s, we_s, lhs_s

    tw=[t for t in A if tablebases.probe_result(t[-1])==1]
    td=[t for t in A if tablebases.probe_result(t[-1])==0]
    tb=[t for t in A if tablebases.probe_result(t[-1])==-1]

    Sw=[t[-1] for t in tw]
    dtmw=[t[1] for t in tw]
    print min(dtmw)
    wdlw=[t[0] for t in tw]

    Sb=[t[-1] for t in tb]
    dtmb=[t[1] for t in tb]
    wdlb=[t[0] for t in tb]
    print min(dtmb)

    vw_s=Approximator.V(Sw,model_fn_stem)
    vb_s=Approximator.V(Sb,model_fn_stem)

    hist_wcs=20*[0]
    hist_wes=20*[0]
    hist_lhss=20*[0]
    hist_dcs=20*[0]
    avg_vs=20*[0]
    std_vs=20*[0]
    avg_vsb=20*[0]
    std_vsb=20*[0]
    #print A
    for i in xrange(len(hist_wcs)):
        hist_wcs[i]=np.mean(np.array([wc(t) for t in A if wc(t) is not None and 
                    t[1]==i+1]))
        hist_wes[i]=np.mean(np.array([we(t) for t in A if we(t) is not None and 
                    t[1]==i+1]))
        hist_lhss[i]=np.mean(np.array([lhs(t) for t in A if lhs(t) is not None and 
                    t[1]==i+1]))
        hist_dcs[i]=np.mean(np.array([dc(t) for t in A if dc(t) is not None and 
                    t[1]==i+1]))
        avg_vs[i]=np.mean(np.array([vw_s[j] for j in xrange(vw_s.shape[0]) if
                                   dtmw[j]==i+1 ]))
        std_vs[i]=np.std(np.array([vw_s[j] for j in xrange(vw_s.shape[0]) if
                                   dtmw[j]==i+1 and wdlw[j]==1]))
        avg_vsb[i]=np.mean(np.array([vb_s[j] for j in xrange(vb_s.shape[0]) if
                                   dtmb[j]==i+1 ]))
        std_vsb[i]=np.std(np.array([vb_s[j] for j in xrange(vb_s.shape[0]) if
                                   dtmb[j]==i+1 and wdlb[j]==1]))

    model_fn_leaf='Models/KQK/TDL/network'
    with open('Models/KQK/TDL/sim2','rb') as f:
        A,evaldict,S=cp.load(f)
    wc_l=np.mean(np.array(evaldict['wc']))
    we_l=np.mean(np.array(evaldict['we']))
    lhs_l=np.mean(np.array(evaldict['lhs']))
    #t=stem['']
    print wc_l, we_l, lhs_l

    tw=[t for t in A if tablebases.probe_result(t[-1])==1]
    td=[t for t in A if tablebases.probe_result(t[-1])==0]
    tb=[t for t in A if tablebases.probe_result(t[-1])==-1]
    Sw=[t[-1] for t in tw]
    dtmw=[t[1] for t in tw]
    wdlw=[t[0] for t in tw]
    Sb=[t[-1] for t in tb]
    dtmb=[t[1] for t in tb]
    wdlb=[t[0] for t in tb]

    vw_l=Approximator.V(Sw,model_fn_leaf)
    vb_l=Approximator.V(Sb,model_fn_leaf)

    hist_wcl=20*[0]
    hist_wel=20*[0]
    hist_lhsl=20*[0]
    hist_dcl=20*[0]
    avg_vl=20*[0]
    std_vl=20*[0]
    avg_vlb=20*[0]
    std_vlb=20*[0]

    for i in xrange(len(hist_wcs)):
        hist_wcl[i]=np.mean(np.array([wc(t) for t in A if wc(t) is not None and 
                    t[1]==i+1]))
        hist_wel[i]=np.mean(np.array([we(t) for t in A if we(t) is not None and 
                    t[1]==i+1]))
        hist_lhsl[i]=np.mean(np.array([lhs(t) for t in A if lhs(t) is not None and 
                    t[1]==i+1]))
        hist_dcl[i]=np.mean(np.array([dc(t) for t in A if dc(t) is not None and 
                    t[1]==i+1]))
        avg_vl[i]=np.mean(np.array([vw_l[j] for j in xrange(vw_l.shape[0]) if
                                   dtmw[j]==i+1 ]))
        std_vl[i]=np.std(np.array([vw_l[j] for j in xrange(vw_l.shape[0]) if
                                   dtmw[j]==i+1 and wdlw[j]==1]))
        avg_vlb[i]=np.mean(np.array([vb_l[j] for j in xrange(vb_l.shape[0]) if
                                   dtmb[j]==i+1 ]))
        std_vlb[i]=np.std(np.array([vb_l[j] for j in xrange(vb_l.shape[0]) if
                                   dtmb[j]==i+1 and wdlb[j]==1]))

    x=np.array(range(1,len(hist_wcs)+1))
    plt.figure(2)
    plt.subplot(111)
    b1=plt.bar(x-1./6, hist_wcs,width=1./3,align='center',label='TD-Stem'+r'$(\lambda)$ ')
    b2=plt.bar(x+1./6, hist_wcl,width=1./3,align='center',label='TD-Leaf'+r'$(\lambda)$ ')
    #plt.title('kqk endgame win conversion rate')
    plt.legend(handles=[b1,b2])
    plt.xlabel('DTM')
    plt.ylabel('WCR')
    plt.xlim(0,x.max())
    plt.ylim(0,1)
    plt.xticks(x,x)
    plt.show()

    plt.figure(3)
    plt.subplot(111)
    b1=plt.bar(x-1./6, hist_wes,width=1./3,align='center',label='TD-Stem'+r'$(\lambda)$ ')
    b2=plt.bar(x+1./6, hist_wel,width=1./3,align='center',label='TD-Leaf'+r'$(\lambda)$ ')
    #plt.title('kqk endgame win efficiency')
    plt.legend(handles=[b1,b2])
    plt.xlabel('DTM')
    plt.ylabel('WE')
    plt.xlim(0,x.max())
    plt.ylim(0,1)
    plt.xticks(x,x)
    plt.show()

    plt.figure(4)
    plt.subplot(111)
    b1=plt.bar(x-1./6, hist_lhss,width=1./3,align='center',label='TD-Stem'+r'$(\lambda)$ ')
    b2=plt.bar(x+1./6, hist_lhsl,width=1./3,align='center',label='TD-Leaf'+r'$(\lambda)$ ')
    #plt.title('kqk endgame loss holding score')
    plt.legend(handles=[b1,b2])
    plt.xlabel('DTM')
    plt.ylabel('LHS')
    plt.xlim(0,x.max())
    plt.ylim(0,1)
    plt.xticks(x,x)
    plt.show()

    plt.figure(5)
    b1,=plt.plot(x,avg_vs,label='TD-Stem'+r'$(\lambda)$ ')
    b2,=plt.plot(x,avg_vl,label='TD-Leaf'+r'$(\lambda)$ ')
    s1,=plt.plot(x,np.array(avg_vs)+2*np.array(std_vs),color='#99ccff')
    s2,=plt.plot(x,np.array(avg_vs)-2*np.array(std_vs),color='#99ccff')
    s3,=plt.plot(x,np.array(avg_vl)+2*np.array(std_vl),color='#ffc266')
    s4,=plt.plot(x,np.array(avg_vl)-2*np.array(std_vl),color='#ffc266')

    c1,=plt.plot(x,avg_vsb,label='TD-Stem'+r'$(\lambda)$ ',color=b1.get_color())
    c2,=plt.plot(x,avg_vlb,label='TD-Leaf'+r'$(\lambda)$ ',color=b2.get_color())
    t1,=plt.plot(x,np.array(avg_vsb)+2*np.array(std_vsb),color='#99ccff')
    t2,=plt.plot(x,np.array(avg_vsb)-2*np.array(std_vsb),color='#99ccff')
    t3,=plt.plot(x,np.array(avg_vlb)+2*np.array(std_vlb),color='#ffc266')
    t4,=plt.plot(x,np.array(avg_vlb)-2*np.array(std_vlb),color='#ffc266')
    plt.xticks(x,x)

    #plt.title('krk endgame win conversion rate')
    plt.legend(handles=[b1,b2])
    plt.xlabel('DTM')
    plt.ylabel('E[V]')
    plt.xlim(0,x.max())
    #plt.ylim(0,1)
    plt.show()


def comparison_3p():
    import tablebases

    with open('Models/3P/TDS4/meta','rb') as f:
        stem=cp.load(f)
    with open('Models/3P/TDL4/meta','rb') as f:
        leaf=cp.load(f)

    settings.init()
    load_DS('dataset/3p_2.epd')
    settings.params['PL']='KQRBNPkqrbnp'
    settings.params['USE_DSET']=True

    N_l=np.array([0]+leaf['N'],dtype=float) 
    N_s=np.array([1]+stem['N'],dtype=float) 
    eps_l=leaf['eps'] 
    eps_s=stem['eps']
    w_l=np.array([0]+leaf['w_list'])/N_l
    w_s=np.array([0]+stem['w_list'])/N_s
    e_l=np.cumsum(N_l)
    e_s=np.cumsum(N_s)

    plt.figure(1)
    plt.subplot(221)
    line_stem, =plt.plot(e_s,w_s,label='TD-Stem'+r'$(\lambda)$')
    line_leaf, =plt.plot(e_l,w_l,label='TD-Leaf'+r'$(\lambda)$')
    plt.xlabel(r'$N$')
    plt.ylabel('winning rate')
    plt.legend(handles=[line_stem])
    #plt.show()

    for i in range(3,8):
        with open('Models/3P/TDL{}/meta'.format(i),'rb') as f:
            leaf=cp.load(f)
        with open('Models/3P/TDS{}/meta'.format(i),'rb') as f:
            stem=cp.load(f)
        print leaf['lambda']

        with open('Models/3P/TDL{}/sim'.format(i),'rb') as f:
            A,evaldict,S=cp.load(f)
        wcl=np.mean(np.array(evaldict['wc']))
        wel=np.mean(np.array(evaldict['we']))
        lhsl=np.mean(np.array(evaldict['lhs']))


        with open('Models/3P/TDS{}/sim'.format(i),'rb') as f:
            A,evaldict,S=cp.load(f)
        wcs=np.mean(np.array(evaldict['wc']))
        wes=np.mean(np.array(evaldict['we']))
        lhss=np.mean(np.array(evaldict['lhs']))
        print '{}: STEM: {}, {}, {} \tLEAF: {}, {}, {}'.format(i,wcs,wes,lhss,wcl,wel,lhsl)
        mps_s=np.mean(np.array(stem['mps']))
        mps_l=np.mean(np.array(leaf['mps']))
        print '{}, {}'.format(mps_s,mps_l)
        print stem['episodes']

    '''
    Sw=[t[-1] for t in tw]
    print Sw[:100]
    dtmw=[t[1] for t in tw]
    wdlw=[t[0] for t in tw]

    vw_s=Approximator.V(Sw,model_fn)

    hist_wcs=90*[0]
    hist_wes=90*[0]
    hist_lhss=90*[0]
    hist_dcs=90*[0]
    avg_vs=90*[0]
    #print A
    for i in xrange(len(hist_wcs)):
        hist_wcs[i]=np.mean(np.array([wc(t) for t in A if wc(t) is not None and 
                    t[1]==i+1]))
        hist_wes[i]=np.mean(np.array([we(t) for t in A if we(t) is not None and 
                    t[1]==i+1]))
        hist_lhss[i]=np.mean(np.array([lhs(t) for t in A if lhs(t) is not None and 
                    t[1]==i+1]))
        hist_dcs[i]=np.mean(np.array([dc(t) for t in A if dc(t) is not None and 
                    t[1]==i+1]))
        avg_vs[i]=np.mean(np.array([vw_s[j] for j in xrange(vw_s.shape[0]) if
                                   dtmw[j]==i and wdlw[j]==1]))

    '''

    '''
    model_fn='Models/3PIECES_LEAF/TDLeaf_3pieces_leaf_5__25_07/TDLeaf_3pieces_leaf_5__25_07-1_67793-40'
    with open('Models/3PIECES_LEAF/sim','rb') as f:
        A,evaldict,S=cp.load(f)
    dcl=np.mean(np.array(evaldict['dc']))

    tw=[t for t in A if tablebases.probe_result(t[-1])==1]
    td=[t for t in A if tablebases.probe_result(t[-1])==0]
    tb=[t for t in A if tablebases.probe_result(t[-1])==-1]

    Sw=[t[-1] for t in tw]
    dtmw=[t[1] for t in tw]
    wdlw=[t[0] for t in tw]

    vw_l=Approximator.V(Sw,model_fn)
    hist_wcl=90*[0]
    hist_wel=90*[0]
    hist_lhsl=90*[0]
    hist_dcl=90*[0]
    avg_vl=90*[0]
    #print A
    for i in xrange(len(hist_wcs)):
        hist_wcl[i]=np.mean(np.array([wc(t) for t in A if wc(t) is not None and 
                    t[1]==i+1]))
        hist_wel[i]=np.mean(np.array([we(t) for t in A if we(t) is not None and 
                    t[1]==i+1]))
        hist_lhsl[i]=np.mean(np.array([lhs(t) for t in A if lhs(t) is not None and 
                    t[1]==i+1]))
        hist_dcl[i]=np.mean(np.array([dc(t) for t in A if dc(t) is not None and 
                    t[1]==i+1]))
        avg_vl[i]=np.mean(np.array([vw_l[j] for j in xrange(vw_l.shape[0]) if
                                   dtmw[j]==i and wdlw[j]==1]))
    '''

    '''
    x=np.array(range(1,len(hist_wcs)+1))
    plt.figure(2)
    plt.subplot(221)
    b1=plt.bar(x-1./6, hist_wcs,width=1./3,align='center',label='TD-Stem'+r'$(\lambda)$ ')
    #b2=plt.bar(x+1./6, hist_wcl,width=1./3,align='center',label='TD-Leaf'+r'$(\lambda)$ ')
    plt.title('krk endgame win conversion rate')
    #plt.legend(handles=[b1])
    plt.xlabel('DTM')
    plt.ylabel('WCR')
    #plt.show()

    #plt.figure(3)
    plt.subplot(222)
    b1=plt.bar(x-1./6, hist_wes,width=1./3,align='center',label='TD-Stem'+r'$(\lambda)$ ')
    #b2=plt.bar(x+1./6, hist_wel,width=1./3,align='center',label='TD-Leaf'+r'$(\lambda)$ ')
    plt.title('krk endgame win efficiency')
    plt.legend(handles=[b1])
    plt.xlabel('DTM')
    plt.ylabel('WE')
    #plt.show()

    #plt.figure(4)
    plt.subplot(223)
    b1=plt.bar(x-1./6, hist_lhss,width=1./3,align='center',label='TD-Stem'+r'$(\lambda)$ ')
    #b2=plt.bar(x+1./6, hist_lhsl,width=1./3,align='center',label='TD-Leaf'+r'$(\lambda)$ ')
    plt.title('krk endgame loss holding score')
    plt.legend(handles=[b1])
    plt.xlabel('DTM')
    plt.ylabel('LHS')

    print 'DC STEM: {}\t DC LEAF: {}'.format(dcs,dcl)
   
    plt.figure(3)
    b1,=plt.plot(x,avg_vs,label='TD-Stem'+r'$(\lambda)$ ')
    #b2,=plt.plot(x,avg_vl,label='TD-Leaf'+r'$(\lambda)$ ')
    s1,=plt.plot(x,np.array(avg_vs)+2*np.array(std_vs),color='#99ccff')
    s2,=plt.plot(x,np.array(avg_vs)-2*np.array(std_vs),color='#99ccff')
    #s3,=plt.plot(x,np.array(avg_vl)+2*np.array(std_vl),color='#ffc266')
    #s4,=plt.plot(x,np.array(avg_vl)-2*np.array(std_vl),color='#ffc266')
    plt.title('krk endgame win conversion rate')
    plt.legend(handles=[b1])
    plt.xlabel('DTM')
    plt.ylabel('E[V]')
    plt.show()
    '''

if __name__=='__main__':
    #comparison_stem_leaf()
    #comparison_stem_leaf_kqk()
    comparison_3p()
