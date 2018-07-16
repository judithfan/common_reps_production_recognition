## jefan 
## Created 2/21/16 to help with standard analyses for drawing experiment data

import numpy as np
import os
import sys
import matplotlib
from matplotlib import pylab, mlab, pyplot
plt = pyplot

from pylab import *
from numpy import *

import pandas as pd

## modify python path in order to load in useful variables from current directory
CURR_DIR = os.getcwd()
if os.path.join(CURR_DIR) not in sys.path:
    sys.path.append(CURR_DIR)
    
import clusterPairList as cpl
import clusterObjList as col    

def computeRankdiff(X,dv):

    XPP = X[X['phase'] != 2]
    Rankdifftitc = []
    Rankdiffuitc = []
    Rankdiffuiuc = []

    Preranktitc = []
    Postranktitc = []
    Prerankuitc = []
    Postrankuitc = []
    Prerankuiuc = []
    Postrankuiuc = []

    workers = []
    wid = []
    exp = []

    idvar = 'sID'
    sessions = np.unique(XPP[idvar].values)
    for s_ind,s in enumerate(sessions):
        if s_ind%100==0:
            print 'Analyzing {} of {} participants...'.format(s_ind,len(sessions))
        xpps = XPP[XPP[idvar] == s]
        try:
            xpps_titc = xpps[xpps['TITC'] == 1]
            x1 = xpps_titc[xpps_titc['phase'] == 1][['obj', dv]]
            x1.sort_values(by=['obj'],inplace=True)
            x3 = xpps_titc[xpps_titc['phase'] == 3][['obj', dv]]
            x3.sort_values(by=['obj'],inplace=True)
            rankdifftitc = (x3[dv].values - x1[dv].values).tolist()
            preranktitc = x1[dv].values.tolist()
            postranktitc = x3[dv].values.tolist()  
            if not (x1['obj'].values == x3['obj'].values).all():
                print('Bad: %s' % s)
                
            xpps_uitc = xpps[xpps['UITC'] == 1]
            x1 = xpps_uitc[xpps_uitc['phase'] == 1][['obj', dv]]
            x1.sort_values(by=['obj'],inplace=True)
            x3 = xpps_uitc[xpps_uitc['phase'] == 3][['obj', dv]]
            x3.sort_values(by=['obj'],inplace=True)
            rankdiffuitc = (x3[dv].values - x1[dv].values).tolist()
            prerankuitc = x1[dv].values.tolist()
            postrankuitc = x3[dv].values.tolist()
            if not (x1['obj'].values == x3['obj'].values).all():
                print('Bad: %s' % s)
                
        
            xpps_uiuc = xpps[xpps['UIUC'] == 1]
            x1 = xpps_uiuc[xpps_uiuc['phase'] == 1][['obj', dv]]
            x1.sort_values(by=['obj'],inplace=True)
            x3 = xpps_uiuc[xpps_uiuc['phase'] == 3][['obj', dv]]
            x3.sort_values(by=['obj'],inplace=True)
            rankdiffuiuc = (x3[dv].values - x1[dv].values).tolist()
            prerankuiuc = x1[dv].values.tolist()
            postrankuiuc = x3[dv].values.tolist()
            if not (x1['obj'].values == x3['obj'].values).all():
                print('Bad: %s' % s)  
            
            workers.append(s)
            
        except Exception as e: 
            print(e)  
            print("Bad session: %s" % s)        
        else:
            Rankdifftitc.append(rankdifftitc)
            Rankdiffuitc.append(rankdiffuitc)
            Rankdiffuiuc.append(rankdiffuiuc)
            Preranktitc.append(preranktitc)
            Postranktitc.append(postranktitc)
            Prerankuitc.append(prerankuitc)
            Postrankuitc.append(postrankuitc)
            Prerankuiuc.append(prerankuiuc)
            Postrankuiuc.append(postrankuiuc)
            exp.append(np.unique(xpps['expid'].values)[0])
            wid.append(np.unique(xpps[idvar].values)[0])

            
    Rankdifftitc = np.array(Rankdifftitc)
    Rankdiffuitc = np.array(Rankdiffuitc)
    Rankdiffuiuc = np.array(Rankdiffuiuc)

    Preranktitc = np.array(Preranktitc)
    Prerankuitc = np.array(Prerankuitc)
    Prerankuiuc = np.array(Prerankuiuc)

    Postranktitc = np.array(Postranktitc)
    Postrankuitc = np.array(Postrankuitc)
    Postrankuiuc = np.array(Postrankuiuc)

    Rankdifftitc_all = Rankdifftitc
    Rankdiffuitc_all = Rankdiffuitc
    Rankdiffuiuc_all = Rankdiffuiuc

    wid = np.array(wid)
    exp = np.array(exp)
    print('Finished analyzing {} participants!'.format(len(sessions)))
    
    return Rankdifftitc,Rankdiffuitc,Rankdiffuiuc,Preranktitc,Prerankuitc,Prerankuiuc, Postranktitc,Postrankuiuc,Postrankuiuc,wid,exp

def getTrainingTimecourse(X):
    '''
    returns Ranktrainmat, Traintrialmat, Ranktraintimecourse
    '''
    XPP = X[X['phase'] == 2]
    
    numTrained = 4
    numReps = 5
    numTrials = numTrained*numReps
    a = np.arange(numTrained*numReps)
    a.shape = (numTrained,numReps) # initialize array with object x repetition

    Ranktrainmat = []
    Traintrialmat = []
    Ranktraintimecourse = []
    Top32timecourse = []
    Top16timecourse = []
    Top8timecourse = []
    Top4timecourse = []
    Top1timecourse = []

    workers = []
    dv = 'rank'
    idvar = 'sID'    
    sessions = np.unique(X[idvar])        
    for s_ind,s in enumerate(sessions):
        if s_ind%100==0:
            print 'Analyzing {} of {} participants...'.format(s_ind,len(sessions))        
        ## if filtered by cue-type
        xpps = XPP[XPP[idvar] == s]
        if sum(xpps['expid']==1)>-1: ## filter by image (set to 0) /verbal (set to >0) cue type only; neutral = >-1
            xpps.sort_values(by=['obj','trial'],inplace=True)
            try:
                assert xpps.shape[0]==20
                _ranktrainmat = xpps[dv].values
                ranktrainmat = np.reshape(_ranktrainmat,(numTrained,numReps))
                ranktraintimecourse = np.mean(ranktrainmat,0)

                # get topk timecourse (i.e., proportion of trials on each rep that rank was <= k)
                top32 = np.zeros(numTrials).reshape(numTrained,numReps).astype(bool)
                top16 = np.zeros(numTrials).reshape(numTrained,numReps).astype(bool)
                top8 = np.zeros(numTrials).reshape(numTrained,numReps).astype(bool)
                top4 = np.zeros(numTrials).reshape(numTrained,numReps).astype(bool)
                top1 = np.zeros(numTrials).reshape(numTrained,numReps).astype(bool)

                top32[np.where(ranktrainmat<=32)] = 1
                top16[np.where(ranktrainmat<=16)] = 1
                top8[np.where(ranktrainmat<=8)] = 1
                top4[np.where(ranktrainmat<=4)] = 1
                top1[np.where(ranktrainmat==1)] = 1

                top32 = top32.mean(0)
                top16 = top16.mean(0)
                top8 = top8.mean(0)
                top4 = top4.mean(0)
                top1 = top1.mean(0)
    
                _traintrialmat = xpps['trial'].values
                traintrialmat = np.reshape(_traintrialmat,(numTrained,numReps))

                workers.append(s)
            except Exception as e: 
                print(e)  
                print("Bad session: %s" % s)
                pass

            try:
                if len(Ranktrainmat) == 0:
                    Ranktrainmat = ranktrainmat
                    Traintrialmat = traintrialmat 
                    Ranktraintimecourse = ranktraintimecourse
                    Top32timecourse = top32
                    Top16timecourse = top16
                    Top8timecourse = top8
                    Top4timecourse = top4
                    Top1timecourse = top1
                else:
                    Ranktrainmat = np.dstack((Ranktrainmat,ranktrainmat))
                    Traintrialmat = np.dstack((Traintrialmat,traintrialmat))
                    Ranktraintimecourse = np.vstack((Ranktraintimecourse,ranktraintimecourse))
                    Top32timecourse = np.vstack((Top32timecourse,top32))
                    Top16timecourse = np.vstack((Top16timecourse,top16))
                    Top8timecourse = np.vstack((Top8timecourse,top8))
                    Top4timecourse = np.vstack((Top4timecourse,top4))
                    Top1timecourse = np.vstack((Top1timecourse,top1))

            except NameError:
                pass  
    print('Done analyzing {} participants!'.format(len(sessions)))            
    return Ranktrainmat, Traintrialmat, Ranktraintimecourse, Top32timecourse, Top16timecourse, Top8timecourse, Top4timecourse, Top1timecourse 

def subsampleUIUC(Rankdiffuiuc,Prerankuiuc):
    import random
    # subsample UIUC in order to match for power
    Rankdiffuiuc_sub = []
    Prerankuiuc_sub = []
    for i in range(len(Rankdiffuiuc)):
        inds = np.random.RandomState(i).choice(range(8),4,replace=False)    
        if i == 0:
            Rankdiffuiuc_sub = Rankdiffuiuc[i][inds]
            Prerankuiuc_sub = Prerankuiuc[i][inds]        
        else:
            Rankdiffuiuc_sub = np.vstack((Rankdiffuiuc_sub,Rankdiffuiuc[i][inds]))
            Prerankuiuc_sub = np.vstack((Prerankuiuc_sub,Prerankuiuc[i][inds]))  
    return Rankdiffuiuc_sub,Prerankuiuc_sub

def getSubjMean(array):
    sm = []
    for r in array:
        m = np.mean(r)
        if math.isnan(m)==False:
            sm.append(m)
    return np.array(sm)

def compute_sem(x):
    return np.std(x)/np.sqrt(len(x))

def remove_nans(x):
    y = x[~np.isnan(x)]
    return y

def rmse(a):
    return np.sqrt(np.mean(map(np.square,a)))    

def bootstrap_subjects(data,nIter=1000):
    boot_mean = []
    for i in np.arange(nIter):
        boot_inds = np.random.RandomState(i).choice(data.shape[0],data.shape[0])
        boot_mean.append(np.mean(data[boot_inds],0))
    return np.array(boot_mean)