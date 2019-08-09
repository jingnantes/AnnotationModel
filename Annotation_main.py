#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import division
"""
Created on Wed Aug  7 17:06:15 2019
@author: jingli, jing.li.univ@gmail.com

This code is used for the annotator's discrete labeling model
where spamminess -  Bernouli distribution
      malicious voting - Uniform distribution
      normal voting - Multinomial (categorical) distribution
are considered.

EM algorithm is used due to the latent variable z -  spamminess

"""



import numpy as np
import pandas as pd



def Real_world_data(filename):
    #dfs = pd.read_excel(filename)
    dfs = pd.read_csv(filename,sep=',', header=None)
    filedata = dfs.values
    return filedata
    

def Observed_theta(data):
    init_theta_tmp = np.ones((pvs_num,K))
    for e in range(pvs_num):
        idx = [j for j in range(len(data)) if data[j,0]==e]
        tmp = data[idx,2]
        sum_all = len(tmp)
        sum_k = np.zeros((K))
        for kk in range(K):
            sum_k[kk] = len(np.where(tmp==kk+1)[0]) 
            init_theta_tmp[e,kk] = sum_k[kk]/sum_all
    observed_theta = init_theta_tmp
    return observed_theta



def Q_loss(para, mu_t1):
    """ loss of Q function """
    nlabels = len(data)
    avoidZero =  1e-15
    
    (eps,theta,lambda_e) = para
    Q_res = 0
    for ii in range(nlabels):
        e = data[ii,0]; # pvs_idx
        s = data[ii,1]; # obs_idx
        y = data[ii,2]; # observed_label
        Q_res = Q_res + mu_t1[e,s,y-1]*(np.log(eps[s]+avoidZero)+np.log(theta[e,y-1]+avoidZero))+(1-mu_t1[e,s,y-1])*(np.log(1-eps[s]+avoidZero)+np.log(1.0/K))
    for j in range(pvs_num):
        Q_res = Q_res + lambda_e[j]*(np.sum(theta[j,:])-1)
    
    return Q_res     
        
def Q_loss_matrix(para, mu_t1):
    """ loss of Q function """
    
    avoidZero =  1e-15
    
    (eps,theta,lambda_e) = para

    ee = data[:,0]; # pvs_idx
    ss = data[:,1]; # obs_idx
    yy = data[:,2]; # observed_label
        
    tmp1 = mu_t1[ee,ss,yy-1]*(np.log(eps[ss].flatten()+avoidZero)+np.log(theta[ee,yy-1]+avoidZero))+(1-mu_t1[ee,ss,yy-1])*(np.log(1-eps[ss]+avoidZero)+np.log(1/K)).flatten()
    Q_res = np.sum(tmp1)
    

        
    Q_res = Q_res + np.sum(lambda_e.flatten()*(np.sum(theta,1)-1))
    
    return Q_res             

def EM_process(data):
    # data is in the format: pvs_ID, obs_ID, vote
    maxIter = 10
    thresh = 0.0001
    
    ## initialization
    init_eps = 0.5*np.ones((obs_num,1))
    init_theta = Observed_theta(data)
    init_lambda = -np.ones((pvs_num,1))
    init_para = (init_eps,init_theta,init_lambda)
    
    tmp_eps = 0.5*np.ones_like(init_eps)
    tmp_theta = np.zeros_like(init_theta)
    tmp_lambda = -np.ones_like(init_lambda)
    
    Q_old = 0
    
    mu_t1 = np.zeros((pvs_num,obs_num,K))
    for iter in range(maxIter):

        """ E Step"""
        
        
        for i in range(len(data)):
            e = data[i,0]
            ss = data[i,1]
            y = data[i,2]
            #print i
            mu_t1[e,ss,y-1] = (init_eps[ss]*init_theta[e,y-1])/(init_eps[ss]*init_theta[e,y-1]+(1-init_eps[ss])/K)
        
        """ loss of Q function """
        Q_new = Q_loss(init_para, mu_t1)
        print("iter=", iter, "loss=", Q_new)
        
        if abs(Q_new-Q_old) < thresh:
           print("Breaking out since the cost function difference < ", thresh)
           break 
        Q_old = Q_new
        
        """ M step"""
        for ss in range(obs_num):
            idx = [j for j in range(len(data)) if data[j,1]==ss]
            s_pvsnum = len(idx)
            e = data[idx,0]
            y = data[idx,2]
            tmp_eps[ss] = np.sum(mu_t1[e,ss,y-1])/s_pvsnum
        
        "for theta"
        for e in range(pvs_num):
            idx = [j for j in range(len(data)) if data[j,0]==e]
            
            ss = data[idx,1]
            y = data[idx,2]
            for i in range(K):
                idx2 = [j for j in range(len(y)) if y[j]==i+1]
            
                tmp_theta[e,i] = -np.sum(mu_t1[e,ss[idx2],y[idx2]-1])/init_lambda[e]
                
            
        " for lambda"
        for e in range(pvs_num):
            idx = [j for j in range(len(data)) if data[j,0]==e]
            ss = data[idx,1]
            y = data[idx,2]
            tmp_lambda[e] = -np.sum(mu_t1[e,ss,y-1])
            
        init_eps = tmp_eps
        init_theta = tmp_theta
        init_lambda = tmp_lambda
        init_para = (init_eps,init_theta,init_lambda)
        
    if iter >=maxIter:
        print "maximum iteration has been reached"
        
    return init_para, Q_new    
       
def EM_process_matrix(data):
    # data is in the format: pvs_ID, obs_ID, vote
    maxIter = 100
    thresh = 0.0001
    
    ## initialization
    init_eps = 0.5*np.ones((obs_num,1))
    init_theta = Observed_theta(data)
    init_lambda = -np.ones((pvs_num,1))
    init_para = (init_eps,init_theta,init_lambda)
    
    tmp_eps = 0.5*np.ones_like(init_eps)
    tmp_theta = np.zeros_like(init_theta)
    tmp_lambda = -np.ones_like(init_lambda)
    
    Q_old = 0
    
    mu_t1 = np.zeros((pvs_num,obs_num,K))
    #mu_t2 = np.zeros((pvs_num,obs_num,K))
    for iter in range(maxIter):

        """ E Step"""

        tmp1 = init_eps[data[:,1]]*np.reshape(init_theta[data[:,0],data[:,2]-1],[-1,1])
        tmp2 = tmp1/(tmp1+(1-init_eps[data[:,1]])/K)
        mu_t1[data[:,0],data[:,1],data[:,2]-1] = tmp2.flatten()
        
        """ loss of Q function """
        Q_new = Q_loss_matrix(init_para, mu_t1)
        #print("iter=", iter, "loss=", Q_new)
        
        if abs(Q_new-Q_old) < thresh:
           print("Breaking out since the cost function difference < ", thresh)
           break 
        Q_old = Q_new
        
        """ M step"""
        for ss in range(obs_num):
            idx = [j for j in range(len(data)) if data[j,1]==ss]
            s_pvsnum = len(idx)
            e = data[idx,0]
            y = data[idx,2]
            tmp_eps[ss] = np.sum(mu_t1[e,ss,y-1])/s_pvsnum
        
        "for theta"
        for e in range(pvs_num):
            idx = [j for j in range(len(data)) if data[j,0]==e]
            
            ss = data[idx,1]
            y = data[idx,2]
            for i in range(K):
                idx2 = [j for j in range(len(y)) if y[j]==i+1]
            
                tmp_theta[e,i] = -np.sum(mu_t1[e,ss[idx2],y[idx2]-1])/init_lambda[e]
                
            
        " for lambda"
        for e in range(pvs_num):
            idx = [j for j in range(len(data)) if data[j,0]==e]
            ss = data[idx,1]
            y = data[idx,2]
            tmp_lambda[e] = -np.sum(mu_t1[e,ss,y-1])
            
        init_eps = tmp_eps
        init_theta = tmp_theta
        init_lambda = tmp_lambda
        init_para = (init_eps,init_theta,init_lambda)
        
    if iter >=maxIter:
        print "maximum iteration has been reached"
        
    return init_para, Q_new    
       
def Structure_real_data(realdata):
    ## please note that the index of pvs_id and obs_id is not increased with increament of 1
    
    obs_num, realdata, userID = Calculate_obs_number(realdata)    
    pvs_num, realdata, objID = Calculate_pvs_number(realdata)    
    data = realdata
    return data, obs_num, pvs_num, objID, userID
    

def Calculate_obs_number(datatmp):
    set11 = set(datatmp[:,1])
    userID = list(set11)
    userID = np.array(userID)
    tmp_obsid = datatmp[:,1]
    data_tmp = np.zeros_like(tmp_obsid)
    for k in range(len(tmp_obsid)):
        data_tmp[k] = np.where(userID == tmp_obsid[k])[0]
    datatmp[:,1] = data_tmp    
    obs_num_res = len(set11)
    return obs_num_res, datatmp,userID

def Calculate_pvs_number(datatmp):
    set11 = set(datatmp[:,0])
    objID = list(set11)
    objID = np.array(objID)
    tmp_obsid = datatmp[:,0]
    data_tmp = np.zeros_like(tmp_obsid)
    for k in range(len(tmp_obsid)):
        data_tmp[k] = np.where(objID == tmp_obsid[k])[0]
    datatmp[:,0] = data_tmp    
    pvs_num_res = len(set11)
    
    
    return pvs_num_res, datatmp, objID



def Calculate_majority(distribution):
    major = np.argmax(distribution,1)+1
    return major

def Calculate_expectation(distribution):
    expectation = np.zeros((distribution.shape[0],1))
    for i in range(distribution.shape[0]):
        for j in range(K):
            
            expectation[i] = expectation[i]+(j+1)*distribution[i,j]
            
        
    return expectation     


    


def main_realdata(data, pvs_num, obs_num, K):
    
    est_para, loss = EM_process(data)

if __name__ == "__main__":
    
    
    """ for real world data"""
    
    data_path = './data/'
    
    """----- VQEG HD -------"""
#    observation_file = 'VQEG_HD/data_VQEGHD.csv'
#    data_name = 'VQEG_HD'
#    K = 5
#    
    
    """----- FTV -------"""
    observation_file = 'FTV/data_FTV.csv'
    data_name = 'FTV'
    K = 5
    
    
    """----- UHD4U -------"""
#    observation_file = 'UHD4U/data_UHD4U.csv'
#    data_name = 'UHD4U'
#    K = 5
#    
    
    
    """ processing data """
    rawdata = Real_world_data(data_path+observation_file)
    data, obs_num, pvs_num, objID, userID = Structure_real_data (rawdata)
    
    data = data.astype(int)
   
    est_para, loss = EM_process_matrix(data)
    eps_s, theta_s, lambda_s = est_para
    
    
    predict = Calculate_expectation(theta_s)
    
    res = np.c_[objID, predict]
    annotator = np.c_[userID, eps_s]
    
    
    """save data"""
 
    ## save estimated ground truth 
    df= pd.DataFrame(res)
    df.to_csv("./res/"+data_name+"_predicted_mos.csv", header=None,index=None)
    ## save estimated annotator reliability
    df= pd.DataFrame(annotator)
    df.to_csv("./res/"+data_name+"_predicted_annotator.csv", header=None, index=None)
    
    
    
   