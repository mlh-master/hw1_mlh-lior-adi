# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 17:14:23 2019

@author: smorandv
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rm_ext_and_nan(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A dictionary of clean CTG called c_ctg
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    CTG_features = CTG_features.apply(lambda features: pd.to_numeric(features, errors='coerce')).drop(columns=[extra_feature], inplace=False)
    c_ctg = {}
    for col in CTG_features.columns:
        c_ctg[col] = CTG_features[col].dropna()
    # --------------------------------------------------------------------------
    return c_ctg


def nan2num_samp(CTG_features, extra_feature):
    """

    :param CTG_features: Pandas series of CTG features
    :param extra_feature: A feature to be removed
    :return: A pandas dataframe of the dictionary c_cdf containing the "clean" features
    """
    c_cdf = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    CTG_features = CTG_features.apply(lambda features: pd.to_numeric(features, errors='coerce')).drop(columns=[extra_feature], inplace=False)
    for col in CTG_features.columns:
        possible_choices = list(CTG_features[col].dropna().values)
        c_cdf[col] = CTG_features[col].fillna(pd.Series(np.random.choice(possible_choices, size=len(possible_choices))))
    
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_cdf)


def sum_stat(c_feat):
    """

    :param c_feat: Output of nan2num_cdf
    :return: Summary statistics as a dicionary of dictionaries (called d_summary) as explained in the notebook
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    d_summary ={}
    for col in c_feat.columns:
        #print(c_feat[col])
        #temp = list(c_feat[col])
        #print(temp[0])
        dic ={}
        dic["min"] = c_feat[col].describe()[3]
        dic["Q1"] = c_feat[col].describe()[4]
        dic["median"] = c_feat[col].describe()[5]
        dic["Q3"] = c_feat[col].describe()[6]
        dic["max"] = c_feat[col].describe()[7]
        d_summary[col] = dic
        
        
    # -------------------------------------------------------------------------
    return d_summary


def rm_outlier(c_feat, d_summary):
    """

    :param c_feat: Output of nan2num_cdf
    :param d_summary: Output of sum_stat
    :return: Dataframe of the dictionary c_no_outlier containing the feature with the outliers removed
    """
    c_no_outlier = {}
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    c_feat1 = c_feat.copy()
    for col in c_feat1.columns:
        LW = d_summary[col]["Q1"]-1.5*(d_summary[col]["Q3"]-d_summary[col]["Q1"])
        RW = d_summary[col]["Q3"]+1.5*(d_summary[col]["Q3"]-d_summary[col]["Q1"])
        c_feat1[col] = c_feat1[col].apply(lambda x: np.where(x < LW or x > RW,np.nan,x))
        c_no_outlier[col] = c_feat1[col]
    # -------------------------------------------------------------------------
    return pd.DataFrame(c_no_outlier)


def phys_prior(c_cdf, feature, thresh):
    """

    :param c_cdf: Output of nan2num_cdf
    :param feature: A string of your selected feature
    :param thresh: A numeric value of threshold
    :return: An array of the "filtered" feature called filt_feature
    """
    # ------------------ IMPLEMENT YOUR CODE HERE:-----------------------------
    filt_feature = c_cdf[feature].apply(lambda x: np.where(x > thresh,np.nan,x))
    # -------------------------------------------------------------------------
    return filt_feature


def norm_standard(CTG_features, selected_feat=('LB', 'ASTV'), mode='none', flag=False):
    """

    :param CTG_features: Pandas series of CTG features
    :param selected_feat: A two elements tuple of strings of the features for comparison
    :param mode: A string determining the mode according to the notebook
    :param flag: A boolean determining whether or not plot a histogram
    :return: Dataframe of the normalized/standardazied features called nsd_res
    """
    x, y = selected_feat
    # ------------------ IMPLEMENT YOUR CODE HERE:------------------------------
    if mode == 'standard':
        #scaler = preprocessing.StandardScaler()
        #nsd_res = scaler.fit_transform(CTG_features)
        nsd_res = (CTG_features - CTG_features.mean()) / (CTG_features.std())
    elif mode == 'MinMax':
        #scaler = preprocessing.MinMaxScaler()
        #nsd_res = scaler.fit_transform(CTG_features)
        nsd_res = (CTG_features - CTG_features.min()) / (CTG_features.max()-CTG_features.min())
    elif mode == 'mean':
        nsd_res = (CTG_features - CTG_features.mean()) / (CTG_features.max()-CTG_features.min())
    else:
        nsd_res = CTG_features
    if flag == True:
        nsd_res[x].hist(figsize=(40,15))
        CTG_features[x].hist()
        plt.title(x,fontsize=50)
        plt.xlabel(x,fontsize=40)
        plt.ylabel('count',fontsize=40)
        plt.legend([mode,'Original'],fontsize=40)
        plt.show()
        
        nsd_res[y].hist()
        CTG_features[y].hist(figsize=(40,15))
        plt.title(y,fontsize=50)
        plt.xlabel(y,fontsize=40)
        plt.ylabel('count',fontsize=40)
        plt.legend([mode,'Original'],fontsize=40)
        plt.show()
    # -------------------------------------------------------------------------
    return pd.DataFrame(nsd_res)
