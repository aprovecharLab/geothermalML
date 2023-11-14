#!/usr/bin/env python
# coding: utf-8
# Geothermal and Machine Learning Sandbox

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import distance_transform_cdt

from sklearn import preprocessing 

# SHOULD RENAME THESE FUNCTIONS AND MAKE THIS MORE GENERAL ... FOR EXAMPLE
# MAYBE SHOULD MAKE NEW FUNCTION JUST FOR PROCESSING INTERMEDIATE K FAULT TRACES WITH THREE OPTIONS
#  1. LEAVE ALONE
#  2. 2D GAUSSIAN FILTER
#  3. DISTANCE TO FAULT
#  could also use ones in a nxn convolution to find local density and do this at different scales by preprocessing with
#    skimage.measure.block_reduce(image, block_size, func=np.max, cval=0, func_kwargs=None), e.g., 
    # a = np.array([
    #       [  20,  200,   -5,   23],
    #       [ -13,  134,  119,  100],
    #       [ 120,   32,   49,   25],
    #       [-120,   12,    9,   23]
    # ])
    # skimage.measure.block_reduce(a, (2,2), np.max)
      # this might be used for box counting fractal dimension
#  maybe make use of steerable filters somehow


# +
""
def preprocess_features_LocalCategorical(df_features, df_SSlookup, resetLocal=None, 
                                         transformFaultTraces=None,
                                         extraFeatures=None, prescaleFeatures=None):
    
    #####################################
    # prepare Structural Settings Category lookup dictionary and replace numerical values
    df_SSlookup = df_SSlookup.replace(np.NaN, '')
    df_SSlookup['combinedLabels']= np.sort(df_SSlookup.iloc[:,2:5].values).tolist()
    df_SSlookup['combinedLabels'] = df_SSlookup['combinedLabels'].str.join('-').str.lstrip('-')

    SSlabels = np.sort(df_SSlookup['combinedLabels'].unique())
    
    Idlist = df_SSlookup['Local_polygon_Id'].to_list()
    Idlist.insert(0,0)
    # how do we use 'Local_polygon_overlap_Id'? This implies some multi-valued grid points ... skip for now.
    
    SSlabelslist = df_SSlookup['combinedLabels'].to_list()
#     SSlabelslist.insert(0, 'none')
    SSlabelslist.insert(0, 'noLSS')
    
    lookupDictionary = dict(zip(Idlist, SSlabelslist))
    
    df_features['Local-StructuralSetting'] = df_features['Local_polygon_Id']
    df_features.replace({'Local-StructuralSetting': lookupDictionary}, inplace=True)
    
    #####################################
    # save indexes of nulls for masking maps later on
#     nullIndexesList = df_features.index[df_features['NullInfo']=='nullValue'].to_numpy().tolist()
    nullIndexes = df_features[df_features['NullInfo'] == 'nullValue'].index

    #####################################
    # replace nulls by a value for now
    # df_features = df_features.replace(-9999, np.nan)
    df_features = df_features.replace(-9999, 0.0)

    #####################################
    # make a dataframe of just the extra info
    df_info = df_features[['row', 'column', 'id_rc', \
                           'X_83UTM11', 'Y_83UTM11', 'NullInfo', \
                           'TrainCodeNeg', 'TrainCodePos', 'TrainCodePosT130', \
                           'PosSite130_Id','PosSite130_Distance', \
                           'PosSite_Id', 'PosSite_Distance', \
                           'NegSite_Id', 'NegSite_Distance', \
                           'Local_polygon_Id', 'Local_polygon_overlap_Id']]


    #####################################
    # make working copy of just the features
    feature_names = ['Local-StructuralSetting', \
                     'Local-QuaternaryFaultRecency', \
                     'Local-QuaternaryFaultSlipDilation', \
                     'Local-QuaternaryFaultSlipRate', \
                     'QuaternaryFaultTraces', \
                     'HorizGravityGradient2', \
                     'HorizMagneticGradient2', \
                     'GeodeticStrainRate', \
                     'QuaternarySlipRate', \
                     'FaultRecency', \
                     'FaultSlipDilationTendency2', \
                     'Earthquakes', \
                     'Heatflow']
    
    if extraFeatures:
        feature_names.extend(extraFeatures)
        
    df_features = df_features[feature_names]

    # separate categorical and numerical feature columns
    # localK features are really categorical, but use value as numerical scaled by perceived importance
    colsCategorical = list(range(0,4))
    colsNumerical = list(range(4,len(feature_names)))

    df_categorical_features = df_features[df_features.columns[colsCategorical]].copy()
    df_numerical_features = df_features[df_features.columns[colsNumerical]].copy()


    #####################################
    # Preprocessing Step 1: transform binary fault traces
    if transformFaultTraces:
        
        QFT = df_numerical_features['QuaternaryFaultTraces'].to_numpy().astype(float)
        QFT = np.reshape(QFT,(1000,1728))

        # option: 2D gaussian filter
        if transformFaultTraces == 'gaussianFilter':
            sigma_QFT = 20
            QFT = gaussian_filter(QFT,sigma=sigma_QFT)
        # option: distance to fault trace
        elif transformFaultTraces == 'distance_edt':
            QFT = distance_transform_edt(np.logical_not(QFT))
        elif transformFaultTraces == 'distance_cdt_taxicab':
            QFT = distance_transform_cdt(np.logical_not(QFT),metric='taxicab')
        elif transformFaultTraces == 'distance_cdt_chessboard':
            QFT = distance_transform_cdt(np.logical_not(QFT),metric='chessboard')

        df_numerical_features['QuaternaryFaultTraces'] = np.reshape(QFT,-1)


    #####################################
    # Preprocessing Step 2: drop rows with nulls for embedding and prescaling
    # drop rows with nullValue
    df_numerical_features_noNull = df_numerical_features.copy()
    df_numerical_features_noNull.drop(nullIndexes, inplace=True)

    df_categorical_features_noNull = df_categorical_features.copy()
    df_categorical_features_noNull.drop(nullIndexes, inplace=True)


    #####################################
    # Preprocessing Step 3: do a feature scaler 'fit' to noNull data to be applied to ALL data
    # use standard scaler from scikits-learn - remove mean and scale by standard deviation
    
    # for the future:
    # https://stackoverflow.com/questions/38420847/apply-standardscaler-to-parts-of-a-data-set

    if prescaleFeatures:    
        scaler1 = StandardScaler()
        scaler1.fit(df_numerical_features_noNull)

        # scale numerical features
        df_numerical_features_scaled = scaler1.transform(df_numerical_features)

        # # transform method returns numpy array, so need to make a pandas dataframe again so rest of code works as before
        df_numerical_features_scaled = pd.DataFrame(df_numerical_features_scaled)
    else:
        scaler1 = []
        df_numerical_features_scaled = df_numerical_features.copy()
        
    # Reset null rows to zero
    df_numerical_features_scaled.set_index(df_features.index)
    df_numerical_features_scaled.iloc[nullIndexes] = 0.0


    #####################################
    # Preprocessing Step 4: encoding of categorical features

    #####################################
    # encode all categorical features
    onehot_encoder = preprocessing.OneHotEncoder(sparse=False)
    ordinal_encoder = preprocessing.OrdinalEncoder() 

    categoriesLSS = np.sort(df_categorical_features_noNull['Local-StructuralSetting'].unique())
    categoriesLQFR = np.sort(df_categorical_features_noNull['Local-QuaternaryFaultRecency'].unique())
    categoriesLQFSD = np.sort(df_categorical_features_noNull['Local-QuaternaryFaultSlipDilation'].unique())
    categoriesLQFSR = np.sort(df_categorical_features_noNull['Local-QuaternaryFaultSlipRate'].unique())
    
    onehot_encoder.fit(np.expand_dims(categoriesLSS,1))
    categories = np.expand_dims(df_categorical_features['Local-StructuralSetting'],1)
    onehot_encoded_LSS = onehot_encoder.transform(categories)
    
    ordinal_encoder.fit(np.expand_dims(categoriesLQFR,1))
    categories = np.expand_dims(df_categorical_features['Local-QuaternaryFaultRecency'],1)
    ordinal_encoded_LQFR = ordinal_encoder.transform(categories)

    ordinal_encoder.fit(np.expand_dims(categoriesLQFSD,1))
    categories = np.expand_dims(df_categorical_features['Local-QuaternaryFaultSlipDilation'],1)
    ordinal_encoded_LQFSD = ordinal_encoder.transform(categories)

    ordinal_encoder.fit(np.expand_dims(categoriesLQFSR,1))
    categories = np.expand_dims(df_categorical_features['Local-QuaternaryFaultSlipRate'],1)
    ordinal_encoded_LQFSR = ordinal_encoder.transform(categories)

    df_onehot_features = pd.DataFrame(onehot_encoded_LSS)
    df_ordinal_features = pd.DataFrame(np.hstack([ordinal_encoded_LQFR,
                                                  ordinal_encoded_LQFSD,
                                                  ordinal_encoded_LQFSR]))
    
    #####################################
    # in new ordinal dataframe - drop rows with nullValue for prescaling
    df_ordinal_features_noNull = df_ordinal_features.copy()
    df_ordinal_features_noNull.drop(nullIndexes, inplace=True)
    
#     print(np.unique(df_ordinal_features_noNull.iloc[0].values))

    #####################################
    # Preprocessing Step 4: do a feature scaler 'fit' to noNull data to be applied to ALL data

    if prescaleFeatures:
        scaler2 = StandardScaler()
        scaler2.fit(df_ordinal_features_noNull)

        # scale ordinal features
        ordinal_features_scaled = scaler2.transform(df_ordinal_features)

        # # transform method returns numpy array, so need to make a pandas dataframe again so rest of code works as before
        df_ordinal_features_scaled = pd.DataFrame(ordinal_features_scaled)
    else:
        scaler2 = []
        df_ordinal_features_scaled = df_ordinal_features.copy()

    # Reset null rows to zero
    df_ordinal_features_scaled.set_index(df_features.index)
    df_ordinal_features_scaled.iloc[nullIndexes] = 0.0

    #####################################
    # set local features to "dont care" values if desired
    if resetLocal == 'zeros':
        # set categorical features to 0.0
        df_onehot_features = pd.DataFrame(np.zeros(df_onehot_features.shape))
        df_ordinal_features_scaled = pd.DataFrame(np.zeros(df_ordinal_features_scaled.shape))
    elif resetLocal == 'random':
        # set categorical features to random.uniform(0,1)
        df_onehot_features = pd.DataFrame(np.random.uniform(0, 1, df_onehot_features.shape))
        df_ordinal_features_scaled = pd.DataFrame(np.random.uniform(0, 1, 
                                                    df_ordinal_features_scaled.shape))

    # Reset null rows to zero
    df_onehot_features.set_index(df_features.index)
    df_onehot_features.iloc[nullIndexes] = 0.0
    df_ordinal_features_scaled.set_index(df_features.index)
    df_ordinal_features_scaled.iloc[nullIndexes] = 0.0


    #####################################
    # fix the names
    onehotColumns = []
    for col in list(df_onehot_features.columns):
#         onehotColumns.append('Local-StructuralSetting_'+str(col))
        onehotColumns.append(categoriesLSS[col])

    df_onehot_features.columns = onehotColumns

    df_ordinal_features_scaled.columns = df_categorical_features.columns[1:]
    df_numerical_features_scaled.columns = df_numerical_features.columns
    
    
    #####################################
    # Reassemble Feature Data Frame
    df_features_new = pd.concat([df_onehot_features, 
                                 df_ordinal_features_scaled, 
                                 df_numerical_features_scaled], axis=1)
    
    return df_features_new, df_info, nullIndexes, [scaler1, scaler2]
# -
""
def preprocess_features_LocalNumerical(df_features, resetLocal=None, 
                                       transformFaultTraces=None, 
                                       extraFeatures=None, prescaleFeatures=None):

    #####################################
    # save indexes of nulls for masking maps later on
#     nullIndexesList = df_features.index[df_features['NullInfo']=='nullValue'].to_numpy().tolist()
    nullIndexes = df_features[df_features['NullInfo'] == 'nullValue'].index

    #####################################
    # replace nulls by a value for now
    # df_features = df_features.replace(-9999, np.nan)
    df_features = df_features.replace(-9999, 0.0)

    #####################################
    # make a dataframe of just the extra info
    df_info = df_features[['row', 'column', 'id_rc', \
                           'X_83UTM11', 'Y_83UTM11', 'NullInfo', \
                           'TrainCodeNeg', 'TrainCodePos', 'TrainCodePosT130', \
                           'PosSite130_Id','PosSite130_Distance', \
                           'PosSite_Id', 'PosSite_Distance', \
                           'NegSite_Id', 'NegSite_Distance', \
                           'Local_polygon_Id', 'Local_polygon_overlap_Id']]
 
    #####################################
    # make working copy of just the features
    feature_names = ['Local-StructuralSetting', \
                     'Local-QuaternaryFaultRecency', \
                     'Local-QuaternaryFaultSlipDilation', \
                     'Local-QuaternaryFaultSlipRate', \
                     'QuaternaryFaultTraces', \
                     'HorizGravityGradient2', \
                     'HorizMagneticGradient2', \
                     'GeodeticStrainRate', \
                     'QuaternarySlipRate', \
                     'FaultRecency', \
                     'FaultSlipDilationTendency2', \
                     'Earthquakes', \
                     'Heatflow']
    
    if extraFeatures:
        feature_names.extend(extraFeatures)
        
    df_features = df_features[feature_names]

    # separate categorical and numerical feature columns
    # localK features are really categorical, but use value as numerical scaled by perceived importance
    colsCategorical = list(range(0,4))
    colsNumerical = list(range(4,len(feature_names)))

    df_categorical_features = df_features[df_features.columns[colsCategorical]].copy()
    df_numerical_features = df_features[df_features.columns[colsNumerical]].copy()
    

    #####################################
    # Preprocessing Step 1: transform binary fault traces
    if transformFaultTraces:
        
        QFT = df_numerical_features['QuaternaryFaultTraces'].to_numpy().astype(float)
        QFT = np.reshape(QFT,(1000,1728))

        # option: 2D gaussian filter
        if transformFaultTraces == 'gaussianFilter':
            sigma_QFT = 20
            QFT = gaussian_filter(QFT,sigma=sigma_QFT)
        # option: distance to fault trace
        elif transformFaultTraces == 'distance_edt':
            QFT = distance_transform_edt(np.logical_not(QFT))
        elif transformFaultTraces == 'distance_cdt_taxicab':
            QFT = distance_transform_cdt(np.logical_not(QFT),metric='taxicab')
        elif transformFaultTraces == 'distance_cdt_chessboard':
            QFT = distance_transform_cdt(np.logical_not(QFT),metric='chessboard')

        df_numerical_features['QuaternaryFaultTraces'] = np.reshape(QFT,-1)


    #####################################
    # Preprocessing Step 2: drop rows with nulls for embedding and prescaling
    # drop rows with nullValue
    df_numerical_features_noNull = df_numerical_features.copy()
    df_numerical_features_noNull.drop(nullIndexes, inplace=True)

    df_categorical_features_noNull = df_categorical_features.copy()
    df_categorical_features_noNull.drop(nullIndexes, inplace=True)

    
    #####################################
    # Preprocessing Step 3: do a feature scaler 'fit' to noNull data to be applied to ALL data
    # use standard scaler from scikits-learn - remove mean and scale by standard deviation
    
    # for the future:
    # https://stackoverflow.com/questions/38420847/apply-standardscaler-to-parts-of-a-data-set

    if prescaleFeatures:
        scaler1 = StandardScaler()
        scaler1.fit(df_numerical_features_noNull)

        # scale numerical features
        df_numerical_features_scaled = scaler1.transform(df_numerical_features)

        # # transform method returns numpy array, so need to make a pandas dataframe again so rest of code works as before
        df_numerical_features_scaled = pd.DataFrame(df_numerical_features_scaled)
    else:
        scaler1 = []
        df_numerical_features_scaled = df_numerical_features.copy()

    # Reset null rows to zero
    df_numerical_features_scaled.set_index(df_features.index)
    df_numerical_features_scaled.iloc[nullIndexes] = 0.0
    
    #####################################
    # Preprocessing Step 4: do a feature scaler 'fit' to noNull data to be applied to ALL data

    if prescaleFeatures:
        scaler2 = StandardScaler()
        scaler2.fit(df_categorical_features_noNull)

        # scale categorical features
        df_categorical_features_scaled = scaler2.transform(df_categorical_features)

        # # transform method returns numpy array, so need to make a pandas dataframe again so rest of code works as before
        df_categorical_features_scaled = pd.DataFrame(df_categorical_features_scaled)
    else:
        scaler2 = []
        df_categorical_features_scaled = df_categorical_features.copy()
        
    # Reset null rows to zero
    df_categorical_features_scaled.set_index(df_features.index)
    df_categorical_features_scaled.iloc[nullIndexes] = 0.0

    #####################################
    # set local features to "dont care" values if desired
    if resetLocal == 'zeros':
        # set categorical features to 0.0
        df_categorical_features_scaled = pd.DataFrame(np.zeros(df_categorical_features_scaled.shape))
    elif resetLocal == 'random':
        # set categorical features to random.uniform(0,1)
        df_categorical_features_scaled = pd.DataFrame(np.random.uniform(0, 1, 
                                                        df_categorical_features_scaled.shape))

    # Reset null rows to zero
    df_categorical_features_scaled.set_index(df_features.index)
    df_categorical_features_scaled.iloc[nullIndexes] = 0.0

    #####################################
    # fix the names
    df_numerical_features_scaled.columns = df_numerical_features.columns
    df_categorical_features_scaled.columns = df_categorical_features.columns


    #####################################
    # Reassemble Feature Data Frame
    df_features_new = pd.concat([df_categorical_features_scaled, df_numerical_features_scaled], axis=1)

    return df_features_new, df_info, nullIndexes, [scaler1, scaler2]


""
def preprocess_features_AllNumerical(df_features, transformFeatures=None, 
                                        extraFeatures=None, prescaleFeatures=None, withMean=True):

    #####################################
    # save indexes of nulls for masking maps later on
#     nullIndexesList = df_features.index[df_features['NullInfo']=='nullValue'].to_numpy().tolist()
    nullIndexes = df_features[df_features['NullInfo'] == 'nullValue'].index

    #####################################
    # replace nulls by a value for now
    # df_features = df_features.replace(-9999, np.nan)
    df_features = df_features.replace(-9999, 0.0)

    #####################################
    # make a dataframe of just the extra info
    df_info = df_features[['row', 'column', 'id_rc', \
                           'X_83UTM11', 'Y_83UTM11', 'NullInfo', \
                           'TrainCodeNeg', 'TrainCodePos', 'TrainCodePosT130', \
                           'PosSite130_Id','PosSite130_Distance', \
                           'PosSite_Id', 'PosSite_Distance', \
                           'NegSite_Id', 'NegSite_Distance', \
                           'Local_polygon_Id', 'Local_polygon_overlap_Id']]
 
    #####################################
    # make working copy of just the features
    feature_names = ['Local-StructuralSetting', \
                     'Local-QuaternaryFaultRecency', \
                     'Local-QuaternaryFaultSlipDilation', \
                     'Local-QuaternaryFaultSlipRate', \
                     'QuaternaryFaultTraces', \
                     'HorizGravityGradient2', \
                     'HorizMagneticGradient2', \
                     'GeodeticStrainRate', \
                     'QuaternarySlipRate', \
                     'FaultRecency', \
                     'FaultSlipDilationTendency2', \
                     'Earthquakes', \
                     'Heatflow']
    
    if extraFeatures:
        feature_names.extend(extraFeatures)
        
    df_features = df_features[feature_names]
    
    df_numerical_features = df_features.copy()


    #####################################
    # Preprocessing Step 1: transform features
    if transformFeatures:
        for feature, transform, param in zip(transformFeatures['features'],
                                             transformFeatures['transforms'],
                                             transformFeatures['params']):

            TF = df_numerical_features[feature].to_numpy().astype(float)
            TF = np.reshape(TF,(1000,1728))

            # option: 2D gaussian filter
            if transform == 'gaussianFilter':
                sigma_TF = param
                TF = gaussian_filter(TF,sigma=sigma_TF)
            # option: distance to fault trace
            elif transform  == 'distance_edt':
                TF = distance_transform_edt(np.logical_not(TF))
            elif transform  == 'distance_cdt_taxicab':
                TF = distance_transform_cdt(np.logical_not(TF),metric='taxicab')
            elif transform  == 'distance_cdt_chessboard':
                TF = distance_transform_cdt(np.logical_not(TF),metric='chessboard')

            #####################################
            # set features to "dont care" values if desired
            elif transform == 'zeros':
                # set categorical features to 0.0
                TF = np.zeros(TF.shape)
            elif transform == 'random':
                # set categorical features to random.uniform(0,1)
                TF = np.random.uniform(0, 1, TF.shape)

            df_numerical_features[feature] = np.reshape(TF,-1)


    #####################################
    # Preprocessing Step 2: drop rows with nulls for embedding and prescaling
    # drop rows with nullValue
    df_numerical_features_noNull = df_numerical_features.copy()
    df_numerical_features_noNull.drop(nullIndexes, inplace=True)


    #####################################
    # Preprocessing Step 3: do a feature scaler 'fit' to noNull data to be applied to ALL data
    # use standard scaler from scikits-learn - remove mean and scale by standard deviation
    
    # for the future:
    # https://stackoverflow.com/questions/38420847/apply-standardscaler-to-parts-of-a-data-set

    if prescaleFeatures:
        scaler1 = StandardScaler(with_mean=withMean)
        scaler1.fit(df_numerical_features_noNull)

        # scale numerical features
        df_numerical_features_scaled = scaler1.transform(df_numerical_features)

        # # transform method returns numpy array, so need to make a pandas dataframe again so rest of code works as before
        df_numerical_features_scaled = pd.DataFrame(df_numerical_features_scaled)
    else:
        scaler1 = []
        df_numerical_features_scaled = df_numerical_features.copy()

    # Reset null rows to zero
    df_numerical_features_scaled.set_index(df_features.index)
    df_numerical_features_scaled.iloc[nullIndexes] = 0.0
    

    #####################################
    # fix the names
    df_numerical_features_scaled.columns = df_numerical_features.columns

    #####################################
    # Reassemble Feature Data Frame
    df_features_new = df_numerical_features_scaled

    return df_features_new, df_info, nullIndexes, [scaler1]


""
def makeBenchmarks(df_features, df_info, nullIndexes, trainCode=1, randomize=True, balance=False):
    # specify trainCode <= to use
    # select benchmark sites

    df_features_noNull = df_features.copy()
    df_features_noNull.drop(nullIndexes, inplace=True)

    df_info_noNull = df_info.copy()
    df_info_noNull.drop(nullIndexes, inplace=True)

    TrainCodePos = df_info_noNull['TrainCodePos']
    TrainCodeNeg = df_info_noNull['TrainCodeNeg']

    #####################################
    # choose which sites to use based on distance from known resource
    df_benchmark_pos = df_features_noNull.loc[TrainCodePos <= trainCode].copy()
    df_benchmark_neg = df_features_noNull.loc[TrainCodeNeg <= trainCode].copy()

    df_info_pos = df_info_noNull.loc[TrainCodePos <= trainCode].copy()
    df_info_neg = df_info_noNull.loc[TrainCodeNeg <= trainCode].copy()

    # assign a label of 1, indicating a positive site
    df_benchmark_pos['labels'] = 1
    # assign a label of 0, indicating a negative site
    df_benchmark_neg['labels'] = 0

    npos = len(df_benchmark_pos)
    nneg = len(df_benchmark_neg)
    print ('Number of (+): ',npos, '  ; Number of (-): ',nneg)

    #####################################
    if balance == True:
        # find which is largest, (+) or (-) and truncate to smallest
        # could also select difference at random from study area if negative is deficient
        nmin = min(npos,nneg)
        df_benchmark_pos = df_benchmark_pos.iloc[:nmin].copy()
        df_benchmark_neg = df_benchmark_neg.iloc[:nmin].copy()
        df_info_pos = df_info_pos.iloc[:nmin].copy()
        df_info_neg = df_info_neg.iloc[:nmin].copy()
        print ('number of (+) and (-) truncated to ',nmin)

    #####################################
    # append (+) and (-) examples as a complete data set to use for training and testing
#    df_complete_dataset = df_benchmark_pos.append(df_benchmark_neg, ignore_index = True)
    df_complete_dataset = pd.concat([df_benchmark_pos, df_benchmark_neg], ignore_index = False)
    df_complete_info = pd.concat([df_info_pos, df_info_neg], ignore_index = False)

    print (df_complete_dataset.columns)
    print (df_complete_info.columns)
    
    # df_complete_dataset = df_benchmark_pos.append(df_benchmark_neg, ignore_index = False)
    # df_complete_info = df_info_pos.append(df_info_neg, ignore_index = False)

    #####################################
    if randomize == True:
    # shuffle row by row to mix positive and negative sites
        random_index = np.random.permutation(df_complete_dataset.index)
        df_complete_dataset = df_complete_dataset.reindex(random_index,copy=True)
        df_complete_info = df_complete_info.reindex(random_index,copy=True)

    #####################################
    # separate features and labels
    X = df_complete_dataset.iloc[:,0:-1]
    y = df_complete_dataset.iloc[:,-1]

    return X, y, df_complete_info


""
def makeUnknown(df_features, df_info, nullIndexes, trainCode=12, randomize=True):
    # specify trainCode >= to use
    # select benchmark sites

    df_features_noNull = df_features.copy()
    df_features_noNull.drop(nullIndexes, inplace=True)

    df_info_noNull = df_info.copy()
    df_info_noNull.drop(nullIndexes, inplace=True)

    TrainCodePos = df_info_noNull['TrainCodePos']
    TrainCodeNeg = df_info_noNull['TrainCodeNeg']

    #####################################
    # choose which sites to use based on distance from known resource
    df_benchmark_unk = df_features_noNull.loc[~((TrainCodePos < trainCode) | (TrainCodeNeg < trainCode)),:].copy()

    df_info_unk = df_info_noNull.loc[~((TrainCodePos < trainCode) & (TrainCodeNeg < trainCode)),:].copy()

    # assign a label of 999, indicating an unknown site
    df_benchmark_unk['labels'] = 999

    nunk = len(df_benchmark_unk)
    print ('Number of (unknown): ',nunk)


    #####################################
    df_complete_dataset = df_benchmark_unk
    df_complete_info = df_info_unk

    #####################################
    if randomize == True:
    # shuffle row by row
        random_index = np.random.permutation(df_complete_dataset.index)
        df_complete_dataset = df_complete_dataset.reindex(random_index,copy=True)
        df_complete_info = df_complete_info.reindex(random_index,copy=True)

    #####################################
    # separate features and labels
    X = df_complete_dataset.iloc[:,0:-1]
    y = df_complete_dataset.iloc[:,-1]

    return X, y, df_complete_info


