import tensorflow as tf
import numpy as np
import pandas as pd
import csv
from readFasta import read_fasta
from getProteins import get_protein
from feature_encoding import extracted_features
from keras.models import load_model
from keras.utils import to_categorical
import Efficient_CapsNet_sORF150
import Efficient_CapsNet_sORF250
import lightgbm as lgb
import argparse
import sys

# import warnings
# warnings.filterwarnings("ignore")
# import os
# os.environ['CUDA_visible_devices'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


##prediction model
def test_model(inputfile, outputfile, model):

    sORF_seq = read_fasta('../input_files/' + inputfile, 'sORF')
    protein_seq = get_protein(sORF_seq)
    fea = extracted_features(sORF_seq, protein_seq, model)

    if model == 'H.sapiens-CDS':
        trainX = pd.read_csv('./files/human_cds_trainX_feature150.csv').values
        trainX_ = trainX.reshape([14464, 10, 15, 1])
        y1 = np.array([1] * 7232 + [0] * 7232)
        ##LightGBM model
        lgbClf = lgb.LGBMClassifier(boosting_type='gbdt', learning_rate=0.01, n_estimators=400,
                                    max_depth=5, num_leaves=57, max_bin=99, min_data_in_leaf=66,
                                    bagging_fraction=0.8, feature_fraction=0.7,
                                    lambda_l1=0.2604, lambda_l2=0.7363, verbose=-1)
        lgbClf.fit(trainX, y1)
        ##Efficient-CapsNet model
        model = Efficient_CapsNet_sORF150.build_sORF(trainX_.shape[1:], mode='test', verbose=False)
        feature_importance = pd.read_excel('./files/featureRank_lightgbm.xlsx', sheet_name=0, header=None)
        selector = feature_importance.iloc[0:150, 0].tolist()
        fea_sel = fea.loc[:, selector].values
        testX = [[float(i) for i in l] for l in fea_sel]
        testX_ = np.array(testX).reshape([len(sORF_seq), 10, 15, 1])
        model_name = './files/human_cds.h5'
    elif model == 'H.sapiens-non-CDS':
        trainX = pd.read_csv('./files/human_noncds_trainX_feature250.csv').values
        trainX_ = trainX.reshape([15500, 10, 25, 1])
        y1 = np.array([1] * 7750 + [0] * 7750)
        ##LightGBM model
        lgbClf = lgb.LGBMClassifier(boosting_type='gbdt', learning_rate=0.05, n_estimators=300,
                                    max_depth=5, num_leaves=30, max_bin=138, min_data_in_leaf=121,
                                    bagging_fraction=0.9, feature_fraction=0.6,
                                    lambda_l1=0.6399, lambda_l2=0.4156)
        lgbClf.fit(trainX, y1)
        ##Efficient-CapsNet model
        model = Efficient_CapsNet_sORF250.build_sORF(trainX_.shape[1:], mode='test', verbose=False)
        feature_importance = pd.read_excel('./files/featureRank_lightgbm.xlsx', sheet_name=1, header=None)
        selector = feature_importance.iloc[0:250, 0].tolist()
        fea_sel = fea.loc[:, selector].values
        testX = [[float(i) for i in l] for l in fea_sel]
        testX_ = np.array(testX).reshape([len(sORF_seq), 10, 25, 1])
        model_name = './files/human_noncds.h5'
    elif model == 'M.musculus-CDS':
        trainX = pd.read_csv('./files/mouse_cds_trainX_feature150.csv').values
        trainX_ = trainX.reshape([5230, 10, 15, 1])
        y1 = np.array([1] * 2615 + [0] * 2615)
        ##LightGBM model
        lgbClf = lgb.LGBMClassifier(boosting_type='gbdt', learning_rate=0.1, n_estimators=100,
                                    max_depth=7, num_leaves=41, max_bin=224, min_data_in_leaf=93,
                                    bagging_fraction=0.9, feature_fraction=0.6,
                                    lambda_l1=0.4224, lambda_l2=0.2594)
        lgbClf.fit(trainX, y1)
        ##Efficient-CapsNet model
        model = Efficient_CapsNet_sORF150.build_sORF(trainX_.shape[1:], mode='test', verbose=False)
        feature_importance = pd.read_excel('./files/featureRank_lightgbm.xlsx', sheet_name=2, header=None)
        selector = feature_importance.iloc[0:150, 0].tolist()
        fea_sel = fea.loc[:, selector].values
        testX = [[float(i) for i in l] for l in fea_sel]
        testX_ = np.array(testX).reshape([len(sORF_seq), 10, 15, 1])
        model_name = './files/mouse_cds.h5'
    elif model == 'M.musculus-non-CDS':
        trainX = pd.read_csv('./files/mouse_noncds_trainX_feature250.csv').values
        trainX_ = trainX.reshape([6132, 10, 25, 1])
        y1 = np.array([1] * 3066 + [0] * 3066)
        ##LightGBM
        lgbClf = lgb.LGBMClassifier(boosting_type='gbdt', learning_rate=0.05, n_estimators=300,
                                    max_depth=4, num_leaves=15, max_bin=17, min_data_in_leaf=75,
                                    bagging_fraction=0.5, feature_fraction=0.7,
                                    lambda_l1=0.9026, lambda_l2=0.6507)
        lgbClf.fit(trainX, y1)
        ##Efficient-CapsNet
        model = Efficient_CapsNet_sORF250.build_sORF(trainX_.shape[1:], mode='test', verbose=False)
        feature_importance = pd.read_excel('./files/featureRank_lightgbm.xlsx', sheet_name=3, header=None)
        selector = feature_importance.iloc[0:250, 0].tolist()
        fea_sel = fea.loc[:, selector].values
        testX = [[float(i) for i in l] for l in fea_sel]
        testX_ = np.array(testX).reshape([len(sORF_seq), 10, 25, 1])
        model_name = './files/mouse_noncds.h5'
    elif model == 'D.melanogaster-CDS':
        trainX = pd.read_csv('./files/fruitfly_cds_trainX_feature150.csv').values
        trainX_ = trainX.reshape([1364, 10, 15, 1])
        y1 = np.array([1] * 682 + [0] * 682)
        ##LightGBM model
        lgbClf = lgb.LGBMClassifier(boosting_type='gbdt', learning_rate=0.1, n_estimators=100,
                                    max_depth=5, num_leaves=25, max_bin=48, min_data_in_leaf=94,
                                    bagging_fraction=0.5, feature_fraction=0.8,
                                    lambda_l1=0.3255, lambda_l2=0.5864)
        lgbClf.fit(trainX, y1)
        ##Efficient-CapsNet model
        model = Efficient_CapsNet_sORF150.build_sORF(trainX_.shape[1:], mode='test', verbose=False)
        feature_importance = pd.read_excel('./files/featureRank_lightgbm.xlsx', sheet_name=4, header=None)
        selector = feature_importance.iloc[0:150, 0].tolist()
        fea_sel = fea.loc[:, selector].values
        testX = [[float(i) for i in l] for l in fea_sel]
        testX_ = np.array(testX).reshape([len(sORF_seq), 10, 15, 1])
        model_name = './files/fruitfly_cds.h5'
    elif model == 'D.melanogaster-non-CDS':
        trainX = pd.read_csv('./files/fruitfly_noncds_trainX_feature150.csv').values
        trainX_ = trainX.reshape([13956, 10, 15, 1])
        y1 = np.array([1] * 6978 + [0] * 6978)
        ##LightGBM model
        lgbClf = lgb.LGBMClassifier(boosting_type='gbdt', learning_rate=0.05, n_estimators=300,
                                    max_depth=5, num_leaves=30, max_bin=225, min_data_in_leaf=129,
                                    bagging_fraction=0.9, feature_fraction=0.9,
                                    lambda_l1=0.2516, lambda_l2=0.0407)
        lgbClf.fit(trainX, y1)
        ##Efficient-CapsNet model
        model = Efficient_CapsNet_sORF150.build_sORF(trainX_.shape[1:], mode='test', verbose=False)
        feature_importance = pd.read_excel('./files/featureRank_lightgbm.xlsx', sheet_name=5, header=None)
        selector = feature_importance.iloc[0:150, 0].tolist()
        fea_sel = fea.loc[:, selector].values
        testX = [[float(i) for i in l] for l in fea_sel]
        testX_ = np.array(testX).reshape([len(sORF_seq), 10, 15, 1])
        model_name = './files/fruitfly_noncds.h5'
    else:
        print("Model error")

    lgb_proba_testlabel = lgbClf.predict_proba(testX)
    lgb_prob = lgb_proba_testlabel[:, 1]
    model.load_weights(model_name)
    pred_testlabel, score = model.predict(testX_)
    EffiCaps_prob = pred_testlabel[:, 1]
    prob = (lgb_prob + EffiCaps_prob) / 2
    pre = np.zeros(len(prob))

    print('-' * 30 + 'Finish: test' + '-' * 30)
    coding_potential = []
    for i in range(len(prob)):
        if prob[i] > 0.5:
            pre[i] = 1
            coding_potential.append('Coding')
        else:
            pre[i] = 0
            coding_potential.append('Non-coding')

    with open('../output_files/' + outputfile, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(zip(sORF_seq, coding_potential, prob))



parser = argparse.ArgumentParser(description='csORF-finder: an effective ensemble learning framework for accurate identification of multi-species coding short open reading frames')
parser.add_argument('-i', dest='inputfile', type=str, required=True, help='Query sequences to be predicted in fasta format.')
parser.add_argument('-o', dest='outputfile', type=str, required=False, help='Save the prediction results.')
parser.add_argument('-m', dest='model', type=str, required=False,
                    help='Please choose the model, six options: H.sapiens-CDS, H.sapiens-non-CDS, M.musculus-CDS, M.musculus-non-CDS, D.melanogaster-CDS, D.melanogaster-non-CDS')
args = parser.parse_args()

for file in ([args.inputfile, args.outputfile, args.model]):
 if not (file):
    print(sys.stderr, "\nError: Lack of input file!\n")
    parser.print_help()
    sys.exit(0)

test_model(args.inputfile, args.outputfile, args.model)







