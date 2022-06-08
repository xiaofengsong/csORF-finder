
##This model can predict the csORFs of D.rerio, C.elegans
import tensorflow as tf
import numpy as np
import pandas as pd
import csv
from readFasta import read_fasta
from getProteins import get_protein
from getsORFs import get_sORF
from feature_encoding import extracted_features
from keras.models import load_model
from keras.utils import to_categorical
import Efficient_CapsNet_sORF150
import Efficient_CapsNet_sORF250
import lightgbm as lgb
import argparse



def test_model(inputfile, outputfile, seqtype):
    sORF_seq = read_fasta('../input_files/' + inputfile, seqtype)
    protein_seq = get_protein(sORF_seq)
    fea = extracted_features(sORF_seq, protein_seq, 'M.musculus', 'non-CDS')
    trainX = pd.read_csv('./files/mouse_noncds_trainX_feature250.csv').values
    trainX_ = trainX.reshape([6132, 10, 25, 1])
    y1 = np.array([1] * 3066 + [0] * 3066)
    feature_importance = pd.read_excel('./files/featureRank_lightgbm.xlsx', sheet_name=3, header=None)
    selector = feature_importance.iloc[0:250, 0].tolist()
    fea_sel = fea.loc[:, selector].values
    testX = [[float(i) for i in l] for l in fea_sel]
    testX_ = np.array(testX).reshape([len(sORF_seq), 10, 25, 1])
    ##LightGBM
    lgbClf = lgb.LGBMClassifier(boosting_type='gbdt', learning_rate=0.05, n_estimators=300,
                                max_depth=4, num_leaves=15, max_bin=17, min_data_in_leaf=75,
                                bagging_fraction=0.5, feature_fraction=0.7,
                                lambda_l1=0.9026, lambda_l2=0.6507)
    lgbClf.fit(trainX, y1)
    lgb_proba_testlabel = lgbClf.predict_proba(testX)
    lgb_prob = lgb_proba_testlabel[:, 1]
    ##Efficient-CapsNet
    model = Efficient_CapsNet_sORF250.build_sORF(trainX_.shape[1:], mode='test', verbose=True)
    model_name = './files/mouse_noncds.h5'
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
parser.add_argument('-t', dest='seqtype', type=str, required=True, choices=['sORF', 'other'], help='Query sequence is sORF or other')
args = parser.parse_args()

for file in ([args.inputfile, args.outputfile, args.seqtype]):
 if not (file):
    print(sys.stderr, "\nError: Lack of input file!\n")
    parser.print_help()
    sys.exit(0)


test_model(args.inputfile, args.outputfile, args.seqtype)



