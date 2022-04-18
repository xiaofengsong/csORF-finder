import tensorflow as tf
import numpy as np
import scipy.io as sio
import pandas as pd
import os
import csv
from feature_encoding import *
from keras.models import load_model
from keras.utils import to_categorical
import Efficient_CapsNet_sORF150
import Efficient_CapsNet_sORF250
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
import sys
from optparse import OptionParser

##read Fasta sequence
def readFasta(file):
    if os.path.exists(file) == False:
        print('Error: "' + file + '" does not exist.')
        sys.exit(1)

    with open(file) as f:
        records = f.read()

    if re.search('>', records) == None:
        print('The input file seems not in fasta format.')
        sys.exit(1)

    records = records.split('>')[1:]
    myFasta = []
    for fasta in records:
        array = fasta.split('\n')
        name, sequence = array[0].split()[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(array[1:]).upper())
        myFasta.append([name, sequence])
    return myFasta

##extract sORF sequence
def get_sORF(fastas):
    sORF_seq = []
    for i in fastas:
        name, seq = i[0], re.sub('-', '', i[1])
        g = 0
        if len(seq) > 303:
            for j in range(len(seq)-2):
                seg_start = seq[j:j+3]
                if seg_start == 'ATG':
                    for k in range(j+3, len(seq)-2, 3):
                        seg_end = seq[k:k+3]
                        if seg_end == 'TAA':
                            sequence = seq[j:k+3]
                            if np.mod(len(sequence), 3) == 0 and 12 <= len(sequence) <= 303:
                                g+=1
                                sequence_name = '>' + name + '_sORF' + str(g)
                                sORF_seq.append([sequence_name, sequence])
                            break
                        if seg_end == 'TAG':
                            sequence = seq[j:k+3]
                            if np.mod(len(sequence), 3) == 0 and 12 <= len(sequence) <= 303:
                                g+=1
                                sequence_name = '>' + name + '_sORF' + str(g)
                                sORF_seq.append([sequence_name, sequence])
                            break
                        if seg_end == 'TGA':
                            sequence = seq[j:k+3]
                            if np.mod(len(sequence), 3) == 0 and 12 <= len(sequence) <= 303:
                                g+=1
                                sequence_name = '>' + name + '_sORF' + str(g)
                                sORF_seq.append([sequence_name, sequence])
                            break
        elif len(seq) <= 303 and np.mod(len(seq), 3) != 0:
            for j in range(len(seq)-2):
                seg_start = seq[j:j+3]
                if seg_start == 'ATG':
                    for k in range(j+3, len(seq)-2, 3):
                        seg_end = seq[k:k+3]
                        if seg_end == 'TAA':
                            sequence = seq[j:k+3]
                            if np.mod(len(sequence), 3) == 0 and 12 <= len(sequence) <= 303:
                                g+=1
                                sequence_name = '>' + name + '_sORF' + str(g)
                                sORF_seq.append([sequence_name, sequence])
                            break
                        if seg_end == 'TAG':
                            sequence = seq[j:k+3]
                            if np.mod(len(sequence), 3) == 0 and 12 <= len(sequence) <= 303:
                                g+=1
                                sequence_name = '>' + name + '_sORF' + str(g)
                                sORF_seq.append([sequence_name, sequence])
                            break
                        if seg_end == 'TGA':
                            sequence = seq[j:k+3]
                            if np.mod(len(sequence), 3) == 0 and 12 <= len(sequence) <= 303:
                                g+=1
                                sequence_name = '>' + name + '_sORF' + str(g)
                                sORF_seq.append([sequence_name, sequence])
                            break
        elif seq[0:3] == 'ATG' and seq[len(seq)-3:len(seq)] == 'TAA' and np.mod(len(seq), 3) == 0 and 12 <= len(seq) <= 303:
            sORF_seq.append([name, seq])
        elif seq[0:3] == 'ATG' and seq[len(seq)-3:len(seq)] == 'TAG' and np.mod(len(seq), 3) == 0 and 12 <= len(seq) <= 303:
            sORF_seq.append([name, seq])
        elif seq[0:3] == 'ATG' and seq[len(seq)-3:len(seq)] == 'TGA' and np.mod(len(seq), 3) == 0 and 12 <= len(seq) <= 303:
            sORF_seq.append([name, seq])
    return sORF_seq

##get protein sequence
def get_protein(fastas):
    protein_seq=[]
    start_codon = 'ATG'
    codon_table = {
        'ATA': 'I', 'ATC': 'I', 'ATT': 'I', 'ATG': 'M',
        'ACA': 'T', 'ACC': 'T', 'ACG': 'T', 'ACT': 'T',
        'AAC': 'N', 'AAT': 'N', 'AAA': 'K', 'AAG': 'K',
        'AGC': 'S', 'AGT': 'S', 'AGA': 'R', 'AGG': 'R',
        'CTA': 'L', 'CTC': 'L', 'CTG': 'L', 'CTT': 'L',
        'CCA': 'P', 'CCC': 'P', 'CCG': 'P', 'CCT': 'P',
        'CAC': 'H', 'CAT': 'H', 'CAA': 'Q', 'CAG': 'Q',
        'CGA': 'R', 'CGC': 'R', 'CGG': 'R', 'CGT': 'R',
        'GTA': 'V', 'GTC': 'V', 'GTG': 'V', 'GTT': 'V',
        'GCA': 'A', 'GCC': 'A', 'GCG': 'A', 'GCT': 'A',
        'GAC': 'D', 'GAT': 'D', 'GAA': 'E', 'GAG': 'E',
        'GGA': 'G', 'GGC': 'G', 'GGG': 'G', 'GGT': 'G',
        'TCA': 'S', 'TCC': 'S', 'TCG': 'S', 'TCT': 'S',
        'TTC': 'F', 'TTT': 'F', 'TTA': 'L', 'TTG': 'L',
        'TAC': 'Y', 'TAT': 'Y', 'TAA': '', 'TAG': '',
        'TGC': 'C', 'TGT': 'C', 'TGA': '', 'TGG': 'W'}

    for i in fastas:
        name, seq = i[0], re.sub('-', '', i[1])
        start_site = re.search(start_codon, seq)
        protein = ''
        for site in range(start_site.start(), len(seq), 3):
            protein = protein + codon_table[seq[site:site+3]]
        protein_name = '>Micropeptide_' + name
        protein_seq.append([protein_name, protein])
    return protein_seq

##extract features
def feature_encode(datapath, dna_seq, protein_seq, s_type, d_type):
    if s_type == 'H.sapiens':
        if d_type == 'CDS':
            c_m = pd.read_csv(datapath + 'human_cds_trainp_6mermean.csv', header=None, delimiter=',')
            nc_m = pd.read_csv(datapath + 'human_cds_trainn_6mermean.csv', header=None, delimiter=',')
            Tc_pos1 = pd.read_csv(datapath + 'human_cds_trainp_framed_3mer_1.csv', header=None, delimiter=',')
            Tc_neg1 = pd.read_csv(datapath + 'human_cds_trainn_framed_3mer_1.csv', header=None, delimiter=',')
            Tc_pos2 = pd.read_csv(datapath + 'human_cds_trainp_framed_3mer_2.csv', header=None, delimiter=',')
            Tc_neg2 = pd.read_csv(datapath + 'human_cds_trainn_framed_3mer_2.csv', header=None, delimiter=',')
            Tc_pos3 = pd.read_csv(datapath + 'human_cds_trainp_framed_3mer_3.csv', header=None, delimiter=',')
            Tc_neg3 = pd.read_csv(datapath + 'human_cds_trainn_framed_3mer_3.csv', header=None, delimiter=',')
            fea1_1 = np.array(ratio_ORFlength_hcds(dna_seq))
            dna_fea = np.array(extract_DNAfeatures(dna_seq, c_m, nc_m, Tc_pos1, Tc_neg1, Tc_pos2, Tc_neg2, Tc_pos3, Tc_neg3, fea1_1))
        elif d_type == 'non-CDS':
            c_m = pd.read_csv(datapath + 'human_noncds_trainp_6mermean.csv', header=None, delimiter=',')
            nc_m = pd.read_csv(datapath + 'human_noncds_trainn_6mermean.csv', header=None, delimiter=',')
            Tc_pos1 = pd.read_csv(datapath + 'human_noncds_trainp_framed_3mer_1.csv', header=None, delimiter=',')
            Tc_neg1 = pd.read_csv(datapath + 'human_noncds_trainn_framed_3mer_1.csv', header=None, delimiter=',')
            Tc_pos2 = pd.read_csv(datapath + 'human_noncds_trainp_framed_3mer_2.csv', header=None, delimiter=',')
            Tc_neg2 = pd.read_csv(datapath + 'human_noncds_trainn_framed_3mer_2.csv', header=None, delimiter=',')
            Tc_pos3 = pd.read_csv(datapath + 'human_noncds_trainp_framed_3mer_3.csv', header=None, delimiter=',')
            Tc_neg3 = pd.read_csv(datapath + 'human_noncds_trainn_framed_3mer_3.csv', header=None, delimiter=',')
            fea1_1 = np.array(ratio_ORFlength_hnoncds(dna_seq))
            dna_fea = np.array(extract_DNAfeatures(dna_seq, c_m, nc_m, Tc_pos1, Tc_neg1, Tc_pos2, Tc_neg2, Tc_pos3, Tc_neg3, fea1_1))
        else:
            print("Type error")
    elif s_type == 'M.musculus':
        if d_type == 'CDS':
            c_m = pd.read_csv(datapath + 'mouse_cds_trainp_6mermean.csv', header=None, delimiter=',')
            nc_m = pd.read_csv(datapath + 'mouse_cds_trainn_6mermean.csv', header=None, delimiter=',')
            Tc_pos1 = pd.read_csv(datapath + 'mouse_cds_trainp_framed_3mer_1.csv', header=None, delimiter=',')
            Tc_neg1 = pd.read_csv(datapath + 'mouse_cds_trainn_framed_3mer_1.csv', header=None, delimiter=',')
            Tc_pos2 = pd.read_csv(datapath + 'mouse_cds_trainp_framed_3mer_2.csv', header=None, delimiter=',')
            Tc_neg2 = pd.read_csv(datapath + 'mouse_cds_trainn_framed_3mer_2.csv', header=None, delimiter=',')
            Tc_pos3 = pd.read_csv(datapath + 'mouse_cds_trainp_framed_3mer_3.csv', header=None, delimiter=',')
            Tc_neg3 = pd.read_csv(datapath + 'mouse_cds_trainn_framed_3mer_3.csv', header=None, delimiter=',')
            fea1_1 = np.array(ratio_ORFlength_mcds(dna_seq))
            dna_fea = np.array(extract_DNAfeatures(dna_seq, c_m, nc_m, Tc_pos1, Tc_neg1, Tc_pos2, Tc_neg2, Tc_pos3, Tc_neg3, fea1_1))
        elif d_type == 'non-CDS':
            c_m = pd.read_csv(datapath + 'mouse_noncds_trainp_6mermean.csv', header=None, delimiter=',')
            nc_m = pd.read_csv(datapath + 'mouse_noncds_trainn_6mermean.csv', header=None, delimiter=',')
            Tc_pos1 = pd.read_csv(datapath + 'mouse_noncds_trainp_framed_3mer_1.csv', header=None, delimiter=',')
            Tc_neg1 = pd.read_csv(datapath + 'mouse_noncds_trainn_framed_3mer_1.csv', header=None, delimiter=',')
            Tc_pos2 = pd.read_csv(datapath + 'mouse_noncds_trainp_framed_3mer_2.csv', header=None, delimiter=',')
            Tc_neg2 = pd.read_csv(datapath + 'mouse_noncds_trainn_framed_3mer_2.csv', header=None, delimiter=',')
            Tc_pos3 = pd.read_csv(datapath + 'mouse_noncds_trainp_framed_3mer_3.csv', header=None, delimiter=',')
            Tc_neg3 = pd.read_csv(datapath + 'mouse_noncds_trainn_framed_3mer_3.csv', header=None, delimiter=',')
            fea1_1 = np.array(ratio_ORFlength_mnoncds(dna_seq))
            dna_fea = np.array(extract_DNAfeatures(dna_seq, c_m, nc_m, Tc_pos1, Tc_neg1, Tc_pos2, Tc_neg2, Tc_pos3, Tc_neg3, fea1_1))
        else:
            print("Type error")
    elif s_type == 'D.melanogaster':
        if d_type == 'CDS':
            c_m = pd.read_csv(datapath + 'fruitfly_cds_trainp_6mermean.csv', header=None, delimiter=',')
            nc_m = pd.read_csv(datapath + 'fruitfly_cds_trainn_6mermean.csv', header=None, delimiter=',')
            Tc_pos1 = pd.read_csv(datapath + 'fruitfly_cds_trainp_framed_3mer_1.csv', header=None, delimiter=',')
            Tc_neg1 = pd.read_csv(datapath + 'fruitfly_cds_trainn_framed_3mer_1.csv', header=None, delimiter=',')
            Tc_pos2 = pd.read_csv(datapath + 'fruitfly_cds_trainp_framed_3mer_2.csv', header=None, delimiter=',')
            Tc_neg2 = pd.read_csv(datapath + 'fruitfly_cds_trainn_framed_3mer_2.csv', header=None, delimiter=',')
            Tc_pos3 = pd.read_csv(datapath + 'fruitfly_cds_trainp_framed_3mer_3.csv', header=None, delimiter=',')
            Tc_neg3 = pd.read_csv(datapath + 'fruitfly_cds_trainn_framed_3mer_3.csv', header=None, delimiter=',')
            fea1_1 = np.array(ratio_ORFlength_fcds(dna_seq))
            dna_fea = np.array(extract_DNAfeatures(dna_seq, c_m, nc_m, Tc_pos1, Tc_neg1, Tc_pos2, Tc_neg2, Tc_pos3, Tc_neg3, fea1_1))
        elif d_type == 'non-CDS':
            c_m = pd.read_csv(datapath + 'fruitfly_noncds_trainp_6mermean.csv', header=None, delimiter=',')
            nc_m = pd.read_csv(datapath + 'fruitfly_noncds_trainn_6mermean.csv', header=None, delimiter=',')
            Tc_pos1 = pd.read_csv(datapath + 'fruitfly_noncds_trainp_framed_3mer_1.csv', header=None, delimiter=',')
            Tc_neg1 = pd.read_csv(datapath + 'fruitfly_noncds_trainn_framed_3mer_1.csv', header=None, delimiter=',')
            Tc_pos2 = pd.read_csv(datapath + 'fruitfly_noncds_trainp_framed_3mer_2.csv', header=None, delimiter=',')
            Tc_neg2 = pd.read_csv(datapath + 'fruitfly_noncds_trainn_framed_3mer_2.csv', header=None, delimiter=',')
            Tc_pos3 = pd.read_csv(datapath + 'fruitfly_noncds_trainp_framed_3mer_3.csv', header=None, delimiter=',')
            Tc_neg3 = pd.read_csv(datapath + 'fruitfly_noncds_trainn_framed_3mer_3.csv', header=None, delimiter=',')
            fea1_1 = np.array(ratio_ORFlength_fnoncds(dna_seq))
            dna_fea = np.array(extract_DNAfeatures(dna_seq, c_m, nc_m, Tc_pos1, Tc_neg1, Tc_pos2, Tc_neg2, Tc_pos3, Tc_neg3, fea1_1))
        else:
            print("Type error")
    else:
        print("Species error")

    protein_fea = np.array(extract_Proteinfeatures(protein_seq))
    fea = np.concatenate((dna_fea, protein_fea), axis=1)
    filename='feature_name.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        header_row = next(reader)
    features = pd.DataFrame(fea[1:, 1:], columns=header_row)
    return features

##test model
def test_model(datapath,outpath,datafile,s_type,d_type):

    test_sequence = readFasta(datapath + datafile)
    sORF_seq = get_sORF(test_sequence)
    protein_seq = get_protein(sORF_seq)
    fea = feature_encode(datapath, sORF_seq, protein_seq, s_type, d_type)

    if s_type=='H.sapiens':
        if d_type=='CDS':
            trainX = pd.read_csv(datapath + 'human_cds_trainX_feature150.csv').values
            trainX_ = trainX.reshape([14464, 10, 15, 1])  
            y1 = np.array([1] * 7232 + [0] * 7232)
            ##LightGBM
            lgbClf = lgb.LGBMClassifier(boosting_type='gbdt', learning_rate=0.01, n_estimators=400,
                                        max_depth=5, num_leaves=57, max_bin=99, min_data_in_leaf=66,
                                        bagging_fraction=0.8, feature_fraction=0.7,
                                        lambda_l1=0.2604, lambda_l2=0.7363)
            lgbClf.fit(trainX, y1)
            ##Efficien-CapsNet
            model = Efficient_CapsNet_sORF150.build_sORF(trainX_.shape[1:], mode='test', verbose=True)
            feature_importance = pd.read_csv(datapath + 'human_cds_featureRank_lightgbm.csv')
            selector = feature_importance.iloc[0:150, 0].tolist()
            fea_sel = fea.loc[:, selector]
            fea_sel.to_csv(outpath + 'human_cds_testX_feature150.csv', index=False)
            testX = pd.read_csv(outpath + 'human_cds_testX_feature150.csv').values
            testX_ = testX.reshape([len(sORF_seq), 10, 15, 1])
            lgb_proba_testlabel = lgbClf.predict_proba(testX)
            lgb_prob = lgb_proba_testlabel[:, 1]
            model_name='human_cds.h5'
        elif d_type=='non-CDS':
            trainX = pd.read_csv(datapath + 'human_noncds_trainX_feature250.csv').values
            trainX_ = trainX.reshape([15500, 10, 25, 1])
            y1 = np.array([1] * 7750 + [0] * 7750)
            ##LightGBM
            lgbClf = lgb.LGBMClassifier(boosting_type='gbdt', learning_rate=0.05, n_estimators=300,
                                        max_depth=5, num_leaves=30, max_bin=138, min_data_in_leaf=121,
                                        bagging_fraction=0.9, feature_fraction=0.6,
                                        lambda_l1=0.6399, lambda_l2=0.4156)
            lgbClf.fit(trainX, y1)
            ##Efficien-CapsNet
            model = Efficient_CapsNet_sORF250.build_sORF(trainX_.shape[1:], mode='test', verbose=True)
            feature_importance = pd.read_csv(datapath + 'human_noncds_featureRank_lightgbm.csv')
            selector = feature_importance.iloc[0:250, 0].tolist()
            fea_sel = fea.loc[:, selector]
            fea_sel.to_csv(outpath + 'human_noncds_testX_feature250.csv', index=False)
            testX = pd.read_csv(outpath + 'human_noncds_testX_feature250.csv').values
            testX_ = testX.reshape([len(sORF_seq), 10, 25, 1])
            lgb_proba_testlabel = lgbClf.predict_proba(testX)
            lgb_prob = lgb_proba_testlabel[:, 1]
            model_name = 'human_noncds.h5'
        else:
            print ("Type error")

    elif s_type=='M.musculus':
        if d_type=='CDS':
            trainX = pd.read_csv(datapath + 'mouse_cds_trainX_feature150.csv').values
            trainX_ = trainX.reshape([5230, 10, 15, 1])
            y1 = np.array([1] * 2615 + [0] * 2615)
            ##LightGBM
            lgbClf = lgb.LGBMClassifier(boosting_type='gbdt', learning_rate=0.1, n_estimators=100,
                                        max_depth=7, num_leaves=41, max_bin=224, min_data_in_leaf=93,
                                        bagging_fraction=0.9, feature_fraction=0.6,
                                        lambda_l1=0.4224, lambda_l2=0.2594)
            lgbClf.fit(trainX, y1)
            ##Efficien-CapsNet
            model = Efficient_CapsNet_sORF150.build_sORF(trainX_.shape[1:], mode='test', verbose=True)
            feature_importance = pd.read_csv(datapath + 'mouse_cds_featureRank_lightgbm.csv')
            selector = feature_importance.iloc[0:150, 0].tolist()
            fea_sel = fea.loc[:, selector]
            fea_sel.to_csv(outpath + 'mouse_cds_testX_feature150.csv', index=False)
            testX = pd.read_csv(outpath + 'mouse_cds_testX_feature150.csv').values
            testX_ = testX.reshape([len(sORF_seq), 10, 15, 1])
            lgb_proba_testlabel = lgbClf.predict_proba(testX)
            lgb_prob = lgb_proba_testlabel[:, 1]
            model_name='mouse_cds.h5'
        elif d_type=='non-CDS':
            trainX = pd.read_csv(datapath + 'mouse_noncds_trainX_feature250.csv').values
            trainX_ = trainX.reshape([6132, 10, 25, 1])
            y1 = np.array([1] * 3066 + [0] * 3066)
            ##LightGBM
            lgbClf = lgb.LGBMClassifier(boosting_type='gbdt', learning_rate=0.05, n_estimators=300,
                                        max_depth=4, num_leaves=15, max_bin=17, min_data_in_leaf=75,
                                        bagging_fraction=0.5, feature_fraction=0.7,
                                        lambda_l1=0.9026, lambda_l2=0.6507)
            lgbClf.fit(trainX, y1)
            ##Efficien-CapsNet
            model = Efficient_CapsNet_sORF250.build_sORF(trainX_.shape[1:], mode='test', verbose=True)
            feature_importance = pd.read_csv(datapath + 'mouse_noncds_featureRank_lightgbm.csv')
            selector = feature_importance.iloc[0:250, 0].tolist()
            fea_sel = fea.loc[:, selector]
            fea_sel.to_csv(outpath + 'mouse_noncds_testX_feature250.csv', index=False)
            testX = pd.read_csv(outpath + 'mouse_noncds_testX_feature250.csv').values
            testX_ = testX.reshape([len(sORF_seq), 10, 25, 1])
            lgb_proba_testlabel = lgbClf.predict_proba(testX)
            lgb_prob = lgb_proba_testlabel[:, 1]
            model_name = 'mouse_noncds.h5'
        else:
            print ("Type error")

    elif s_type=='D.melanogaster':
        if d_type=='CDS':
            trainX = pd.read_csv(datapath + 'fruitfly_cds_trainX_feature150.csv').values
            trainX_ = trainX.reshape([1364, 10, 15, 1])
            y1 = np.array([1] * 682 + [0] * 682)
            ##LightGBM
            lgbClf = lgb.LGBMClassifier(boosting_type='gbdt', learning_rate=0.1, n_estimators=100,
                                        max_depth=5, num_leaves=25, max_bin=48, min_data_in_leaf=94,
                                        bagging_fraction=0.5, feature_fraction=0.8,
                                        lambda_l1=0.3255, lambda_l2=0.5864)
            lgbClf.fit(trainX, y1)
            ##Efficien-CapsNet
            model = Efficient_CapsNet_sORF150.build_sORF(trainX_.shape[1:], mode='test', verbose=True)
            feature_importance = pd.read_csv(datapath + 'fruitfly_cds_featureRank_lightgbm.csv')
            selector = feature_importance.iloc[0:150, 0].tolist()
            fea_sel = fea.loc[:, selector]
            fea_sel.to_csv(outpath + 'fruitfly_cds_testX_feature150.csv', index=False)
            testX = pd.read_csv(outpath + 'fruitfly_cds_testX_feature150.csv').values
            testX_ = testX.reshape([len(sORF_seq), 10, 15, 1])
            lgb_proba_testlabel = lgbClf.predict_proba(testX)
            lgb_prob = lgb_proba_testlabel[:, 1]
            model_name='fruitfly_cds.h5'
        elif d_type=='non-CDS':
            trainX = pd.read_csv(datapath + 'fruitfly_noncds_trainX_feature150.csv').values
            trainX_ = trainX.reshape([13956, 10, 15, 1])
            y1 = np.array([1] * 6978 + [0] * 6978)
            ##LightGBM
            lgbClf = lgb.LGBMClassifier(boosting_type='gbdt', learning_rate=0.05, n_estimators=300,
                                        max_depth=5, num_leaves=30, max_bin=225, min_data_in_leaf=129,
                                        bagging_fraction=0.9, feature_fraction=0.9,
                                        lambda_l1=0.2516, lambda_l2=0.0407)
            lgbClf.fit(trainX, y1)
            ##Efficien-CapsNet
            model = Efficient_CapsNet_sORF150.build_sORF(trainX_.shape[1:], mode='test', verbose=True)
            feature_importance = pd.read_csv(datapath + 'fruitfly_noncds_featureRank_lightgbm.csv')
            selector = feature_importance.iloc[0:150, 0].tolist()
            fea_sel = fea.loc[:, selector]
            fea_sel.to_csv(outpath + 'fruitfly_noncds_testX_feature150.csv', index=False)
            testX = pd.read_csv(outpath + 'fruitfly_noncds_testX_feature150.csv').values
            testX_ = testX.reshape([len(sORF_seq), 10, 15, 1])
            lgb_proba_testlabel = lgbClf.predict_proba(testX)
            lgb_prob = lgb_proba_testlabel[:, 1]
            model_name = 'fruitfly_noncds.h5'
        else:
            print ("Type error")
    else:
        print("Species error")



    #load weight
    model.load_weights(datapath + model_name)
    print('-' * 30 + 'Begin: test' + '-' * 30)
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

    with open(outpath + 'predict_results.csv', 'w', newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(zip(sORF_seq, coding_potential, prob))

    return


parse = OptionParser()
parse.add_option('-d','--dir',dest = 'inputpath',action = 'store',metavar = 'input file path',help = 'Please enter the input file path')
parse.add_option('-f','--input',dest = 'inputfile',action = 'store',metavar = 'input file name',help = 'Please enter the input file name (FASTA format)')
parse.add_option('-o','--output',dest = 'outputpath',action = 'store',metavar = 'output file path',help = 'Please enter output file name')
parse.add_option('-s','--species',dest = 'species', action = 'store', metavar = 'species name', help = 'Please enter the species name to choose the model, three options: H.sapiens, M.musculus, and D.melanogaster')
parse.add_option('-t','--type',dest = 'regiontype', action = 'store', metavar = 'region type', help = 'Please enter the region type to choose the model, two options: CDS and non-CDS')

(options,args) = parse.parse_args()

for file in ([options.inputpath,options.inputfile,options.outputpath,options.species,options.regiontype]):
	if not (file):
		print(sys.stderr,"\nError: Lack of input file!\n")
		parse.print_help()
		sys.exit(0)

test_model(options.inputpath,options.outputpath,options.inputfile,options.species,options.regiontype)
