##Feature encoding based on DNA and protein sequences
import pandas as pd
import csv
import re
import numpy as np
import math
from collections import Counter


##extracting species-specific DNA and protein sequence feature
def extracted_features(dna_seq, protein_seq, model):
    if model == 'H.sapiens-CDS':
        c_m = pd.read_excel('./files/6mermean.xlsx', sheet_name=0, nrows=1, header=None)
        nc_m = pd.read_excel('./files/6mermean.xlsx', sheet_name=0, skiprows=1, header=None)
        Tc = pd.read_csv('./files/human_cds_train_framed_3mer.csv', header=None, delimiter=',').values
        fea1_1 = np.array(ratio_ORFlength_hcds(dna_seq))
        dna_fea = np.array(extract_DNAfeatures(dna_seq, c_m, nc_m, Tc[:, 0:64], Tc[:, 64:128], Tc[:, 128:192], fea1_1))
    elif model == 'H.sapiens-non-CDS':
        c_m = pd.read_excel('./files/6mermean.xlsx', sheet_name=1, nrows=1, header=None)
        nc_m = pd.read_excel('./files/6mermean.xlsx', sheet_name=1, skiprows=1, header=None)
        Tc = pd.read_csv('./files/human_noncds_train_framed_3mer.csv', header=None, delimiter=',').values
        fea1_1 = np.array(ratio_ORFlength_hnoncds(dna_seq))
        dna_fea = np.array(extract_DNAfeatures(dna_seq, c_m, nc_m, Tc[:, 0:64], Tc[:, 64:128], Tc[:, 128:192], fea1_1))
    elif model == 'M.musculus-CDS':
        c_m = pd.read_excel('./files/6mermean.xlsx', sheet_name=2, nrows=1, header=None)
        nc_m = pd.read_excel('./files/6mermean.xlsx', sheet_name=2, skiprows=1, header=None)
        Tc = pd.read_csv('./files/mouse_cds_train_framed_3mer.csv', header=None, delimiter=',').values
        fea1_1 = np.array(ratio_ORFlength_mcds(dna_seq))
        dna_fea = np.array(extract_DNAfeatures(dna_seq, c_m, nc_m, Tc[:, 0:64], Tc[:, 64:128], Tc[:, 128:192], fea1_1))
    elif model == 'M.musculus-non-CDS':
        c_m = pd.read_excel('./files/6mermean.xlsx', sheet_name=3, nrows=1, header=None)
        nc_m = pd.read_excel('./files/6mermean.xlsx', sheet_name=3, skiprows=1, header=None)
        Tc = pd.read_csv('./files/mouse_noncds_train_framed_3mer.csv', header=None, delimiter=',').values
        fea1_1 = np.array(ratio_ORFlength_mnoncds(dna_seq))
        dna_fea = np.array(extract_DNAfeatures(dna_seq, c_m, nc_m, Tc[:, 0:64], Tc[:, 64:128], Tc[:, 128:192], fea1_1))
    elif model == 'D.melanogaster-CDS':
        c_m = pd.read_excel('./files/6mermean.xlsx', sheet_name=4, nrows=1, header=None)
        nc_m = pd.read_excel('./files/6mermean.xlsx', sheet_name=4, skiprows=1, header=None)
        Tc = pd.read_csv('./files/fruitfly_cds_train_framed_3mer.csv', header=None, delimiter=',').values
        fea1_1 = np.array(ratio_ORFlength_fcds(dna_seq))
        dna_fea = np.array(extract_DNAfeatures(dna_seq, c_m, nc_m, Tc[:, 0:64], Tc[:, 64:128], Tc[:, 128:192], fea1_1))
    elif model == 'D.melanogaster-non-CDS':
        c_m = pd.read_excel('./files/6mermean.xlsx', sheet_name=5, nrows=1, header=None)
        nc_m = pd.read_excel('./files/6mermean.xlsx', sheet_name=5, skiprows=1, header=None)
        Tc = pd.read_csv('./files/fruitfly_noncds_train_framed_3mer.csv', header=None, delimiter=',').values
        fea1_1 = np.array(ratio_ORFlength_fnoncds(dna_seq))
        dna_fea = np.array(extract_DNAfeatures(dna_seq, c_m, nc_m, Tc[:, 0:64], Tc[:, 64:128], Tc[:, 128:192], fea1_1))
    else:
        print("Model error")

    protein_fea = np.array(extract_Proteinfeatures(protein_seq))
    fea = np.concatenate((dna_fea, protein_fea), axis=1)
    filename='./files/feature_name.csv'
    with open(filename) as f:
        reader = csv.reader(f)
        header_row = next(reader)
    features = pd.DataFrame(fea[1:, 1:], columns=header_row)
    return features


def extract_DNAfeatures(data, c_m, nc_m, Tc_1, Tc_2, Tc_3, fea1_1):
    sequences = data
    #DNA
    fea1 = np.array(ORF_length(sequences))
    fea2 = np.array(Hexamer_score(sequences, c_m, nc_m, k=6))
    fea3 = np.array(i_framed_3mer_1(sequences)[0])
    fea4 = np.array(i_framed_3mer_2(sequences))
    fea5 = np.array(i_framed_3mer_3(sequences))
    fea6 = np.array(i_framed_CKSNAP_1(sequences))
    fea7 = np.array(i_framed_CKSNAP_2(sequences))
    fea8 = np.array(i_framed_CKSNAP_3(sequences))
    fea9 = np.array(i_framed_TDE_1(sequences, Tc_1))
    fea10 = np.array(i_framed_TDE_2(sequences, Tc_2))
    fea11 = np.array(i_framed_TDE_3(sequences, Tc_3))


    feature_vector_dna = np.concatenate((fea1, fea1_1[:, 1:], fea2[:, 1:], fea3[:, 1:], fea4[:, 1:], fea5[:, 1:], fea6[:, 1:], fea7[:, 1:], fea8[:, 1:], fea9[:, 1:], fea10[:, 1:], fea11[:, 1:]), axis=1)
    return feature_vector_dna


def extract_Proteinfeatures(data):
    sequences = data
    #Protein
    fea1 = np.array(AAC(sequences))
    fea2 = np.array(DPC(sequences))
    fea3 = np.array(CTDC(sequences))
    fea4 = np.array(CTDD(sequences))
    fea5 = np.array(CTDT(sequences))

    feature_vector_protein = np.concatenate((fea1[:, 1:], fea2[:, 1:], fea3[:, 1:], fea4[:, 1:], fea5[:, 1:]), axis=1)
    return feature_vector_protein


###__________________________ORF length_________________________
def ORF_length(fastas):
    ORFlength_encoding = []
    header = ['#'] + ['ORFlength']
    ORFlength_encoding.append(header)
    for i in fastas:
        name, seq = i[0], re.sub('-', '', i[1])
        code = [name]
        l = math.log10(len(seq))
        code.append(l)
        ORFlength_encoding.append(code)
    return ORFlength_encoding


###__________________________hcds ratio ORF length_________________________
def ratio_ORFlength_hcds(fastas):
    diff_ORFlength_encoding = []
    header = ['#'] + ['diff_ORFlength']
    diff_ORFlength_encoding.append(header)
    for i in fastas:
        name, seq = i[0], re.sub('-', '', i[1])
        code = [name]
        l = (len(seq)) / 162.6482   ##note
        code.append(l)
        diff_ORFlength_encoding.append(code)
    return diff_ORFlength_encoding


###__________________________hnoncds ratio ORF length_________________________
def ratio_ORFlength_hnoncds(fastas):
    diff_ORFlength_encoding = []
    header = ['#'] + ['diff_ORFlength']
    diff_ORFlength_encoding.append(header)
    for i in fastas:
        name, seq = i[0], re.sub('-', '', i[1])
        code = [name]
        l = (len(seq)) / 143.5887   ##note
        code.append(l)
        diff_ORFlength_encoding.append(code)
    return diff_ORFlength_encoding


###__________________________mcds ratio ORF length_________________________
def ratio_ORFlength_mcds(fastas):
    diff_ORFlength_encoding = []
    header = ['#'] + ['diff_ORFlength']
    diff_ORFlength_encoding.append(header)
    for i in fastas:
        name, seq = i[0], re.sub('-', '', i[1])
        code = [name]
        l = (len(seq)) / 161.6805   ##note
        code.append(l)
        diff_ORFlength_encoding.append(code)
    return diff_ORFlength_encoding


###__________________________mnoncds ratio ORF length_________________________
def ratio_ORFlength_mnoncds(fastas):
    diff_ORFlength_encoding = []
    header = ['#'] + ['diff_ORFlength']
    diff_ORFlength_encoding.append(header)
    for i in fastas:
        name, seq = i[0], re.sub('-', '', i[1])
        code = [name]
        l = (len(seq)) / 151.4276   ##note
        code.append(l)
        diff_ORFlength_encoding.append(code)
    return diff_ORFlength_encoding


###__________________________fcds ratio ORF length_________________________
def ratio_ORFlength_fcds(fastas):
    diff_ORFlength_encoding = []
    header = ['#'] + ['diff_ORFlength']
    diff_ORFlength_encoding.append(header)
    for i in fastas:
        name, seq = i[0], re.sub('-', '', i[1])
        code = [name]
        l = (len(seq)) / 165.3607   ##note
        code.append(l)
        diff_ORFlength_encoding.append(code)
    return diff_ORFlength_encoding


###__________________________fnoncds ratio ORF length_________________________
def ratio_ORFlength_fnoncds(fastas):
    diff_ORFlength_encoding = []
    header = ['#'] + ['diff_ORFlength']
    diff_ORFlength_encoding.append(header)
    for i in fastas:
        name, seq = i[0], re.sub('-', '', i[1])
        code = [name]
        l = (len(seq)) / 98.1887   ##note
        code.append(l)
        diff_ORFlength_encoding.append(code)
    return diff_ORFlength_encoding



###_____________________Hexamer score_____________________________
def Hexamer_score(fastas, c_m, nc_m, k=6):
    hexamer_score_encoding = []
    header = ['#', 'hexamer_score']
    hexamer_score_encoding.append(header)
    ntarr = 'ACGT'
    hexnuc = [nt1 + nt2 + nt3 + nt4 + nt5 + nt6 for nt1 in ntarr for nt2 in ntarr for nt3 in ntarr for nt4 in ntarr for nt5 in ntarr for nt6 in ntarr]
    for i in fastas:
        name, seq = i[0], re.sub('-', '', i[1])
        code = [name]
        if len(seq) > 5:
            l = len(seq) - k + 1
            log_r = np.zeros((l-6))
            for j in range(3, l-3):
                tempseq = seq[j: j + k]
                idx = hexnuc.index(tempseq)
                Fc = c_m[idx].values
                Fnc = nc_m[idx].values
                if Fc == 0 and Fnc == 0:
                    log_r[j-3] = 0
                elif Fc == 0 and Fnc != 0:
                    log_r[j-3] = -1
                elif Fc != 0 and Fnc == 0:
                    log_r[j-3] = 1
                else:
                    log_r[j-3] = math.log(Fc / Fnc)
            miu = sum(log_r) / (l-6)
            code.append(miu)
        else:
            code.append(0)
        hexamer_score_encoding.append(code)
    return hexamer_score_encoding


###___________________________i-framed-kmer_______________________________________
def i_framed_3mer_1(fastas):
    NA = 'ACGT'
    NApairs = [na1 + na2 + na3 for na1 in NA for na2 in NA for na3 in NA]
    i_framed_3mer_1_encoding = []
    header = ['#']
    for na in NApairs:
        header.append('1_framed_' + na)
    i_framed_3mer_1_encoding.append(header)

    for i in fastas:
        name, seq = i[0], re.sub('-', '', i[1])
        l = len(seq)
        code = [name]
        kmerArr = []
        for j in range(3, l - 3 - 2, 3):
            kmerArr.append(seq[j:j + 3])
        count = Counter()
        count.update(kmerArr)
        for key in count:
            count[key] = count[key] / len(kmerArr)
        for g in range(len(NApairs)):
            if NApairs[g] in count:
                code.append(count[NApairs[g]])
            else:
                code.append(0)
        i_framed_3mer_1_encoding.append(code)
    #delet the stop codon column
    index_to_delet = [49, 51, 57]
    i_framed_3mer_1_fea = []
    for i in i_framed_3mer_1_encoding:
        fea = [i[j] for j in range(len(i)) if j not in index_to_delet]
        i_framed_3mer_1_fea.append(fea)

    return i_framed_3mer_1_fea, i_framed_3mer_1_encoding

def i_framed_3mer_2(fastas):
    NA = 'ACGT'
    NApairs = [na1 + na2 + na3 for na1 in NA for na2 in NA for na3 in NA]
    i_framed_3mer_2_encoding = []
    header = ['#']
    for na in NApairs:
        header.append('2_framed_' + na)
    i_framed_3mer_2_encoding.append(header)

    for i in fastas:
        name, seq = i[0], re.sub('-', '', i[1])
        l = len(seq)
        code = [name]
        kmerArr = []
        for j in range(4, l - 3 - 4, 3):
            kmerArr.append(seq[j:j + 3])
        count = Counter()
        count.update(kmerArr)
        for key in count:
            count[key] = count[key] / len(kmerArr)
        for g in range(len(NApairs)):
            if NApairs[g] in count:
                code.append(count[NApairs[g]])
            else:
                code.append(0)
        i_framed_3mer_2_encoding.append(code)
    return i_framed_3mer_2_encoding


def i_framed_3mer_3(fastas):
    NA = 'ACGT'
    NApairs = [na1 + na2 + na3 for na1 in NA for na2 in NA for na3 in NA]
    i_framed_3mer_3_encoding = []
    header = ['#']
    for na in NApairs:
        header.append('3_framed_' + na)
    i_framed_3mer_3_encoding.append(header)

    for i in fastas:
        name, seq = i[0], re.sub('-', '', i[1])
        l = len(seq)
        code = [name]
        kmerArr = []
        for j in range(5, l - 3 - 3, 3):
            kmerArr.append(seq[j:j + 3])
        count = Counter()
        count.update(kmerArr)
        for key in count:
            count[key] = count[key] / len(kmerArr)
        for g in range(len(NApairs)):
            if NApairs[g] in count:
                code.append(count[NApairs[g]])
            else:
                code.append(0)
        i_framed_3mer_3_encoding.append(code)
    return i_framed_3mer_3_encoding


###_________________________i_framed_CKSNAP_____________________________
def i_framed_CKSNAP_1(fastas, gap=1):
    i_framed_cksnap_1 = []
    AA = 'ACGT'
    AApairs = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = ['#']
    for g in range(gap, gap+1):
        for aa in AApairs:
            header.append('1_framed_' + aa + '.gap' + str(g))
    i_framed_cksnap_1.append(header)
    for i in fastas:
        name, seq = i[0], i[1]
        l = len(seq)
        code = [name]
        for g in range(gap, gap+1):
            myDict = {}
            for pair in AApairs:
                myDict[pair] = 0
            sum = 0
            for index1 in range(3, l-3-g-1, 3):
                index2 = index1 + g + 1
                if index1 < l and index2 < l and seq[index1] in AA and seq[index2] in AA:
                    myDict[seq[index1] + seq[index2]] = myDict[seq[index1] + seq[index2]] + 1
                    sum = sum + 1
            for pair in AApairs:
                code.append(myDict[pair] / sum)
        i_framed_cksnap_1.append(code)
    return i_framed_cksnap_1

def i_framed_CKSNAP_2(fastas, gap=1):
    i_framed_cksnap_2 = []
    AA = 'ACGT'
    AApairs = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = ['#']
    for g in range(gap, gap+1):
        for aa in AApairs:
            header.append('2_framed_' + aa + '.gap' + str(g))
    i_framed_cksnap_2.append(header)
    for i in fastas:
        name, seq = i[0], i[1]
        l = len(seq)
        code = [name]
        for g in range(gap, gap+1):
            myDict = {}
            for pair in AApairs:
                myDict[pair] = 0
            sum = 0
            for index1 in range(4, l-3-g-3, 3):
                index2 = index1 + g + 1
                if index1 < l and index2 < l and seq[index1] in AA and seq[index2] in AA:
                    myDict[seq[index1] + seq[index2]] = myDict[seq[index1] + seq[index2]] + 1
                    sum = sum + 1
            for pair in AApairs:
                code.append(myDict[pair] / sum)
        i_framed_cksnap_2.append(code)
    return i_framed_cksnap_2

def i_framed_CKSNAP_3(fastas, gap=1):
    i_framed_cksnap_3 = []
    AA = 'ACGT'
    AApairs = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = ['#']
    for g in range(gap, gap+1):
        for aa in AApairs:
            header.append('3_framed_' + aa + '.gap' + str(g))
    i_framed_cksnap_3.append(header)
    for i in fastas:
        name, seq = i[0], i[1]
        l = len(seq)
        code = [name]
        for g in range(gap, gap+1):
            myDict = {}
            for pair in AApairs:
                myDict[pair] = 0
            sum = 0
            for index1 in range(5, l-3-g-2, 3):
                index2 = index1 + g + 1
                if index1 < l and index2 < l and seq[index1] in AA and seq[index2] in AA:
                    myDict[seq[index1] + seq[index2]] = myDict[seq[index1] + seq[index2]] + 1
                    sum = sum + 1
            for pair in AApairs:
                code.append(myDict[pair] / sum)
        i_framed_cksnap_3.append(code)
    return i_framed_cksnap_3


# ###____________________________1-framed-TDE_____________________________
def i_framed_TDE_1(fastas, Tc_1):
    # Tc = np.vstack((Tc_pos1, Tc_neg1))
    Tc = Tc_1
    Tm = sum(Tc) / len(Tc)
    Tc_test = np.array(i_framed_3mer_1(fastas)[1])
    Tc_test = Tc_test[:, 1:]

    AA = 'ACGT'
    i_framed_TDE_1_encoding = []
    header = ['#']
    AApairs = [aa1 + aa2 + aa3 for aa1 in AA for aa2 in AA for aa3 in AA]

    for aa in AApairs:
        header.append('1_framed_TDE.' + aa)
    i_framed_TDE_1_encoding.append(header)
    m = 0
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    for i in fastas:
        m = m + 1
        name, seq = i[0], re.sub('-', '', i[1])
        code = [name]
        l = len(seq)
        Tv = np.sqrt(np.multiply(Tm, (1-Tm)) / ((l - 6) / 3))
        tmpCode = [0] * 64
        for j in range(3, l - 3 - 2, 3):
            r = AADict[seq[j]]
            s = AADict[seq[j + 1]]
            t = AADict[seq[j + 2]]
            Tc_test1 = Tc_test[m]
            p = (float(Tc_test1[r * 16 + s * 4 + t]) - float(Tm[r * 16 + s * 4 + t])) / float(Tv[r * 16 + s * 4 + t])
            tmpCode[r * 16 + s * 4 + t] = tmpCode[r * 16 + s * 4 + t] + p
        code = code + tmpCode
        i_framed_TDE_1_encoding.append(code)

    # delet the stop codon column
    index_to_delet = [49, 51, 57]
    i_framed_TDE_1_fea = []
    for i in i_framed_TDE_1_encoding:
        fea = [i[j] for j in range(len(i)) if j not in index_to_delet]
        i_framed_TDE_1_fea.append(fea)

    return i_framed_TDE_1_fea



# ###____________________________2-framed-TDE_____________________________
def i_framed_TDE_2(fastas, Tc_2):
    # Tc = np.vstack((Tc_pos2, Tc_neg2))
    Tc = Tc_2
    Tm = sum(Tc) / len(Tc)
    Tc_test = np.array(i_framed_3mer_2(fastas))
    Tc_test = Tc_test[:, 1:]

    AA = 'ACGT'
    i_framed_TDE_2_encoding = []
    header = ['#']
    AApairs = [aa1 + aa2 + aa3 for aa1 in AA for aa2 in AA for aa3 in AA]
    for aa in AApairs:
        header.append('2_framed_TDE.' + aa)
    i_framed_TDE_2_encoding.append(header)
    m = 0
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    for i in fastas:
        m = m + 1
        name, seq = i[0], re.sub('-', '', i[1])
        code = [name]
        l = len(seq)
        Tv = np.sqrt(np.multiply(Tm, (1-Tm)) / ((l - 9) / 3))
        tmpCode = [0] * 64
        for j in range(4, l - 3 - 4, 3):
            r = AADict[seq[j]]
            s = AADict[seq[j + 1]]
            t = AADict[seq[j + 2]]
            Tc_test1 = list(Tc_test[m])
            p = (float(Tc_test1[r * 16 + s * 4 + t]) - float(Tm[r * 16 + s * 4 + t])) / float(Tv[r * 16 + s * 4 + t])
            tmpCode[r * 16 + s * 4 + t] = tmpCode[r * 16 + s * 4 + t] + p
        code = code + tmpCode
        i_framed_TDE_2_encoding.append(code)
    return i_framed_TDE_2_encoding


# ###____________________________3-framed-TDE_____________________________
def i_framed_TDE_3(fastas, Tc_3):
    # Tc = np.vstack((Tc_pos3, Tc_neg3))
    Tc = Tc_3
    Tm = sum(Tc) / len(Tc)
    Tc_test = np.array(i_framed_3mer_3(fastas))
    Tc_test = Tc_test[:, 1:]

    AA = 'ACGT'
    i_framed_TDE_3_encoding = []
    header = ['#']
    AApairs = [aa1 + aa2 + aa3 for aa1 in AA for aa2 in AA for aa3 in AA]
    for aa in AApairs:
        header.append('3_framed_TDE.' + aa)
    i_framed_TDE_3_encoding.append(header)
    m = 0
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    for i in fastas:
        m = m + 1
        name, seq = i[0], re.sub('-', '', i[1])
        code = [name]
        l = len(seq)
        Tv = np.sqrt(np.multiply(Tm, (1-Tm)) / ((l - 9) / 3))
        tmpCode = [0] * 64
        for j in range(5, l - 3 - 3, 3):
            r = AADict[seq[j]]
            s = AADict[seq[j + 1]]
            t = AADict[seq[j + 2]]
            Tc_test1 = list(Tc_test[m])
            p = (float(Tc_test1[r * 16 + s * 4 + t]) - float(Tm[r * 16 + s * 4 + t])) / float(Tv[r * 16 + s * 4 + t])
            tmpCode[r * 16 + s * 4 + t] = tmpCode[r * 16 + s * 4 + t] + p
        code = code + tmpCode
        i_framed_TDE_3_encoding.append(code)
    return i_framed_TDE_3_encoding


###____________________________AAC_____________________________
def AAC(fastas):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    AAC_encoding = []
    header = ['#']
    for i in AA:
        header.append('AAC_' + i)
    AAC_encoding.append(header)
    for i in fastas:
        name, seq = i[0], re.sub('-', '', i[1])
        count = Counter(seq)
        for key in count:
            count[key] = count[key] / len(seq)
        code = [name]
        for aa in AA:
            code.append(count[aa])
        AAC_encoding.append(code)
    return AAC_encoding


###___________________________DPC_______________________________
def DPC(fastas):
    AA = 'ACDEFGHIKLMNPQRSTVWY'
    DPC_encoding = []
    diPeptides = [aa1 + aa2 for aa1 in AA for aa2 in AA]
    header = ['#']
    for aa in diPeptides:
        header.append(''.join('DPC_' + aa))
    DPC_encoding.append(header)
    AADict = {}
    for i in range(len(AA)):
        AADict[AA[i]] = i
    for i in fastas:
        name, seq = i[0], re.sub('-', '', i[1])
        code = [name]
        tmpCode = [0]*400
        for j in range(len(seq) - 2 + 1):
            tmpCode[AADict[seq[j]] * 20 + AADict[seq[j + 1]]] = tmpCode[AADict[seq[j]] * 20 + AADict[seq[j + 1]]] + 1
        if sum(tmpCode) != 0:
            tmpCode = [i / sum(tmpCode) for i in tmpCode]
        code = code + tmpCode
        DPC_encoding.append(code)
    return DPC_encoding


###___________________CTDC\ CTDD\ CTDT__________________
def Count_CTDC(seq1, seq2):
    sum = 0
    for aa in seq1:
        sum = sum + seq2.count(aa)
    return sum

def Count_CTDD(aaSet, seq):
    number = 0
    for aa in seq:
        if aa in aaSet:
            number = number + 1
    cutoffNums = [1, math.floor(0.25 * number), math.floor(0.50 * number), math.floor(0.75 * number), number]
    cutoffNums = [i if i >=1 else 1 for i in cutoffNums]

    code = []
    for cutoff in cutoffNums:
        myCount = 0
        for i in range(len(seq)):
            if seq[i] in aaSet:
                myCount += 1
                if myCount == cutoff:
                    code.append((i + 1) / len(seq) * 100)
                    break
        if myCount == 0:
            code.append(0)
    return code

def CTDC(fastas):
    group1 = {
		'hydrophobicity_PRAM900101': 'RKEDQN',
		'hydrophobicity_ARGP820101': 'QSTNGDE',
		'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
		'hydrophobicity_PONP930101': 'KPDESNQT',
		'hydrophobicity_CASG920101': 'KDEQPSRNTG',
		'hydrophobicity_ENGD860101': 'RDKENQHYP',
		'hydrophobicity_FASG890101': 'KERSQD',
		'normwaalsvolume': 'GASTPDC',
		'polarity':        'LIFWCMVY',
		'polarizability':  'GASDT',
		'charge':          'KR',
		'secondarystruct': 'EALMQKRH',
		'solventaccess':   'ALFCGIVW'
	}
    group2 = {
		'hydrophobicity_PRAM900101': 'GASTPHY',
		'hydrophobicity_ARGP820101': 'RAHCKMV',
		'hydrophobicity_ZIMJ680101': 'HMCKV',
		'hydrophobicity_PONP930101': 'GRHA',
		'hydrophobicity_CASG920101': 'AHYMLV',
		'hydrophobicity_ENGD860101': 'SGTAW',
		'hydrophobicity_FASG890101': 'NTPG',
		'normwaalsvolume': 'NVEQIL',
		'polarity':        'PATGS',
		'polarizability':  'CPNVEQIL',
		'charge':          'ANCQGHILMFPSTWYV',
		'secondarystruct': 'VIYCWFT',
		'solventaccess':   'RKQEND'
	}
    group3 = {
		'hydrophobicity_PRAM900101': 'CLVIMFW',
		'hydrophobicity_ARGP820101': 'LYPFIW',
		'hydrophobicity_ZIMJ680101': 'LPFYI',
		'hydrophobicity_PONP930101': 'YMFWLCVI',
		'hydrophobicity_CASG920101': 'FIWC',
		'hydrophobicity_ENGD860101': 'CVLIMF',
		'hydrophobicity_FASG890101': 'AYHWVMFLIC',
		'normwaalsvolume': 'MHKFRYW',
		'polarity':        'HQRKNED',
		'polarizability':  'KMHFRYW',
		'charge':          'DE',
		'secondarystruct': 'GNPSD',
		'solventaccess':   'MSPTHY'
	}
    groups = [group1, group2, group3]
    property = (
	'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
	'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
	'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    ######################################
    CTDC_encoding = []
    header = ['#']
    for i in range(1, 39 + 1):
        header.append('CTDC_' + str(i))
    CTDC_encoding.append(header)
    for i in fastas:
        name, seq = i[0], re.sub('-', '', i[1])
        code = [name]
        for p in property:
            c1 = Count_CTDC(group1[p], seq) / len(seq)
            c2 = Count_CTDC(group2[p], seq) / len(seq)
            c3 = 1 - c1 - c2
            code = code + [c1, c2, c3]
        CTDC_encoding.append(code)
    return CTDC_encoding

def CTDD(fastas):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity': 'LIFWCMVY',
        'polarizability': 'GASDT',
        'charge': 'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess': 'ALFCGIVW'
        }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity': 'PATGS',
        'polarizability': 'CPNVEQIL',
        'charge': 'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess': 'RKQEND'
        }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity': 'HQRKNED',
        'polarizability': 'KMHFRYW',
        'charge': 'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess': 'MSPTHY'
        }
    groups = [group1, group2, group3]
    property = (
    'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
    'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
    'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    #######################################
    CTDD_encoding = []
    header1 = ['#']
    for i in range(1, 195 + 1):
        header1.append('CTDD_' + str(i))
    CTDD_encoding.append(header1)
    for i in fastas:
        name, seq = i[0], re.sub('-', '', i[1])
        code = [name]
        for p in property:
            code = code + Count_CTDD(group1[p], seq) + Count_CTDD(group2[p], seq) + Count_CTDD(group3[p], seq)
        CTDD_encoding.append(code)

    index_to_delete = [21, 36, 56, 66, 86, 101, 116, 121, 146, 156, 166, 191]
    for j in range(len(CTDD_encoding)):
        for counter, index in enumerate(index_to_delete):
            index = index - counter
            CTDD_encoding[j].pop(index)

    return CTDD_encoding

def CTDT(fastas):
    group1 = {
        'hydrophobicity_PRAM900101': 'RKEDQN',
        'hydrophobicity_ARGP820101': 'QSTNGDE',
        'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
        'hydrophobicity_PONP930101': 'KPDESNQT',
        'hydrophobicity_CASG920101': 'KDEQPSRNTG',
        'hydrophobicity_ENGD860101': 'RDKENQHYP',
        'hydrophobicity_FASG890101': 'KERSQD',
        'normwaalsvolume': 'GASTPDC',
        'polarity': 'LIFWCMVY',
        'polarizability': 'GASDT',
        'charge': 'KR',
        'secondarystruct': 'EALMQKRH',
        'solventaccess': 'ALFCGIVW'
        }
    group2 = {
        'hydrophobicity_PRAM900101': 'GASTPHY',
        'hydrophobicity_ARGP820101': 'RAHCKMV',
        'hydrophobicity_ZIMJ680101': 'HMCKV',
        'hydrophobicity_PONP930101': 'GRHA',
        'hydrophobicity_CASG920101': 'AHYMLV',
        'hydrophobicity_ENGD860101': 'SGTAW',
        'hydrophobicity_FASG890101': 'NTPG',
        'normwaalsvolume': 'NVEQIL',
        'polarity': 'PATGS',
        'polarizability': 'CPNVEQIL',
        'charge': 'ANCQGHILMFPSTWYV',
        'secondarystruct': 'VIYCWFT',
        'solventaccess': 'RKQEND'
        }
    group3 = {
        'hydrophobicity_PRAM900101': 'CLVIMFW',
        'hydrophobicity_ARGP820101': 'LYPFIW',
        'hydrophobicity_ZIMJ680101': 'LPFYI',
        'hydrophobicity_PONP930101': 'YMFWLCVI',
        'hydrophobicity_CASG920101': 'FIWC',
        'hydrophobicity_ENGD860101': 'CVLIMF',
        'hydrophobicity_FASG890101': 'AYHWVMFLIC',
        'normwaalsvolume': 'MHKFRYW',
        'polarity': 'HQRKNED',
        'polarizability': 'KMHFRYW',
        'charge': 'DE',
        'secondarystruct': 'GNPSD',
        'solventaccess': 'MSPTHY'
        }
    groups = [group1, group2, group3]
    property = (
    'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101', 'hydrophobicity_PONP930101',
    'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101', 'hydrophobicity_FASG890101', 'normwaalsvolume',
    'polarity', 'polarizability', 'charge', 'secondarystruct', 'solventaccess')

    ###########################################
    CTDT_encoding = []
    header2 = ['#']
    for i in range(1, 39 + 1):
        header2.append('CTDT_' + str(i))
    CTDT_encoding.append(header2)

    for i in fastas:
        name, seq = i[0], re.sub('-', '', i[1])
        code = [name]
        aaPair = [seq[j:j + 2] for j in range(len(seq) - 1)]
        for p in property:
            c1221, c1331, c2332 = 0, 0, 0
            for pair in aaPair:
                if (pair[0] in group1[p] and pair[1] in group2[p]) or (pair[0] in group2[p] and pair[1] in group1[p]):
                    c1221 = c1221 + 1
                    continue
                if (pair[0] in group1[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group1[p]):
                    c1331 = c1331 + 1
                    continue
                if (pair[0] in group2[p] and pair[1] in group3[p]) or (pair[0] in group3[p] and pair[1] in group2[p]):
                    c2332 = c2332 + 1
                    continue
            code = code + [c1221 / len(aaPair), c1331 / len(aaPair), c2332 / len(aaPair)]
        CTDT_encoding.append(code)
    return CTDT_encoding


