##Extract sORF sequence
import re
import numpy as np

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
