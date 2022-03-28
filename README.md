# csORF-finder is an effective ensemble learning approach for accurate identification of multi-species coding short open reading frames
csORF-finder is a series of species-specific ensemble models, integrating Efficient-CapsNet and LightGBM for identifying csORFs in CDS and nonCDS regions of H. sapiens, M. musculus, and D. melanogaster, respectively.

## EXPLANATION
This repository contains four folders, code, raw_data, input_files and output_files.

### Code folder:
This folder contains the python codes.  
```
csORF_finder.py
feature_encoding.py
layers.py
Efficient_CapsNet_sORF150.py
Efficient_CapsNet_sORF250.py
test_csORF-finder.py
```
### raw_data folder:
This folder contains all the training and testing datasets in [1].

### input_files folder:
This folder contains model and feature set files of H. sapiens, M. musculus, and D. melanogaster.

### output_files folder:
This folder contains the predicted results of input file.

## USAGE:
Based on python3.8.  
Python modules:  
```
numpy  
pandas
scipy  
csv  
keras
lightgbm
sklearn
Bio
math
```
will be used. keras is backened on tensorflow.  
Download all the files firstly, open test_csORF-finder.py, change code:  
```
test_model('.../input_files/species/type/','.../output_files/',
           'H.sapiens_sORF.txt',species,type)
```
'H.sapiens_sORF.txt' is the input file name, the sequences in this file must be Fasta format.\
species: three options are avalibale: 'H.sapiens','M.musculus' and 'D.melanogaster'.\
type: 'CDS' or 'non-CDS'. 'CDS' means CDS-sORFs datasets, 'non-CDS' means CDS-sORFs datasets.\
The predicted results is located in output_files.

More details can be found from [1]

## REFERANCE
[1] Zhang M, Zhao J, Li C, Song J, Song X. csORF-finder is an effective ensemble learning approach for accurate identification of multi-species coding short open reading frames.

## CONTACT
If you have any inqueries, please contact xfsong@nuaa.edu.cn；Jiangning.Song@monash.edu；mengzhang@nuaa.edu.cn.


