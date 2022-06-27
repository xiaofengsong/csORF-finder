# csORF-finder: an effective ensemble learning framework for accurate identification of multi-species coding short open reading frames
A series of species-specific ensemble models by integrating Efficient-CapsNet and LightGBM, collectively termed csORF-finder, was designed 
for identifying csORFs in CDS and nonCDS regions of H. sapiens, M. musculus, and D. melanogaster, respectively.

## EXPLANATION
This repository contains four folders: code, raw_data, input_files, and output_files.

### code folder:
This folder contains the following python codes and files subfolder (including model, training features, etc.).  
```
readFasta.py
getsORFs.py
getProteins.py
feature_encoding.py
layers.py
Efficient_CapsNet_sORF150.py
Efficient_CapsNet_sORF250.py
csORF_finder_predict_sORFs.py (for sORF sequences)
csORF_finder_predict_transcripts.py (for transcript sequences)
```

### raw_data folder:
This folder contains all the training and testing datasets in csORF-finder.

### input_files folder:
This folder contains input file.

### output_files folder:
This folder contains the predicted results of input file.

## INSTALLATION AND USAGE:
Based on python3.8.  
Python modules:  
```
conda install cudatoolkit=10.1
conda install cudnn=7.6
pip install tensorflow-gpu==2.3.0
pip install protobuf==3.20.0
pip install keras==2.2.4
pip install lightgbm
pip install pandas
pip install openpyxl
```
Installing all the modules firstly.
Command:  
```
cd csORF_finder/code
python csORF_finder_predict_sORFs.py -i input_file -o output_file -m model
```
```
-i, input_file: input file name located in input_files folder. Query sequences to be predicted in FASTA format.\
-o, output_file: output file name located in output_files folder. Save the prediction results.\
-m, model: Please choose one model, six options: H.sapiens-CDS, H.sapiens-non-CDS, M.musculus-CDS, M.musculus-non-CDS, D.melanogaster-CDS, D.melanogaster-non-CDS.
```

### Example:
```
python csORF_finder_predict_sORFs.py -i H.sapiens_sORFs_test.txt -o H.sapiens_sORFs_results.csv -m H.sapiens-CDS
python csORF_finder_predict_sORFs.py -i M.musculus_sORFs_test.txt -o M.musculus_sORFs_results.csv -m M.musculus-non-CDS
python csORF_finder_predict_sORFs.py -i D.melanogaster_sORFs_test.txt -o D.melanogaster_sORFs_results.csv -m D.melanogaster-CDS
python csORF_finder_predict_transcripts.py -i H.sapiens_transcripts_test.txt -o H.sapiens_transcripts_results.csv -m H.sapiens-CDS
python csORF_finder_predict_transcripts.py -i M.musculus_transcripts_test.txt -o M.musculus_transcripts_results.csv -m M.musculus-non-CDS
python csORF_finder_predict_sORFs.py -i A.thaliana_sORFs_test.txt -o A.thaliana_sORFs_results.csv -m H.sapiens-non-CDS
python csORF_finder_predict_sORFs.py -i C.elegans_sORFs_test.txt -o C.elegans_sORFs_results.csv -m M.musculus-non-CDS

```

More details can be found from [1]

## REFERANCE
[1] Zhang M, Zhao J, Li C, Ge F, Wu J, Jiang B, Song J, Song X. csORF-finder: an effective ensemble learning framework for accurate identification of multi-species coding short open reading frames.

## CONTACT
If you have any inqueries, please contact mengzhang@nuaa.edu.cn.


