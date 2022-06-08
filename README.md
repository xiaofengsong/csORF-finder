# csORF-finder: an effective ensemble learning framework for accurate identification of multi-species coding short open reading frames
csORF-finder is a series of species-specific ensemble models, integrating Efficient-CapsNet and LightGBM for identifying csORFs.\ 
in CDS and nonCDS regions of H. sapiens, M. musculus, and D. melanogaster, respectively.

## EXPLANATION
This repository contains four folders, code, raw_data, input_files and output_files.

### Code folder:
This folder contains the following python codes and files subfolder (including model, training features, etc.).  
```
readFasta.py
getsORFs.py
getProteins.py
feature_encoding.py
layers.py
Efficient_CapsNet_sORF150.py
Efficient_CapsNet_sORF250.py
csORF_finder_predict.py
csORF_finder_predict1.py (for other species, such as R. norvegicus, S. cerevisiae, and A. thaliana)
csORF_finder_predict2.py (for other species, such as D. rerio and C. elegans)
```

### raw_data folder:
This folder contains all the training and testing datasets in csORF-finder.

### input_files folder:
This folder contains input file.

### output_files folder:
This folder contains the predicted results of input file.

## Installation and USAGE:
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
python csORF_finder_predict.py -i input_files -o output_files -s species -r regions -t type
python csORF_finder_predict1.py -i input_files -o output_files -t type
python csORF_finder_predict2.py -i input_files -o output_files -t type
```
-i, input_files: input file name located in input_files folder. Query sequences to be predicted in fasta format.\
-o, input_files: output file name located in output_files folder. Save the prediction results.\
-s, species: 'H.sapiens' or 'M.musculus' or 'D.melanogaster'.Please enter the specific species, currently we accept three options: H.sapiens, M.musculus, and D.melanogaster'.\
-r, regions: 'CDS' or 'non-CDS'. Please enter the region type to choose the model, two options: CDS and non-CDS'.\
-t, type: 'sORF' or 'other'. Query sequences is sORFs or others.

### Example:
```
python csORF_finder_predict.py -i H.sapiens_test.txt -o H.sapiens_results.csv -s H.sapiens -r CDS -t sORF
python csORF_finder_predict.py -i M.musculus_test.txt -o M.musculus_results.csv -s M.musculus -r non-CDS -t sORF
python csORF_finder_predict.py -i D.melanogaster_test.txt -o D.melanogaster_results.csv -s D.melanogaster -r CDS -t sORF
python csORF_finder_predict1.py -i A.thaliana_test.txt -o A.thaliana_results.csv -t sORF
python csORF_finder_predict2.py -i C.elegans_test.txt -o C.elegans_results.csv -t sORF
```

More details can be found from [1]

## REFERANCE
[1] Zhang M, Zhao J, Li C, Song J, Song X. csORF-finder: an effective ensemble learning framework for accurate identification of multi-species coding short open reading frames.

## CONTACT
If you have any inqueries, please contact mengzhang@nuaa.edu.cn.


