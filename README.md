# A tool for identifying coding short ORF (csORF)
csORF-finder is an ensemble learning approach developed for identifying coding short open reading frames (csORFs) in CDS and nonCDS regions of H. sapiens, M. musculus, and D. melanogaster, respectively.

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
```
### raw_data folder:
This folder contains all the training and testing datasets in [1].

### input_files folder:
This folder contains model and feature set files of H. sapiens, M. musculus, and D. melanogaster.

### output_files folder:
This folder contains the predicted results of input file.

## Prerequisites
### Software Prerequisites:
csORF_finder is implemented in Python3.8.\
Python modules used in csORF_finder:  
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
Keras is backened on tensorflow.  

## USAGE:
	Command:
		python csORF_finder.py -d input_path/ -i input_file.fa -o output_dir/ -s species -t region_type
	Options:
		-d,	--dir
			  input file path
		-i,	--input
			  input file name (FASTA format)  
		-o,	--output
			  output file path
		-s,	--species
			  species name, three options: H.sapiens, M.musculus, and D.melanogaster
		-t,	--type
			  region type, two options: CDS and non-CDS
		-h,	--help
          	  	  show the help information

More details can be found from [1]

## REFERANCE
[1] Zhang M, Zhao J, Li C, Jiang B, Song JN, Song XF. csORF-finder: an effective ensemble learning approach for accurate identification of multi-species coding short open reading frames. (Unpublished)

## CONTACT
If you have any inqueries, please contact mengzhang@nuaa.edu.cn.


