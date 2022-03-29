import csORF_finder
from csORF_finder import test_model

from optparse import OptionParser

parser = OptionParser()
parse.add_option('-d','--dir',dest = 'inputpath',action = 'store',metavar = 'input path',help = 'Please enter the input file path')
parse.add_option('-f','--input',dest = 'inputfile',action = 'store',metavar = 'file name',help = 'Please enter the input file name')
parse.add_option('-o','--output',dest = 'outputpath',action = 'store',metavar = 'output path',help = 'Please enter output file name')
parse.add_option('-s','--species',dest = 'species', action = 'store', metavar = 'species name', help = 'Please enter the species name to choose the model, three options: H.sapiens, M.musculus, and D.melanogaster')
parse.add_option('-t','--type',dest = 'regiontype', action = 'store', metavar = 'region type', help = 'Please enter the region type to choose the model, two options: CDS and non-CDS')
(options,args) = parse.parse_args()

test_model(options.inputpath,options.outputpath,options.inputfile,options.species,options.regiontype)

