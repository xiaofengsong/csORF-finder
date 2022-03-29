import sys
import csORF_finder
from csORF_finder import test_model

from optparse import OptionParser

parse = OptionParser()
parse.add_option('-d','--dir',dest = 'inputpath',action = 'store',metavar = 'input file path',help = 'Please enter the input file path')
parse.add_option('-f','--input',dest = 'inputfile',action = 'store',metavar = 'input file name',help = 'Please enter the input file name (FASTA format)')
parse.add_option('-o','--output',dest = 'outputpath',action = 'store',metavar = 'output file path',help = 'Please enter output file name')
parse.add_option('-s','--species',dest = 'species', action = 'store', metavar = 'species name', help = 'Please enter the species name to choose the model, three options: H.sapiens, M.musculus, and D.melanogaster')
parse.add_option('-t','--type',dest = 'regiontype', action = 'store', metavar = 'region type', help = 'Please enter the region type to choose the model, two options: CDS and non-CDS')
parse.add_option('-h','--help',dest = 'help',action = 'store_false',metavar = 'help information',help = 'To show the usage information of csORF-finder')

(options,args) = parse.parse_args()

if (options.help):
	parse.print_help()
	sys.exit(0)

for file in ([options.inputpath,options.inputfile,options.outputpath,options.species,options.regiontype]):
	if not (file):
		print(sys.stderr,"\nError: Lack of input file!\n")
		parse.print_help()
		sys.exit(0)
    
test_model(options.inputpath,options.outputpath,options.inputfile,options.species,options.regiontype)

