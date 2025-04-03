"""
Created on Tue Sep 17 11:02:34 2024

@author: jschellenberg

perform ME analysis of brain
input: multi-echo reconstructions
python me_fmri_brain.py [case_id] [sequence number] [num echos]
example:
python me_fmri_brain.py fm0046 59
three echos assumed

output: files of ME analysis using tedana toolbox

"""

import sys
import get_TE
import nilearn
import nibabel
import numpy
import sklearn
import scipy
import mapca
import tedana
from tedana import workflows

print('')
print('*****************************')
print('Script for ME fMRI Analysis')
print('*****************************')
print('')


# get inputs
case_dir = sys.argv[1] 
nr_me = sys.argv[2] 
num_echos =sys.argv[3]

nr_me = str(nr_me)


# set up paths
input_directory = case_dir + '/ME/n' + nr_me + '/reconstructions/'
print('Directory to reconstructions & labelled mask: ' + input_directory)

# output
output_path = input_directory + 'tedana/'
# to mask
mask_path =  input_directory + 'recon_struct_brain_labels.nii.gz'

# find TEs
dicom_directory = case_dir + '/ME/d' + nr_me + '/'
te = get_TE.read(dicom_directory)
te = te[0:3]
print('TEs: ' + ', '.join(map(str, te)))

# perform ME fMRI analysis
print('')
print('*****************************')
print('...performing ME fMRI analysis...')
print('*****************************')
print('')

# use brain labels
workflows.tedana_workflow(['recon_struct_brain_e01.nii.gz','recon_struct_brain_e02.nii.gz','recon_struct_brain_e00.nii.gz'],te, out_dir = output_path, mask = 'recon_struct_brain_labels.nii.gz')

# call tedana without mask
#workflows.tedana_workflow(['recon_struct_brain_e01.nii.gz','recon_struct_brain_e02.nii.gz','recon_struct_brain_e00.nii.gz'],te,out_dir = output_path)

print('done')


