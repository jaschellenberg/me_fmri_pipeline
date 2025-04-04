#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 25 17:11:59 2023

@author: kpa19

create concat files for reconstruction
python concat_files_new.py [case num] [sequence num]

output: concat files e1, e2, e3
"""

import os
import sys
import nibabel as nib
from subprocess import call


print('')
print('*************************************************')
print('Prepares multi-echo gradient echo images for DSVR')
print('*************************************************')
print('')


# read in case num and scan num
case_dir = sys.argv[1] 
nr_me = sys.argv[2] 

print('processing scan ' + case_dir + ' and scan number ' + str(nr_me))

#setting up variables
dicom_directory = case_dir + '/ME/d' + nr_me + '/'
nifti_directory = case_dir + '/ME/n' + nr_me + '/'
print('NIFTI directory: ' + nifti_directory)
print('DICOM directory: ' + dicom_directory)

# prepping files for denoising, should be concatenated prior to denoising and then split up 
merged_filename = nifti_directory + 'all_echos_denoised_concat.nii.gz'
merged = nifti_directory + 'all_echos_concat.nii.gz'
print(merged_filename)
text_files = ''
denoised_filenames=[]
files_to_process = []

for file in sorted(os.listdir(nifti_directory + 'files_for_recon/')):
    files_to_process.append(file)
    text_files += ' ' + nifti_directory + 'files_for_recon/'+ file 

# get number of echos and image dimensions from the last nifti in the for loop
img_file = nib.load(nifti_directory + 'files_for_recon/' + file)
nifti_img = img_file.get_fdata()

nO_echos = nifti_img.shape[3]
nO_slices = nifti_img.shape[2]
x_dim = nifti_img.shape[0]
y_dim = nifti_img.shape[1]

print()
print("Number of echos acquired: ", nO_echos)
print("Image size: ", x_dim, y_dim, nO_slices)
print()

print(text_files)
print(files_to_process)


# run denoising using dwidenoise from MRTrix
if not os.path.isfile(merged_filename):
    print('Merging dynamics in ' + nifti_directory + 'files_for_recon')
    call('fslmerge -t ' + merged + ' ' + text_files, shell=True)
    print('denoising images')
    call('dwidenoise ' + merged + ' ' + merged_filename, shell=True)
   
    
else:
    print('Concatenated and denoised file exists')


dyn_count = 0
dyn_num = 0
# create individual denoised images from concatenated file
for dyn in files_to_process:
    img = nib.load(merged_filename)
    n_img = img.get_fdata()
    print("{:03d}".format(dyn_num))
    denoised_img_path = nifti_directory + str(nr_me) + '_dyn_' + str("{:03d}".format(dyn_num)) + '_denoised.nii.gz'
    denoised_filenames.append(denoised_img_path)
    if not os.path.isfile(denoised_img_path):   
        print('file to take header from: ' + nifti_directory + dyn)
        img2 = nib.load(nifti_directory + dyn)
        n_img2 = img2.get_fdata()
        dyn_img = n_img[:,:,:,dyn_count:dyn_count+nO_echos]
        
        img_denoised = nib.Nifti1Image(n_img[:,:,:,dyn_count:dyn_count+nO_echos], img2.affine, img2.header)
        nib.save(img_denoised, denoised_img_path)
        dyn_count+=nO_echos

        print('new count: ' + str(dyn_count))
    else:
        print(denoised_img_path + ' already exists.')
        dyn_count+=nO_echos

    dyn_num+=1
           
exit()

merged_folder = nifti_directory + '/'
# merge 2nd echo of each dynamic into one file
# merge t2maps of each dynamic into one file
if not os.path.isfile(merged_folder + 'e2_' + str(len(denoised_filenames)) + '_concat.nii.gz'):
    # print(denoised_filenames)
    denoised_echos = ''
    denoised_echos_0 = ''
    denoised_echos_2 = ''
    # create merged files for reconstruction
    if not os.path.exists(nifti_directory + 'echos_denoised'):
        os.mkdir(nifti_directory + 'echos_denoised')
    
    # merged_folder = nifti_directory + fm_id + '_denoised_' + str(len(denoised_filenames)) + '/'
        
    for file in denoised_filenames:
        
        # split denoised images into separate echos
        # print('fslsplit ' + file + ' ' + nifti_directory + 'echos_denoised/s0' + nr_me +'_' + file[-19:-16] + '_e')
        call('fslsplit ' + file + ' ' + nifti_directory + 'echos_denoised/s0' + nr_me +'_' + file[-19:-16] + '_e', shell=True)
        denoised_echos +=  ' ' + nifti_directory + 'echos_denoised/s0' + nr_me +'_' + file[-19:-16] + '_e0001.nii.gz'
        denoised_echos_0 +=  ' ' + nifti_directory + 'echos_denoised/s0' + nr_me +'_' + file[-19:-16] + '_e0000.nii.gz'
        denoised_echos_2 +=  ' ' + nifti_directory + 'echos_denoised/s0' + nr_me +'_' + file[-19:-16] + '_e0002.nii.gz'
        #call('rm ' + nifti_directory + 'echos_denoised/s0' + nr_me +'_' + file[-19:-16] + '_e0000.nii.gz',shell=True)
        #call('rm ' + nifti_directory + 'echos_denoised/s0' + nr_me +'_' + file[-19:-16] + '_e0002.nii.gz',shell=True)   
        # exit()
    # print()
    print(denoised_echos)

    call('fslmerge -t ' + merged_folder + fm_id + '_e2_' + str(len(denoised_filenames)) + '_concat.nii.gz' + ' ' + denoised_echos, shell=True)
    call('fslmerge -t ' + merged_folder + fm_id + '_e1_' + str(len(denoised_filenames)) + '_concat.nii.gz' + ' ' + denoised_echos_0, shell=True)
    call('fslmerge -t ' + merged_folder + fm_id + '_e3_' + str(len(denoised_filenames)) + '_concat.nii.gz' + ' ' + denoised_echos_2, shell=True)
    
    # remove intermediary files
    call('rm ' + merged, shell=True) 
    call('rm ' + merged_filename, shell=True)
else:
    print('Merged files already exist for ' + case_dir)



