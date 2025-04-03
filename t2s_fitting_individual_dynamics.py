#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  5 20:50:01 2024

create T2* maps
input: single 4D volume with all dynamics in the time direction
outputted from dwidenoise
three echos assumed

output: T2* maps

@author: kpa19
"""


import os
from os import listdir
from os.path import isfile, join, dirname, exists
import numpy as np
import nibabel as nib
import get_TE
import get_dims
from scipy.optimize import least_squares 
from subprocess import call
# from numba import jit

print('')
print('*****************************')
print('Script for T2* Pre-processing')
print('Prepares T2* images for DSVR')
print('*****************************')
print('')


# T2 fit function
# @jit(nopython=True)
def t2fit(X,data,TEs):
    TEs=np.array(TEs,dtype=float)
    X=np.array(X,dtype=float)
    S = X[0] * ((np.exp(-(TEs/X[1]))))                  
    return data - S


freemax_info = '/home/kpa19/t2s_body/'
freemax_data = '/home/jhu14/Dropbox/placentaJhu/'

fm_ids = []
nr_mes=[]

# done



# running
# fm_ids = ['mibirth000101','mibirth000101','mibirth000201','mibirth000201','mibirth000301','mibirth000301','mibirth000401','mibirth000401','mibirth000402','mibirth000402']
# nr_mes=['4','37','3','37','3','34','3','44','3','39']

# fm_ids = ['mibirth000501','mibirth000501','mibirth000601','mibirth000601','mibirth000701','mibirth000701','mibirth000801','mibirth000901']
# nr_mes=['3','30','3','37','3','38','38','3']

# fm_ids = ['mibirth001001','mibirth001001','mibirth001101','mibirth001101','mibirth001201','mibirth001201','mibirth001301','mibirth001301','mibirth001401','mibirth001401']
# nr_mes=['3','37','3','33','3','40','3','31','3','32']

fm_ids = ['mibirth001402','mibirth001402','mibirth001501','mibirth001601','mibirth001601','mibirth001701','mibirth001701','mibirth001801','mibirth001801']
nr_mes=['3','39','39','3','40','3','37','3','37']

# fm_ids = ['mibirth001901','mibirth001901','mibirth002001','mibirth002001','mibirth002101','mibirth002101','mibirth002201','mibirth002301','mibirth002301','mibirth002302','mibirth002302']
# nr_mes=['3','40','3','42','3','38','151','3','37','3','42']




# set up, need to still run
# fm_ids = ['mibirth002401','mibirth002601','mibirth002601','mibirth002701','mibirth002701','mibirth002801','mibirth002801','mibirth002901','mibirth002901']
# nr_mes=['4','3','41','3','44','3','42','3','43']

# fm_ids = ['mibirth003001','mibirth003001','mibirth003002','mibirth003002','mibirth003101','mibirth003101','mibirth003201','mibirth003201','mibirth003201','mibirth003301','mibirth003301']
# nr_mes=['3','34','3','41','3','46','3','5','51','3','43']

# fm_ids = ['mibirth003401','mibirth003401','mibirth003501','mibirth003501','mibirth003601','mibirth003601','mibirth003701','mibirth003701','mibirth003801','mibirth003801']
# nr_mes=['3','41','3','41','3','51','3','49','3','44']

# fm_ids = ['mibirth00','mibirth00','mibirth00','mibirth00','mibirth00','mibirth00','mibirth00','mibirth00','mibirth00','mibirth00']
# nr_mes=['','','','','','','','','','']

# fm_ids = ['mibirth00','mibirth00','mibirth00','mibirth00','mibirth00','mibirth00','mibirth00','mibirth00','mibirth00','mibirth00']
# nr_mes=['','','','','','','','','','']

# fm_ids = ['mibirth00','mibirth00','mibirth00','mibirth00','mibirth00','mibirth00','mibirth00','mibirth00','mibirth00','mibirth00']
# nr_mes=['','','','','','','','','','']


for x,fm_id in enumerate(fm_ids):
    denoised_filenames=[]
    denoised_t2maps = ''
    print('processing scan: ' + fm_id + '\n')
    nr_me=nr_mes[x]
    case_dir = freemax_data + fm_id
    data_directory = case_dir + '/ME/d' + nr_me + '/'
    # need a nifti in order to save with nibabel
    # or need dicom directory in order to get TE times 
    nifti_directory = case_dir + '/ME/n' + nr_me + '/'
    print(nifti_directory)
    print(data_directory)
    if not os.path.exists(nifti_directory + 't2maps_denoised'):
        os.mkdir(nifti_directory + 't2maps_denoised')
    
    # get TE, image dimensions
    te = get_TE.read(data_directory)
    nO_echos = len(te)
    size = get_dims.read(data_directory)
    nO_slices = size[0]
    
    x_dim = size[1]
    y_dim = size[2]
    print(te)
    # print(size)
    # exit()
    te = (np.array(te,dtype=float))[0:3]
    # exit()
    merged_filename = nifti_directory + fm_id + '_all_echos_denoised_concat.nii.gz'
    merged = nifti_directory + fm_id + '_all_echos_concat.nii.gz'
    print(merged_filename)
    text_files = ''
    files_to_process = []
    for file in sorted(os.listdir(nifti_directory)):
        if 's0' + nr_me in file:
            files_to_process.append(file)
            text_files += ' ' + nifti_directory + file 
        elif 's' + nr_me in file:
            files_to_process.append(file)
            text_files += ' ' + nifti_directory + file 
        elif 's00' + nr_me in file:
            files_to_process.append(file)
            text_files += ' ' + nifti_directory + file 
    print(files_to_process)
    # print(text_files)
    
    
    if not os.path.isfile(merged_filename):
        print('Merging dynamics in ' + nifti_directory)
        # call('fslmerge -t ' + merged + ' ' + text_files, shell=True)
        call('mrcat ' + text_files + ' ' + merged,shell=True)
        print('denoising images')
        call('dwidenoise ' + merged + ' ' + merged_filename, shell=True)
   
        
    else:
        print('Concatenated and denoised file exists')
        
    dyn_count = 0
    # create individual denoised images from concatenated file
    for dyn in files_to_process:
        img = nib.load(merged_filename)
        n_img = img.get_fdata()
        # n_img_header = img.header

        denoised_img_path = nifti_directory + fm_id +"_" + str(nr_me) + '_dyn_' + str(dyn[5:8]) + '_denoised.nii.gz'
        denoised_filenames.append(denoised_img_path)
        if not os.path.isfile(denoised_img_path):   
            print('file to take header from: ' + nifti_directory + dyn)
            img2 = nib.load(nifti_directory + dyn)
            n_img2 = img2.get_fdata()
            dyn_img = n_img[:,:,:,dyn_count:dyn_count+3]
            
            if nO_echos == 4:
                img_denoised = nib.Nifti1Image(n_img[:,:,:,dyn_count:dyn_count+4], img2.affine, img2.header)
                nib.save(img_denoised, denoised_img_path)
                dyn_count+=4
            else:
                img_denoised = nib.Nifti1Image(n_img[:,:,:,dyn_count:dyn_count+3], img2.affine, img2.header)
                nib.save(img_denoised, denoised_img_path)
                dyn_count+=3
            print('new count: ' + str(dyn_count))
        else:
            print(denoised_img_path + ' already exists.')
            if nO_echos == 4:
                dyn_count+=4
            else:
                dyn_count+=3
    # exit()            
    for denoised_file in sorted(os.listdir(nifti_directory)):
        
        if 'denoised.nii' in denoised_file and not 't2map' in denoised_file and not 'concat' in denoised_file:
            
            t2map_path =  nifti_directory + "t2maps_denoised/" + fm_id +"_" + str(nr_me) + '_dyn_' + str(denoised_file[-19:-16]) + '_denoised_t2map.nii.gz'
            if not os.path.isfile(t2map_path):
                print(denoised_file)
                denoised_img = nib.load(nifti_directory + denoised_file)
                d_img = denoised_img.get_fdata()
                d_img = d_img[:,:,:,0:3]
                # n_img_header = img.header
                # print(t2map_path)
                print('creating file: ' + t2map_path)          
                print(x_dim, y_dim, nO_slices)
                T2_map = np.zeros((d_img.shape))
                print(T2_map.shape)
                d_img[d_img<0]=0
                
                for iz in range(0,nO_slices):
                    print(iz,end='-')
                    print(iz)
                    for ix in range(0,x_dim):
                        for iy in range(0,y_dim):
                            
                            pix_array = np.array(d_img[ix,iy,iz, :], dtype=float)
                            param_init = np.squeeze([pix_array[2], np.average(te)])
                            result = least_squares(t2fit, param_init, args = (pix_array,te), bounds=([0,0],[10000,1000]))
                            T2_map[ix,iy,iz, 0]= result.x[0]
                            T2_map[ix,iy,iz, 1]= result.x[1]
                        
                t2_val = T2_map[:,:,:,1]
                fit_result = nib.Nifti1Image(t2_val, denoised_img.affine, denoised_img.header)
                nib.save(fit_result, t2map_path)
                denoised_t2maps +=' ' +t2map_path
                
            else:
                print(t2map_path + ' already exists.')
                denoised_t2maps +=' ' +t2map_path
                
call('rm ' + nifti_directory + merged, shell=True) 
call('rm ' + nifti_directory + merged_filename, shell=True)
    
    