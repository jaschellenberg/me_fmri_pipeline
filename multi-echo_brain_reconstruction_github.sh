#!/bin/bash
eval "$(conda shell.bash hook)"
conda init bash

# how to run:
# ./multi_echo_brain_reconstruction_kcl.sh [complete path to folder with files] [scan_number] [number of echos] [dyn to exclude]
# ex - ./multi_echo_brain_reconstruction.sh /home/user/case_number_001 59 3 1,2,3
# this example is for a 3 echo sequence where the first 3 dynamics are motion corrupted  
# if no dynamics are bad, put '0'
# Format of the input files:
# *e1*.nii.gz*, *e2*.nii.gz, *ex*.nii.gz, where x = number of echos in the 
# multi-echo sequence to be fitted. Each echo file is 4D, containing all 
# of the dynamics. For example, if e1.nii.gz is 256 x 256 x 80 x 20, there 
# are 20 dynamics.
# Input File Structure:
# folder structure assumed in the path to files:
# Folder with multi-echo files in nifti format: ME/n[scan_number]/*nii.gz
# Folder with multi-echo dicom files: ME/d[scan_number]/*.dcm

# loading mirtk module
# CHANGE HERE TO YOUR MIRTK LOCATION
module use /dss/dsshome1/02/ge84yah2/reconstruction-software/modules

# setting up directories
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
echo "Running script: ${SCRIPT_DIR}/multi-echo_brain_reconstruction_github.sh"
cd $SCRIPT_DIR

monai_check_path_roi=${SCRIPT_DIR}/monai-checkpoints-unet-t2s-brain-body-placenta-loc-3-lab
check_path_roi_reo_4lab=${SCRIPT_DIR}/monai-checkpoints-unet-notmasked-body-reo-4-lab

# folder with input files to be processed
org_files=$1
nr_me=$2
nr_echos=$3
dyn_to_exclude=$4

echo ""
echo "Processing Scan: ${org_files}"
echo "Sequence number to process: ${nr_me}"
echo "Number of echos: ${nr_echos}"
echo "dynamics to exclude: ${dyn_to_exclude}"
echo ""

# setting up stuff
cd $org_files/ME/n$nr_me
rm -r files_for_brain_recon
rm -r processing_brain
cd reconstructions
rm -r tedana
rm recon_struct_brain_e*
rm t2map*_from_recon_brain.nii.gz
cd
cd $SCRIPT_DIR

monai_check_path_brain_roi=${SCRIPT_DIR}/monai-checkpoints-unet-svr-brain-reo-5-lab

export nnUNet_preprocessed="/dss/dsshome1/02/ge84yah2/thesis/t2s_pipelines/fetal_nnunet/nnUNet_preprocessed"
export nnUNet_raw="/dss/dsshome1/02/ge84yah2/thesis/t2s_pipelines/fetal_nnunet/nnUNet_raw"
export nnUNet_results="/dss/dsshome1/02/ge84yah2/thesis/t2s_pipelines/fetal_nnunet/nnUNet_results"



echo
echo "-----------------------------------------------------------------------------"
echo "-----------------------------------------------------------------------------"
echo
echo "Concatenating and denoising input files ..."
echo

conda activate t2s_venv
python remove_corrupted_brain_dynamics.py $org_files $nr_me $nr_echos $dyn_to_exclude

cd $org_files/ME/n$nr_me

if [[ ! -d processing_brain ]];then
	echo "ERROR: NO INPUT FILES FOUND !!!!" 
	exit
fi

if [[ ! -d reconstructions ]];then
	mkdir reconstructions
else
    echo "dir exists"
fi

# try mrdegibbs to remove ringing artefacts - THIS IDEALLY IS DONE ON THE RAW K-SPACE DATA BEFORE RUNNING THIS PIPELINE
echo
echo "-----------------------------------------------------------------------------"
echo "-----------------------------------------------------------------------------"
echo
echo "MRDEGIBBS - removing ringing artefacts after denoising..."
echo

cd processing_brain
mrdegibbs -force e1_denoised.nii.gz e1_denoised.nii.gz
mrdegibbs -force e2_denoised.nii.gz e2_denoised.nii.gz
mrdegibbs -force e3_denoised.nii.gz e3_denoised.nii.gz
cd ../

echo
echo "-----------------------------------------------------------------------------"
echo "-----------------------------------------------------------------------------"
echo
echo "INPUT FILES ..."
echo

cd processing_brain
echo "files to be processed: " ${org_files}

num_packages=1

dims=$(mrinfo ${org_files}/ME/n${nr_me}/processing_brain/e1_denoised.nii.gz -spacing)
thickness=( $dims )
default_thickness=$(printf "%.1f" ${thickness[0]})

default_resolution=1.2

echo
echo "Slice thickness: ${default_thickness}"
echo "Reconstruction Resolution: ${default_resolution}"
echo

roi_recon="SVR"
roi_names="brain"

# stack_names: list of filenames of concat echo files
stack_names=$(ls *e*.nii*)
# all_og_stacks: list of stack_names
IFS=$'\n' read -rd '' -a all_og_stacks <<<"$stack_names"

echo "Echo files: " ${all_og_stacks[*]}

# processing multi-echo concat files, iterates through the echo concat files
module load mirtk
for ((i=0;i<${#all_og_stacks[@]};i++));
do
    echo "-----------------------------------------------------------------------------"
    echo
    echo "Iteration $i - ${all_og_stacks[i]}"
    echo 

    mkdir stack-t2s-e0${i}
    mkdir stack-t2s-e0${i}/org-files-packages
    mkdir stack-t2s-e0${i}/original-files
    
    # sets voxels = nan and voxels >100000000 to 0
	mirtk nan ${all_og_stacks[i]}  100000000
	# set time resolution to 10ms
	mirtk edit-image ${all_og_stacks[i]}  ${all_og_stacks[i]} -dt 10 
	
	# rescales images to be between 0 and 1500
    mirtk convert-image ${all_og_stacks[i]} ${all_og_stacks[i]::-7}_rescaled.nii.gz -rescale 0 1500
	mirtk extract-image-region ${all_og_stacks[i]} stack-t2s-e0${i}/original-files/t2s -split $nr_echos 	
    mirtk extract-image-region ${all_og_stacks[i]::-7}_rescaled.nii.gz stack-t2s-e0${i}/org-files-packages/${i}-t2s-e0${i} -split $nr_echos 	

done
 
echo
echo "-----------------------------------------------------------------------------"
echo "-----------------------------------------------------------------------------"
echo
echo "3D UNET SEGMENTATION ..."
echo

echo 
echo "-----------------------------------------------------------------------------"
echo "GLOBAL brain segmentation ..."
echo 


conda deactivate
conda activate venv_nnunetv2

mkdir stack-t2s-e01/brain-segmentation-results
mkdir stack-t2s-e01/brain-masks-final

res=128

# rename files for nnUNet brain masking format
for ((k=0; k<$nr_echos; k++)); do

for rename_file in $(ls stack-t2s-e0${k}/org-files-packages/); do 
mv stack-t2s-e0${k}/org-files-packages/$rename_file stack-t2s-e0${k}/org-files-packages/${rename_file::-7}_0000.nii.gz
done

mkdir stack-t2s-e0${k}/recon-stacks-brain
done

cd stack-t2s-e01/


nnUNetv2_predict -d Dataset602_brainsv2 -i org-files-packages/ -o brain-segmentation-results -f  0 1 2 3 4 -tr nnUNetTrainer -c 3d_fullres -p nnUNetPlans -device 'cpu'
nnUNetv2_apply_postprocessing -i brain-segmentation-results -o brain-masks-final -pp_pkl_file /dss/dsshome1/02/ge84yah2/thesis/t2s_pipelines/fetal_nnunet/nnUNet_results/Dataset602_brainsv2/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/postprocessing.pkl -np 8 -plans_json /dss/dsshome1/02/ge84yah2/thesis/t2s_pipelines/fetal_nnunet/nnUNet_results/Dataset602_brainsv2/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4/plans.json

cd ../  

conda deactivate
conda activate t2s_venv

i=1
for dilate_file in $(ls stack-t2s-e0${i}/brain-masks-final/); do 
# dilate the extracted label
# input: global-roi-masks/mask-brain-${jj}-0.nii.gz output: global-roi-masks/mask-brain-${jj}-0.nii.gz
mirtk dilate-image stack-t2s-e0${i}/brain-masks-final/$dilate_file stack-t2s-e0${i}/brain-masks-final/$dilate_file -iterations 2

# crop images for reconstruction
for ((k=0; k<$nr_echos; k++)); do
mirtk mask-image stack-t2s-e0${k}/org-files-packages/${k}${dilate_file:1:7}${k}${dilate_file:9:-7}_0000.nii.gz stack-t2s-e0${i}/brain-masks-final/$dilate_file stack-t2s-e0${k}/recon-stacks-brain/${k}${dilate_file:1:7}${k}${dilate_file:9:-7}_cropped.nii.gz
done
done


echo
echo
echo "-----------------------------------------------------------------------------"
echo "-----------------------------------------------------------------------------"
echo
echo "RUNNING RECONSTRUCTION ..."
echo

echo "ROI : " ${roi_names} " ... "
echo


cd stack-t2s-e01/ 

#calculate the median average template
nStacks=$(ls recon-stacks-brain/*.nii* | wc -l)

mirtk average-images selected_template.nii.gz recon-stacks-brain/*.nii*
mirtk resample-image selected_template.nii.gz selected_template.nii.gz -size 1 1 1
mirtk average-images selected_template.nii.gz recon-stacks-brain/*.nii* -target selected_template.nii.gz

mirtk average-images average_mask_cnn.nii.gz brain-masks-final/*.nii* -target selected_template.nii.gz
mirtk convert-image average_mask_cnn.nii.gz average_mask_cnn.nii.gz -short
mirtk dilate-image average_mask_cnn.nii.gz average_mask_cnn.nii.gz -iterations 2
    	
mirtk mask-image selected_template.nii.gz average_mask_cnn.nii.gz masked-selected_template.nii.gz


cd ../
echo 
echo "-----------------------------------------------------------------------------"
echo "RUNNING SVR" 
echo "-----------------------------------------------------------------------------"
echo

number_of_stacks=$(ls stack-t2s-e01/recon-stacks-brain/*.nii* | wc -l)
mkdir out-proc

nr_channels=$(($nr_echos-1))
 
 
echo "number of additional channels for reconstruction ${nr_channels}" 
cd out-proc


# loop over each dynamic & reconstruct
output_stack=()

for ((i=0; i<number_of_stacks; i++)); do
    if [ $i -lt 10 ]; then
	suffix="${i}"
    else
	suffix="${i}"
    fi

    cd $org_files/ME/n$nr_me/processing_brain/out-proc
    
    # hardcoded ${channel_text} for three echoes
    #channel_text=" ../stack-t2s-e00/recon-stacks-brain/0-t2s-e00_${suffix}_cropped.nii.gz ../stack-t2s-e02/recon-stacks-brain/2-t2s-e02_${suffix}_cropped.nii.gz "
    channel_text=''
    for nr_channel in $(seq 0 $nr_channels); do
	if [ $nr_channel != 2 ] ;
	then
	    channel_text="${channel_text} ../stack-t2s-e0${nr_channel}/recon-stacks-brain/*.nii.gz "
	fi
    done

    # call reconstruction
    echo mirtk reconstruct ${roi_recon}-output-${i}.nii.gz ${number_of_stacks} ../stack-t2s-e02/recon-stacks-brain/*.nii.gz --mc_n $nr_channels --mc_stacks ${channel_text} -mask ../stack-t2s-e01/average_mask_cnn.nii.gz -default_thickness ${default_thickness} -iterations 2 -no_robust_statistics -resolution ${default_resolution} -delta 150 -lambda 0.02 -structural -lastIter 0.015 -no_intensity_matching
    mirtk reconstruct ${roi_recon}-output-${i}.nii.gz ${number_of_stacks} ../stack-t2s-e02/recon-stacks-brain/*.nii.gz --mc_n $nr_channels --mc_stacks ${channel_text} -mask ../stack-t2s-e01/average_mask_cnn.nii.gz -default_thickness ${default_thickness} -iterations 2 -no_robust_statistics -resolution ${default_resolution} -delta 150 -lambda 0.02 -structural -lastIter 0.015 -no_intensity_matching
    
	# store reconstructed stacks in separate file for each echo
	output_stack+=(${roi_recon}-output-${i}.nii.gz)

	for nr_channel in $(seq 0 $nr_channels); do
	    if [ $nr_channel != 2 ]; then
		mv mc-output-${nr_channel}.nii.gz rec_struct_brain_e0${nr_channel}_dyn${i}.nii.gz
		output_stack+=(rec_struct_brain_e0${nr_channel}_dyn${i}.nii.gz)
	    else
		mv SVR-output-${i}.nii.gz rec_struct_brain_e0${nr_channel}_dyn${i}.nii.gz
	    fi
	module unload mirtk

	# convert files
	mrconvert rec_struct_brain_e0${nr_channel}_dyn${i}.nii.gz -axes 0,1,2,-1 -vox ,,,1 recon_struct_brain_e0${nr_channel}_dyn${i}.nii.gz
	done
	
	# recon T2* fitting
	python ${SCRIPT_DIR}/t2s_fitting_brain.py ${org_files} ${nr_me} ${nr_echos} ${i}
	mrconvert t2map_${i}_from_recon_brain.nii.gz -axes 0,1,2,-1 -vox ,,,1 t2map_dyn${i}_from_recon_brain.nii.gz
	

echo "-----------"

echo "-----------"

monai_lab_num=5
brain_recon=${org_files}/*_t2_recon_2.nii.gz
mkdir t2recon-labelmaps/
mkdir met2srecon-labelmaps/

module load mirtk
mirtk prepare-for-monai res-t2recon stack-t2recon t2recon-info.json t2recon-info.csv ${res} 1 ${brain_recon}
module unload mirtk
python ${SCRIPT_DIR}/run_monai_unet_segmentation-2022.py $(pwd)/ ${monai_check_path_brain_roi}/ t2recon-info.json t2recon-labelmaps/ ${res} ${monai_lab_num}

module load mirtk
mirtk prepare-for-monai res-met2srecon stack-met2srecon met2srecon-info.json met2srecon-info.csv ${res} 1 recon_struct_brain_e02_dyn${i}.nii.gz
module unload mirtk
python ${SCRIPT_DIR}/run_monai_unet_segmentation-2022.py $(pwd)/ ${monai_check_path_brain_roi}/ met2srecon-info.json met2srecon-labelmaps/ ${res} ${monai_lab_num}


q1=1; q2=2; q3=3; q4=4; q5=5

new_roi=(1 2 3 4 5)
mkdir t2recon_roi
mkdir met2srecon_roi

# extracts each of the 5 labels
module load mirtk
for ((j=0;j<${#new_roi[@]};j++));
do

    q=${new_roi[$j]}
    #extract each label, store in local roi folder
    # input: monai-segmentation-results-local/cnn-*.nii*; output: local-roi-masks/mask-brain-${jj}-${q}.nii.gz
	mirtk extract-label t2recon-labelmaps/*gz t2recon_roi/mask-brain-${q}.nii.gz ${q} ${q}
	mirtk extract-label met2srecon-labelmaps/*gz met2srecon_roi/mask-brain-${q}.nii.gz ${q} ${q}
	# input: local-roi-masks/mask-brain-${jj}-${q}.nii.gz; output: local-roi-masks/mask-brain-${jj}-${q}.nii.gz
	mirtk extract-connected-components t2recon_roi/mask-brain-${q}.nii.gz t2recon_roi/mask-brain-${q}.nii.gz
    mirtk extract-connected-components met2srecon_roi/mask-brain-${q}.nii.gz met2srecon_roi/mask-brain-${q}.nii.gz
    
done


echo
echo "-----------------------------------------------------------------------------"
echo "-----------------------------------------------------------------------------"
echo
echo "LANDMARK-BASED REGISTRATION ..."
echo

mkdir reo-dofs
# creates an affine dof matrix 
mirtk init-dof init.dof  

z1=1; z2=2; z3=3; z4=4; z5=5
	
total_n_landmarks=5
selected_n_landmarks=5

#mirtk register-landmarks ${template_path}/in-atlas-space-dsvr.nii.gz stack-t2s-e0${i}/stack-files/stack-${jj}.nii.gz stack-t2s-e0${i}/init.dof stack-t2s-e0${i}/reo-dofs/dof-to-atl-${jj}.dof ${total_n_landmarks} ${selected_n_landmarks} ${template_path}/final-mask-${z1}.nii.gz ${template_path}/final-mask-${z2}.nii.gz ${template_path}/final-mask-${z3}.nii.gz ${template_path}/final-mask-${z4}.nii.gz  stack-t2s-e0${i}/organ-roi-masks/mask-${jj}-${z1}.nii.gz stack-t2s-e0${i}/organ-roi-masks/mask-${jj}-${z2}.nii.gz stack-t2s-e0${i}/organ-roi-masks/mask-${jj}-${z3}.nii.gz stack-t2s-e0${i}/organ-roi-masks/mask-${jj}-${z4}.nii.gz 
# Function for rigid landmark-based point registration of two images (the 
# minimum number of landmarks is 4).
# The landmark corrdinates are computed as the centre of the input binary masks
# Usage: mirtk register-landmarks [target_image] [source_image] [init_dof] 
# [output_dof_name] [number_of_landmarks_to_be_used_for_registration] 
# [number_of_input_landmarks_n] [target_landmark_image_1] ... [target_landmark_image_2] 
# [source_landmark_image_1] ... [source_landmark_image_n]
# register generated local masks to template masks
    
echo "registering me-t2s recon to t2 recon"

mirtk register-landmarks ${brain_recon} met2srecon_roi/mask-brain-${z1}.nii.gz init.dof reo-dofs/dof-to-atl.dof ${total_n_landmarks} ${selected_n_landmarks} t2recon_roi/mask-brain-${z1}.nii.gz t2recon_roi/mask-brain-${z2}.nii.gz t2recon_roi/mask-brain-${z3}.nii.gz t2recon_roi/mask-brain-${z4}.nii.gz t2recon_roi/mask-brain-${z5}.nii.gz met2srecon_roi/mask-brain-${z1}.nii.gz met2srecon_roi/mask-brain-${z2}.nii.gz met2srecon_roi/mask-brain-${z3}.nii.gz met2srecon_roi/mask-brain-${z4}.nii.gz met2srecon_roi/mask-brain-${z5}.nii.gz 
# take dof file and apply it to the header of the me-t2s recon

for nr_channel in $(seq 0 $nr_channels); do
mirtk edit-image recon_struct_brain_e0${nr_channel}_dyn${i}.nii.gz ../../reconstructions/recon_struct_brain_e0${nr_channel}_dyn${i}.nii.gz -dofin_i reo-dofs/dof-to-atl.dof
mirtk transform-image ../../reconstructions/recon_struct_brain_e0${nr_channel}_dyn${i}.nii.gz ../../reconstructions/recon_struct_brain_e0${nr_channel}_dyn${i}.nii.gz -target ${brain_recon}

done

mirtk edit-image t2map_dyn${i}_from_recon_brain.nii.gz ../../reconstructions/t2map_dyn${i}_from_recon_brain.nii.gz -dofin_i reo-dofs/dof-to-atl.dof
mirtk transform-image ../../reconstructions/t2map_dyn${i}_from_recon_brain.nii.gz ../../reconstructions/t2map_dyn${i}_from_recon_brain.nii.gz -target ${brain_recon}

cd ../../
for nr_channel in $(seq 0 $nr_channels); do
mirtk edit-image reconstructions/recon_struct_brain_e0${nr_channel}_dyn${i}.nii.gz reconstructions/recon_struct_brain_e0${nr_channel}_dyn${i}.nii.gz -origin 0 0 0 

done

mirtk edit-image reconstructions/t2map_dyn${i}_from_recon_brain.nii.gz reconstructions/t2map_dyn${i}_from_recon_brain.nii.gz -origin 0 0 0    
module unload mirtk

# convert files
cd
cd $org_files/ME/n$nr_me/reconstructions

for nr_channel in $(seq 0 $nr_channels); do
mrconvert -force recon_struct_brain_e0${nr_channel}_dyn${i}.nii.gz -axes 0,1,2,-1 -vox ,,,1 recon_struct_brain_e0${nr_channel}_dyn${i}.nii.gz
done
mrconvert -force t2map_dyn${i}_from_recon_brain.nii.gz -axes 0,1,2,-1 -vox ,,,1 t2map_dyn${i}_from_recon_brain.nii.gz

# register images to first/previous dynamic
module load mirtk
if [ $i != 0 ]; then
    #cd reconstructions
    for nr_channel in $(seq 0 $nr_channels); do
        mirtk register recon_struct_brain_e0${nr_channel}_dyn0.nii.gz recon_struct_brain_e0${nr_channel}_dyn${i}.nii.gz -model Rigid -output recon_struct_brain_e0${nr_channel}_dyn${i}.nii.gz
    done
    mirtk register t2map_dyn0_from_recon_brain.nii.gz t2map_dyn${i}_from_recon_brain.nii.gz -model Rigid -output t2map_dyn${i}_from_recon_brain.nii.gz
    #cd ../
fi
done
module unload mirtk

# merge output files along the fourth dimension (hardcoded for three echoes!)
mrcat -axis 3 recon_struct_brain_e00_dyn*.nii.gz recon_struct_brain_e00.nii.gz
mrinfo recon_struct_brain_e00.nii.gz
mrcat -axis 3 recon_struct_brain_e01_dyn*.nii.gz recon_struct_brain_e01.nii.gz
mrinfo recon_struct_brain_e01.nii.gz
mrcat -axis 3 recon_struct_brain_e02_dyn*.nii.gz recon_struct_brain_e02.nii.gz
mrinfo recon_struct_brain_e02.nii.gz

mrcat -axis 3 t2map_dyn*_from_recon_brain.nii.gz t2map_from_recon_brain.nii.gz
mrinfo t2map_from_recon_brain.nii.gz
#mrcat -axis 3 recon_struct_brain_labels_dyn*.nii.gz recon_struct_brain_labels.nii.gz
#mrinfo recon_struct_brain_labels.nii.gz


echo
echo "-----------------------------------------------------------------------------"
echo "-----------------------------------------------------------------------------"
echo
echo "Brain Segmentation ..."
echo


# segment images
source ${SCRIPT_DIR}/run-segmentation-brain_bounti.sh ${org_files} ${nr_me} 2

cd
cd $org_files/ME/n$nr_me/reconstructions
rm -r *dyn*

echo
echo "-----------------------------------------------------------------------------"
echo "-----------------------------------------------------------------------------"
echo
echo "fMRI Analysis ..."
echo


if [[ ! -d tedana ]];then
	mkdir tedana
else
    echo "dir exists"
fi 

# call tedana
conda activate tedana_env
python ${SCRIPT_DIR}/me_fmri_brain.py ${org_files} ${nr_me} ${nr_echos}

conda deactivate


cd $SCRIPT_DIR

