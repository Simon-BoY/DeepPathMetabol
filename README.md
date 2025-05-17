# DeepPathMetabol
DeepPathMetabol (DPM) is a deep learning based method using H&E stain info and optimal transport theory to predict metabolite distributions and reconstruct high-resolution MSI data from small-sample inputs for cost-efficient, high-precision spatial metabolomics research.
Developer is Bo Yao from School of Life Sciences, Xiamen University, Xiamen, Fujian 361102, China.

# Overview of DeepPathMetabol

<div align=center>
<img src="https://github.com/Simon-BoY/DeepPathMetabol/blob/master/img/DPM.png" width="800" height="550" /><br/>
</div>

__Overview of DeepPathMetabol architecture__. (a) Tissue sections are subjected to MSI and H&E staining experiments. (b) The DPM alignment model registers the H&E images with the MSI images. (c) The DPM transport model uses a CNN backbone to project the information of H&E images into a latent space, along with their corresponding spatial location information. Then utilizes optimal transport module to find the best transport solution between the two and use this solution to predict the MSI of the target tissue section.

# Requirement

# Quickly start

## Run DeepPathMetabol model
cd to the DeepPathMetabol fold
If you want to perform Dpm_Alignment for H&E image and MSI image registration, taking mouse kidney section as an example, run：
  python run.py Align --img_path './data/Kidney/img/03.tif' --msi_path './data/Kidney/count/03.txt' --output_prefix './data/processed_data/k03_80um.h5ad' --cut_threshold 210 --n 0
If you want to perform Dpm_transport for prediction of MSI data for adjacent slices, taking mouse kidney section as an example, run:
  python run.py Predict --h5ad_path './data/processed_data/k03_80um.h5ad' --image_path './data/Kidney/img/12.tif' --output_path './data/processed_data/k12_80um_Dpm.h5ad' --spot_size 100
If you want to perform both Dpm_Alignment (H&E-MSI registration) and Dpm_transport (MSI prediction for adjacent slices) sequentially in one go, taking mouse kidney section as an example, run
  python run.py All --img_path './data/Kidney/img/03.tif' --msi_path './data/Kidney/count/03.txt' --output_prefix './data/processed_data/k03_80um.h5ad' --image_path './data/Kidney/img/12.tif' --pred_output_path './data/processed_data/k12_80um_Dpm.h5ad' --cut_threshold 210 --n 0 --spot_size 70

## Contact
Please contact me if you have any help: 
