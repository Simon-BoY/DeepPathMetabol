# DeepPathMetabol
DeepPathMetabol (DPM) is a deep learning based method using H&E stain info and optimal transport theory to predict metabolite distributions and reconstruct high-resolution MSI data from small-sample inputs for cost-efficient, high-precision spatial metabolomics research.
Developer is Bo Yao from School of Life Sciences, Xiamen University, Xiamen, Fujian 361102, China.

# Overview of DeepPathMetabol

<div align=center>
<img src="https://github.com/Simon-BoY/DeepPathMetabol/blob/master/img/DPM.png" width="800" height="550" /><br/>
</div>

_Overview of DeepPathMetabol architecture_. (a) Tissue sections are subjected to MSI and H&E staining experiments. (b) The DPM alignment model registers the H&E images with the MSI images. (c) The DPM transport model uses a CNN backbone to project the information of H&E images into a latent space, along with their corresponding spatial location information. Then utilizes optimal transport module to find the best transport solution between the two and use this solution to predict the MSI of the target tissue section.
