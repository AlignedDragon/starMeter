# starMeter [![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-FFD21E?style=for-the-badge)](https://huggingface.co/kalandarX/starMeter-model)
Image analysis software to automatically estimate dimensions of synthesized nanostars.


## Task


 Nanoparticles functionalized as surface enhancers for Raman spectroscopy have emerged as a promising technology for imaging, sensing, and catalysis. In particular, gold nanostars have grown appealing for their resonant mode tunability and intense scattered electric fields at the tips of the spikes upon interaction with impinging radiation. To achieve controllable results, it is important to understand the factors affecting the final result of particle synthesis. However doing all the measurements manually is quite time-consuming, which partially hampers the speed of progress in this field. Automatization of this process could greatly enhance synthesis protocols. 

## Target


 Develop a Computer Vision algorithm and a Deep Learning model to segment and quantitatively analyze individual nanostars - measure length and width of branches, dimensions of cores, interparticle distances (clustering),  and adjacent branch angles. 

## Input


 The provided input is TEM (Transmission Electron Microscopy) images (4096 x 4096) in grayscale. The image format is TIF.


## Results

As of June 2026, the segmentation stack has moved from Detectron2 to a transformer-based **Mask2Former** (Swin-Tiny) instance-segmentation model built on HuggingFace `transformers`, with end-to-end training ([src/m2f_train.py](src/m2f_train.py)) and full-frame 4096² inference ([src/m2f_pipeline.py](src/m2f_pipeline.py)). Highlights of the work done since:

- **Overlap-aware training.** A training-only **mask-denoising** branch (`Mask2FormerDN`, [src/m2f_denoise.py](src/m2f_denoise.py)) was ported from the reference paper (Mask2Former + denoising, in the spirit of DN-DETR / MP-Former) to better separate touching nanostars; at inference the model stays plain Mask2Former. See [docs/mask_denoising.md](docs/mask_denoising.md). **SAM 3** was also evaluated as an alternative segmenter.
- **Dataset.** The hand-annotated set grew to **~400 TEM frames** (366 train / 41 val, COCO format). Preprocessing crops out the burned-in scale bar and rotation hints; augmentation uses horizontal flips and every-30° rotations.
- **Reproducible training.** Self-contained **Kaggle notebooks** ([kaggle/](kaggle/)) train both the plain and denoising variants and report per-epoch validation loss + COCO mask AP each epoch.
- **Quantitative analysis.** A first **branch-length** measurement ([utils/branch_length.py](utils/branch_length.py)) skeletonizes each mask, removes the central hub where arms meet, then linearizes the remaining arms to measure their lengths.
- **Human-in-the-loop annotation.** Model predictions are converted back to **VIA** format ([utils/predictions_to_via.py](utils/predictions_to_via.py)) for correction and re-annotation, closing the labeling loop; visualization utilities render segmentations, centroids, and side-by-side comparisons.
- **Released artifacts.** The trained model is published on Hugging Face: [kalandarX/starMeter-model](https://huggingface.co/kalandarX/starMeter-model).

As of December 2024, segmentation of overlapping nanostars has been successfully completed for cases that are not overly complex, even for the human eye. The next steps involve training with more advanced pre-trained models and larger, augmented datasets. The current results were obtained using Detectron2 with a ResNeXt-101 backbone. Future plans include experimenting with YOLOv8 for instance segmentation and considering the use of an additional U-Net model as an expert to refine the segmentation further.

As of April 2024, the segmentation and quantitative analysis of nanostars not overlapping with each other have been finished. However, when the stars’ branches touch each other, the problem becomes harder as it requires separate segmentation of individual nanostars. The plan is to manually segment pictures with VGG Image Annotator and feed them into a Deep Learning model. There are multiple models being considered - Mask R-CNN, SSD+MobileNetV2, SOLO, YOLAct, and PolarMask.


### Example nanostar TEM image:
![image](https://github.com/user-attachments/assets/abfb9a6e-3293-495d-8901-b9cdcb377a72)

### Ground truth:
![Screenshot from 2024-10-28 11-41-04](https://github.com/user-attachments/assets/5e0b07fb-7e07-4cb5-8b8e-ad9bb7d70a69)

### Model prediction:
![r](https://github.com/user-attachments/assets/edbd1924-e27f-43c5-8f4d-e9953090f262)


### Final result obtainable after processing:
![image](https://github.com/user-attachments/assets/eb791378-036b-4e09-a0f3-34ce915e98c7)

### Future work
- improve data augmentation



