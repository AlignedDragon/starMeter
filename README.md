# starMeter
Image analysis software to automatically estimate dimensions of synthesized nanostars.

## Model

The trained model is available on Hugging Face: [kalandarX/starMeter-model](https://huggingface.co/kalandarX/starMeter-model)

## Task


 Nanoparticles functionalized as surface enhancers for Raman spectroscopy have emerged as a promising technology for imaging, sensing, and catalysis. In particular, gold nanostars have grown appealing for their resonant mode tunability and intense scattered electric fields at the tips of the spikes upon interaction with impinging radiation. To achieve controllable results, it is important to understand the factors affecting the final result of particle synthesis. However doing all the measurements manually is quite time-consuming, which partially hampers the speed of progress in this field. Automatization of this process could greatly enhance synthesis protocols. 

## Target


 Develop a Computer Vision algorithm and a Deep Learning model to segment and quantitatively analyze individual nanostars - measure length and width of branches, dimensions of cores, interparticle distances (clustering),  and adjacent branch angles. 

## Input


 The provided input is TEM (Transmission Electron Microscopy) images (4096 x 4096) in grayscale. The image format is TIF.


## Results

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



