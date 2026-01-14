# Traffic sign detection using Faster-RCNN

This repository implements a baseline traffic sign detection pipeline using Faster R-CNN on the German Traffic Sign Detection Benchmark (GTSDB) dataset. [(GTSDB) dataset](https://benchmark.ini.rub.de/gtsdb_news.html). 

In this project, I focus on building a clean, reproducible, and well-established baseline detector using a pre-trained Faster R-CNN with a ResNet50 backbone and Feature Pyramid Network (FPN).
This baseline serves as a reference model for more advanced detection experiments, including a custom neck architecture and a fully custom Faster R-CNN implementation, which are released in separate repositories.

The model performs both bounding box localization and traffic sign class prediction.

The overall workflow and model variants used in the project are illustrated in the accompanying flowchart.
<p align="center">
<img width="500" alt="Detection_flowchart" src="https://github.com/user-attachments/assets/5ad30c66-b45b-4cda-b254-67424d1e85f1" />
</p>

## Repository Structure

* **ground_truth.ipynb:** Implements the Faster R-CNN detector with using pre-trained ResNet50 backbone and FPN Neck.

## Dataset Description, German Traffic Sign Detection Benchmark (GTSDB)

The GTSDB dataset is part of the IJCNN 2013 Traffic Sign Detection benchmark and contains real-world traffic scene images with annotated traffic signs. The dataset provides high-resolution images and standardized ground-truth annotations for evaluating object detection models.

###  Image Format

* Images are stored in PPM format with a resolution of 1360 × 800 pixels.

* Each image contains 0–6 traffic signs, appearing at sizes between 16×16 and 128×128 pixels.

* Signs may vary in perspective, lighting, and environment, making the dataset suitable for training robust detectors.

### Annotation Format

Annotations are provided in a semicolon-separated CSV file (gt.txt), where each entry contains:

* Filename

* Bounding box: x1; y1; x2; y2

* Class ID: integer representing the traffic sign category

#### Example fields:

image_xx.ppm; left; top; right; bottom; class_id

The dataset follows the class ID definitions described in the official ReadMe.txt file of the GTSDB package.


### Dataset Splits

The GTSDB dataset includes the following official splits (IJCNN 2013):

* FullIJCNN2013.zip → 900 total images

* TrainIJCNN2013.zip → 600 training images

* TestIJCNN2013.zip → 300 test images (no ground truth)

* gt.txt → ground-truth annotations for training and evaluation

In the project, after cleaning and filtering bounding boxes, I load:

* Total samples used: 506 images

* Train split: 404 images

* Validation split: 102 images

### Dataset Download

The dataset is downloaded automatically when running the data upload section in the notebooks. Users only need to upload their kaggle.json file, after which the code handles authentication, dataset download, and extraction.

### Data Visualization
I visualize samples from the dataset with bounding boxes and class IDs overlaid on the images.
Each numeric label corresponds to a traffic sign class defined in the GTSDB label specification.

<p align="center">
<img width="794" height="498" alt="image" src="https://github.com/user-attachments/assets/7d012d07-54af-4780-8517-7cd8e31cc608" />
</p>

### Data Augmentation
To improve robustness, I apply lightweight augmentations using Albumentations, ensuring consistency between images and bounding boxes:

~~~python
train_aug = A.Compose([
    A.Rotate(limit=8, p=0.3),
    A.RandomBrightnessContrast(p=0.3),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
~~~

These augmentations help improve model robustness by introducing small rotations and brightness/contrast variations.

<p align="center">
<img width="950" height="290" alt="image-1" src="https://github.com/user-attachments/assets/982e42bc-1f4d-426a-9130-194b3301dbd6" />
</p>


### Train/Validation Split
I split the dataset into 80% training and 20% validation using a reproducible train_test_split based on image indices. Both subsets share the same annotations, with the training set using augmentations and the validation set kept clean for unbiased evaluation.
~~~
Loaded 506 images with bounding boxes. (transforms=no)
Total samples: 506 | Train: 404 | Val: 102
Loaded 404 images with bounding boxes. (transforms=yes)
Loaded 102 images with bounding boxes. (transforms=no)
~~~

## Model Description
This baseline detector uses the standard and widely adopted combination of ResNet50 as the backbone, a Feature Pyramid Network (FPN) as the neck, and Faster R-CNN as the detection head. I design follows the canonical implementation recommended by the official PyTorch documentation for Faster R-CNN, which outlines how pretrained backbones ResNet50 can be combined with FPN for multi-scale feature extraction. I also take inspiration from the article [Pipeline for Training Faster R-CNN Object Detection Models with PyTorch](https://visionbrick.com/pipeline-for-training-custom-faster-rcnn-object-detection-models-with-pytorch/), which explains the training workflow, backbone construction, and practical considerations when fine-tuning detection models. Using these two authoritative sources, I adopt the same best-practice architecture and configuration to ensure strong baseline performance before introducing custom experiments or model variations.

###  Backbone: ResNet50
For feature extraction, I use a ResNet50 backbone initialized with IMAGENET1K_V2 pretrained weights.
ResNet50 is a deep residual network that excels at learning hierarchical visual features, making it ideal for object detection tasks. Using pretrained ImageNet weights helps the model converge faster and improves generalization, especially when the training dataset is relatively small (as in GTSDB).

In implementation, the backbone is created using:

~~~python
backbone = resnet_fpn_backbone(
    backbone_name="resnet50",
    weights="IMAGENET1K_V2",
    trainable_layers=3
)
~~~

The trainable_layers=3 option fine-tunes only the higher layers of ResNet50 while freezing the earliest ones. This balances performance with stability; lower layers capture general textures and edges, while higher layers adapt to traffic sign features.

### Neck: FPN

The neck of the model is a Feature Pyramid Network (FPN) automatically integrated through resnet_fpn_backbone.
FPN enhances multiscale feature representation by combining low-resolution, high-semantic features with high-resolution, low-semantic features. This is essential for GTSDB images, where traffic signs can vary in size from 16×16 to 128×128 pixels.

FPN gives Faster R-CNN access to feature maps at multiple scales, improving detection performance on both small and medium-sized signs. PyTorch’s backbone utility handles this seamlessly, producing a multi-level feature map dictionary compatible with the Faster R-CNN head.

### Head: Faster R-CNN
The detection head is the standard Faster R-CNN module from TorchVision.
It consists of:

* Region Proposal Network (RPN )proposes candidate bounding boxes

* ROIAlign extracts fixed-size features from the FPN outputs

* ROI heads classify each proposal and refine its bounding box

I pass the ResNet50-FPN backbone into the FasterRCNN module:
~~~python
model = FasterRCNN(
    backbone=backbone,
    num_classes=num_classes,
    min_size=600,
    max_size=1000
)
~~~

## Training Configuration

To ensure reliable and comparable results across the three detection algorithms that I'll be uploading later, I enforce the same training hyperparameters for every experiment.

The shared settings are:

* Epochs: 25

* Mixed Precision (AMP): Enabled (using GradScaler)

* Progress Logging: Batch-wise loss monitoring using tqdm

* Checkpointing: Best model saved in a timestamped folder

These settings provide:

* Stable gradients

* Faster training

* Lower memory usage

* A consistent experimental environment

* Fair cross-model comparison
### Training and Validation Loss

The training and validation curves shows the baseline pre-trained model achieving rapid convergence with training loss decreasing from 0.48 to 0.06 and validation loss stabilizing at 0.13 by epoch 25, demonstrating effective transfer learning with minimal overfitting. 

<p align="center">
    
![baseline_curve](https://github.com/user-attachments/assets/3a40af57-c2be-4bc4-9cef-ff84f203a00d)

</p>


## Results

| Metric | Value |
|------|------|
| Precision | 89.0% |
| Recall | 89.0% |
| mAP | 89.0% |

### Qualitative Result
The following visualization shows predicted bounding boxes closely matching the ground truth, demonstrating strong localization and classification performance.
<img width="1016" height="1490" alt="Ground_Truth_Image_result" src="https://github.com/user-attachments/assets/58c736d1-5396-43ba-9ca2-7fa5be095404" />


## Conclusion

In this repository, I established a strong and reliable baseline traffic sign detection system using a pre-trained Faster R-CNN with a ResNet50 backbone and Feature Pyramid Network (FPN), evaluated on the German Traffic Sign Detection Benchmark (GTSDB). The goal of this baseline was not architectural novelty, but correctness, stability, and reproducibility, ensuring that all components of the detection pipeline—from data loading and augmentation to training and evaluation—are properly implemented and well understood.

By leveraging ImageNet-pretrained weights and a standard Faster R-CNN design, the model converges quickly and achieves high precision, recall, and mAP, despite the relatively small size of the dataset. The results demonstrate that transfer learning combined with a well-established detection architecture provides a solid reference point for traffic sign detection tasks. Most importantly, this baseline serves as a controlled foundation against which more advanced architectural modifications can be fairly compared.

This repository intentionally focuses only on the baseline detector. Architectural experimentation and deeper model customization are explored in separate repositories to keep each contribution focused, readable, and easy to evaluate.

## Next Steps and Related Repositories

This project is part of a larger, structured exploration of traffic sign detection using Faster R-CNN. The following two repositories build directly on this baseline:

### 1. Faster R-CNN with Custom Neck

Repository: [traffic-sign-detection-faster-rcnn-custom-neck](https://github.com/Ureed-Hussain/Traffic-sign-detection-faster-rcnn-custom-neck)

In this repository, I extend the baseline detector by designing and integrating a custom neck module in place of the standard FPN. The goal is to investigate how alternative feature aggregation strategies affect multi-scale representation, detection accuracy, and training stability, while keeping the backbone and detection head unchanged. This allows for an isolated and fair evaluation of neck design choices.

### 2. Faster R-CNN Implemented from Scratch

Repository: faster-rcnn-from-scratch-traffic-sign-detection

This repository presents a full custom implementation of Faster R-CNN, including the backbone, neck, Region Proposal Network (RPN), and ROI heads. Instead of relying on TorchVision’s high-level abstractions, I reimplement all major components to gain a deeper understanding of the internal mechanics of two-stage detectors. This version emphasizes architectural transparency, modularity, and learning-driven design decisions rather than pretrained convenience.

Together, these three repositories form a progressive and well-scoped study of traffic sign detection, moving from a strong baseline to partial customization, and finally to a full from-scratch implementation.
