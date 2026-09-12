# MPFNet

A visible-light and infrared fusion object detection model based on YOLOv8.

## Results

All results are reported in the paper **MPFNet: Multiscale Perception Fusion
Network for Multispectral Object Detection**. Models are trained for 300 epochs
with a batch size of 8, SGD (momentum 0.937, initial learning rate 0.01) and
640x640 inputs on a single NVIDIA A40 GPU.

### DroneVehicle

| Method | Modality | mAP50 | mAP |
| --- | --- | --- | --- |
| Halfway Fusion | RGB+T | 70.0 | - |
| TSFADet | RGB+T | 73.1 | - |
| TarDAL | RGB+T | 72.6 | 43.3 |
| C2Former | RGB+T | 74.2 | 47.5 |
| CAGTDet | RGB+T | 74.5 | - |
| MPFNet | RGB+T | **78.7** | **57.3** |

### M3FD

| Method | mAP50 | mAP75 | mAP |
| --- | --- | --- | --- |
| TarDAL | 80.4 | - | 53.1 |
| RFNet | 80.4 | - | 53.2 |
| CFT | 85.3 | 44.6 | 57.1 |
| MFFNet | 88.1 | - | 57.6 |
| MPFNet | **88.4** | **63.3** | **59.3** |

### FLIR (aligned)

| Method | mAP50 | mAP75 | mAP |
| --- | --- | --- | --- |
| YOLOv5s | 72.8 | 32.0 | 37.1 |
| CFT | 77.1 | 35.2 | 39.8 |
| ICAFusion | 78.2 | 36.5 | 40.4 |
| LRAF-Net | 79.5 | - | 40.8 |
| RSDet | 80.1 | - | 40.1 |
| MPFNet | 78.6 | **38.3** | **43.7** |

### Ablation on DroneVehicle (mAP50, %)

| MPC | DAF | CMCC | mAP50 |
| --- | --- | --- | --- |
| | | | 73.5 |
| Yes | | | 75.2 |
| | Yes | | 75.6 |
| Yes | Yes | | 77.2 |
| Yes | Yes | Yes | **78.7** |

## Installation

The project is built on MMYOLO 0.6.0 and MMDetection. The code is developed
with Python 3.8 and PyTorch 1.8.1 on Ubuntu, with CUDA support.

```shell
conda create -n mpfnet python=3.8 -y
conda activate mpfnet

# Install a PyTorch build matching your CUDA version, then:
pip install -r requirements.txt
pip install -v -e .
```

Refer to the MMYOLO installation documentation for the matching versions of
mmcv and mmdet if you resolve dependencies manually.

## Data Preparation

Experiments use the aligned RGB-thermal pairs of three public datasets:

- **DroneVehicle**: 28,439 paired aerial images, five vehicle categories
  (car, truck, freight-car, bus, van).
- **M3FD**: 2,905 pairs for training and 1,295 pairs for testing, six
  categories (person, car, bus, motorcycle, lamp, truck), following the split
  used by TFDet.
- **FLIR**: the strictly aligned version with 4,129 training pairs and 1,013
  testing pairs, three categories (person, car, bicycle).

Organize each dataset in COCO format and update the data roots in the
configuration files before training. Note that the configs under
`test_dataset/` currently contain the absolute paths from the authors' server
(e.g. `/data/zfy/mmyolo/...`); replace them with your local repository and
dataset paths.

## Training and Testing

Dataset-specific configuration files are provided under `test_dataset/`:

```
test_dataset/DRONEVEHICLE/
test_dataset/M3FD/
test_dataset/FLIR/
```

Train a model with the standard MMYOLO entry point:

```shell
python tools/train.py test_dataset/FLIR/v8_pzconv2.py \
    --work-dir work_dirs/mpfnet_flir
```

Test a checkpoint:

```shell
python tools/test.py test_dataset/FLIR/v8_pzconv2.py \
    work_dirs/mpfnet_flir/best_coco_bbox_mAP50_epoch_300.pth
```

Multi-GPU training follows the MMYOLO convention, for example:

```shell
CUDA_VISIBLE_DEVICES=0,1 ./tools/dist_train.sh test_dataset/M3FD/m3fd_v8.py 2
```

## Code Structure

The components introduced by MPFNet are located under
`mmyolo/models/backbones/custom/`:

- `custom.py`: the multi-scale perception convolution (`Pzconv`) and the
  YOLOv8-compatible wrapper used as the backbone convolution.
- `custom_csp_backbone.py`: the YOLOv8 CSPDarknet backbone variant with
  `Pzconv` integrated into its stages.
- `my_fusion_block.py`: channel/spatial attention blocks, the SE layer and the
  dynamic adaptive fusion block (`FCM`).
- `my_fusiondetector_loss.py`: the dual-branch detector variant that combines
  the two backbones with the fusion blocks.

## Acknowledgement

This project is based on [MMYOLO](https://github.com/open-mmlab/mmyolo) and
[MMDetection](https://github.com/open-mmlab/mmdetection). We thank the OpenMMLab
project and the authors of the public datasets used in this work.

## License

This repository is released under the Apache 2.0 license, consistent with
MMYOLO.
