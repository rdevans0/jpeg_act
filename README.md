
# Code examples for JPEG_ACT

These examples are intended to replicate *JPEG-ACT: Accelerating Deep Learning via Transform-based Lossy Compression*. If you use this work, please cite our paper in ISCA 2020:

R. D. Evans, L. Liu and T. M. Aamodt, "JPEG-ACT: Accelerating Deep Learning via Transform-based Lossy Compression," 2020 ACM/IEEE 47th Annual International Symposium on Computer Architecture (ISCA), Valencia, Spain, 2020, pp. 860-873, doi: 10.1109/ISCA45697.2020.00075.

## Setup
JPEG-ACT is tested on Ubuntu 20.04 with Python 3, with CUDA 10.1. Other cuda versions can be used by editing requirements.txt. Note, some requirements are ordered, so must be installed as follows:


```bash
# Set up environment
python3 -m venv jpeg_act_venv
source jpeg_act_venv/bin/activate
bash install_requirements.sh

# Ensure cupy is installed correctly
python3 -c "import cupy; a = cupy.array([0,1,2,3,4]); print(a*2);"
```

## Running

See below for examples for replicating the fixpoint, JPEG, and JPEG_ACT results. These use a fixed seed. 

```bash
source jpeg_act_venv/bin/activate
cd jpeg_act/cifar

TRAIN_CMD="python3 train_cifar_recall_error.py --gpu 0 --dataset cifar10 --augment --snapshot_every 10 -b 128 --learnrate 0.05 -y 70 -w 5e-4 -R 4.5 --seed 12345"

# FIXPOINT 
$TRAIN_CMD --epoch 300 --model vgg --ae_fix35 B123c,Bpd,B45c
$TRAIN_CMD --epoch 300 --model rn50 --ae_fix35 Bc,R2345c,Br,R2345r,Rs,R6rc
$TRAIN_CMD --epoch 300 --model wrn --ae_fix35 BW1234cs,BWrd

# JPEG (quality 95)
$TRAIN_CMD --epoch 300 --model vgg --ae_fix35 Bpd,B45c --ae_jpeg jpeg95 B123c
$TRAIN_CMD --epoch 300 --model rn50 --ae_fix35 Br,R2345r,Rs,R6rc --ae_jpeg jpeg95 Bc,R2345c
$TRAIN_CMD --epoch 300 --model wrn --ae_fix35 BWrd --ae_jpeg jpeg95 BW1234cs

# JPEG_ACT (optsL followed by optsH)
$TRAIN_CMD --epoch 10 --model vgg --ae_fix35 Bpd,B45r --ae_jpeg optsL B123c
$TRAIN_CMD --epoch 300 --model vgg --ae_fix35 Bpd,B45c --ae_jpeg optsH B123c --resume result/snapshot_10_iter_3907

$TRAIN_CMD --epoch 10 --model rn50 --ae_fix35 Br,R2345r,Rs,R6rc --ae_jpeg optsL Bc,R2345c
$TRAIN_CMD --epoch 300 --model rn50 --ae_fix35 Br,R2345r,Rs,R6rc --ae_jpeg optsH Bc,R2345c --resume result/snapshot_10_iter_3907

$TRAIN_CMD --epoch 10 --model wrn --ae_fix35 BWrd --ae_jpeg optsL BW1234cs
$TRAIN_CMD --epoch 300 --model wrn --ae_fix35 BWrd --ae_jpeg optsH BW1234cs --resume result/snapshot_10_iter_3907

```

For more information on specific arguments, use:
```bash
python3 train_cifar_recall_error.py --help
```

# Bibtex Citation

```bibtex
@inproceedings{jpegact2020,
  title = {{{JPEG}}-{{ACT}}: {{Accelerating Deep Learning}} via {{Transform}}-Based {{Lossy Compression}}},
  shorttitle = {{{JPEG}}-{{ACT}}},
  booktitle = {2020 {{ACM}}/{{IEEE}} 47th {{Annual International Symposium}} on {{Computer Architecture}} ({{ISCA}})},
  author = {Evans, R. D. and Liu, L. and Aamodt, T. M.},
  year = {2020},
  month = may,
  pages = {860--873},
  doi = {10.1109/ISCA45697.2020.00075},
  keywords = {CNN Training,Compression,GPU,Hardware Acceleration}
}
```

