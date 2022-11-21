# ILoFGAN: improved local fusion generative adversarial network

Code for our paper *"*Data augmentation and intelligent fault diagnosis of planetary gearbox using ILoFGAN under extremely limited samples*"*.

Created by Mingzhi Chen, Haidong Shao, Haoxuan Dou, Wei Li, and Bin Liu.

Paper Link: [IEEE](https://ieeexplore.ieee.org/document/9931615)

![](figure/framework.jpg)

If you find our work useful in your research, please consider citing:

```
@inproceedings{ILoFGAN for fault diagnosis,
  title={Data augmentation and intelligent fault diagnosis of planetary gearbox using ILoFGAN under extremely limited samples},
  author={Mingzhi Chen, Haidong Shao, Haoxuan Dou, Wei Li, and Bin Liu},
  booktitle={IEEE Transactions on Reliability},
  year={2022}
}
```

## Prerequisites

- Hardware

  - a GPU

- Software

  * Ubuntu
  * Pytorch 1.10

  * OpenCV
  
  * numpy
## Datasets Preparation 

the original datasets can download from [gear vibration dataset of University of Connecticut](https://figshare.com/articles/Gear_Fault_Data/6127874/1),[planetary gearbox dataset of Southeast University](https://github.com/cathysiyu/Mechanical-datasets). You should transform each time-domain waveform into the corresponding time-frequency spectrum. Then, the time-frequency spectrums are visualized into an RGB three-channel time-frequency diagram with pixels of 64*64 in the form of a thermal diagram.

The Processed datasets are in `datasets`.`cwt_gearbox_jet_8_6.npy` corresponds to the dataset of University of Connecticut,`SW_gearbox_30_2_cwt_5_6.npy` corresponds to the dataset of Southeast University.

## Training

```shell
python train.py
--conf configs/cwt_sw_gearbox_ilofgan.yaml
--output_dir results/cwt_sw_gearbox_ilofgan
--gpu 0
```

* You may also customize the parameters in `configs`.
* It takes about 15 hours to train the network on a 2080Ti GPU.


## Generation&Evaluation

```shell
python main_metric.py 
--gpu 0
--dataset cwt_gearbox
--name results/cwt_sw_gearbox_ilofgan
--real_dir datasets/cwt_gearbox
--ckpt gen_00100000.pt
--fake_dir test_for_fid
```

The generated images will be saved in `results/cwt_sw_gearbox_ilofgan/test_for_fid`.

## Acknowledgement

Our code is designed based on [LoFGAN](https://github.com/edward3862/LoFGAN-pytorch)

The code for calculate FID is based on [pytorch-fid](https://github.com/mseitzer/pytorch-fid)

## Contact

If you have any questions about the codes or would like to communicate about intelligent fault diagnosis, fault detection,please contact us: [1297008453@hnu.edu.cn](mailto:fletahsy@hnu.edu.cn)

Mentor E-mail：[hdshao@hnu.edu.cn](mailto:hdshao@hnu.edu.cn)





