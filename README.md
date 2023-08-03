## Description
Most of this repo are copied from ```https://github.com/NVlabs/SegFormer.git``` 
are used for Segformer train and evaluation.
## Small modification
In order to utilize different dataset, simply transform different kind images into mmseg_cityscapes_format. 
![](./example1.jpg)
Please refer to ```https://github.com/open-mmlab/mmsegmentation.git```( using txt file to select)

I provide a simple example to transform different dataset.In this repo: 
```
python mmseg_write_imgs.py
```

Please change ./tool/test.py or ./tool/train.py. I modify their args to select 'scenes', because I want to do traing or test on different section of whole dataset.

If you use mmseg lastest repo or this repo, you should check
```
./mmseg/datasets/pipelines/loading.py #60
```
Be sure that you are loading the img file you need

Example:  test.sh config ckpt gpu_nodes --scene will(select your scene doing test)


```
./tools/dist_test.sh local_configs/segformer/B5/segformer.b5.1024x1024.city.160k.py ./work_dirs/first_sidewalk_curb/iter_156000.pth 1  --scene waymo_test
```

![](./example2.jpg)

output: ./output_temp/ and You can modify here './mmseg/datasets/cityscapes.py #243' for your own output

## Installation

For install and data preparation, please refer to the guidelines in [MMSegmentation v0.13.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.13.0).

Other requirements:
```pip install timm==0.3.2```

An example (works for me): ```CUDA 10.1``` and  ```pytorch 1.7.1``` 

```
pip install torchvision==0.8.2
pip install timm==0.3.2
pip install mmcv-full==1.2.7
pip install opencv-python==4.5.1.48
cd SegFormer && pip install -e . --user
```

## Evaluation

Download `trained weights`. 
(
[google drive](https://drive.google.com/drive/folders/1GAku0G0iR9DsBxCbfENWMJ27c5lYUeQA?usp=sharing) | 
[onedrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/Ept_oetyUGFCsZTKiL_90kUBy5jmPV65O5rJInsnRCDWJQ?e=CvGohw)
)

Example: evaluate ```SegFormer-B1``` on ```ADE20K```:

```
# Single-gpu testing
python tools/test.py local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py /path/to/checkpoint_file

# Multi-gpu testing
./tools/dist_test.sh local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py /path/to/checkpoint_file <GPU_NUM>

# Multi-gpu, multi-scale testing
tools/dist_test.sh local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py /path/to/checkpoint_file <GPU_NUM> --aug-test
```

## Training

Download `weights` 
(
[google drive](https://drive.google.com/drive/folders/1b7bwrInTW4VLEm27YawHOAMSMikga2Ia?usp=sharing) | 
[onedrive](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xieenze_connect_hku_hk/EvOn3l1WyM5JpnMQFSEO5b8B7vrHw9kDaJGII-3N9KNhrg?e=cpydzZ)
) 
pretrained on ImageNet-1K, and put them in a folder ```pretrained/```.

Example: train ```SegFormer-B1``` on ```ADE20K```:

```
# Single-gpu training
python tools/train.py local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py 

# Multi-gpu training
./tools/dist_train.sh local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py <GPU_NUM>
```

## Visualize

Here is a demo script to test a single image. More details refer to [MMSegmentation's Doc](https://mmsegmentation.readthedocs.io/en/latest/get_started.html).

```shell
python demo/image_demo.py ${IMAGE_FILE} ${CONFIG_FILE} ${CHECKPOINT_FILE} [--device ${DEVICE_NAME}] [--palette-thr ${PALETTE}]
```

Example: visualize ```SegFormer-B1``` on ```CityScapes```: 

```shell
python demo/image_demo.py demo/demo.png local_configs/segformer/B1/segformer.b1.512x512.ade.160k.py \
/path/to/checkpoint_file --device cuda:0 --palette cityscapes
```





## License
Please check the LICENSE file. SegFormer may be used non-commercially, meaning for research or 
evaluation purposes only. For business inquiries, please contact 
[researchinquiries@nvidia.com](mailto:researchinquiries@nvidia.com).


## Citation
```
@article{xie2021segformer,
  title={SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers},
  author={Xie, Enze and Wang, Wenhai and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Luo, Ping},
  journal={arXiv preprint arXiv:2105.15203},
  year={2021}
}
```
