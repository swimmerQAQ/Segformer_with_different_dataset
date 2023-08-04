## Description
Most of this repo are copied from ```https://github.com/NVlabs/SegFormer.git``` 
are used for Segformer train and evaluation.
## Small modification
In order to utilize different dataset, simply transform different kind images into [mmseg_cityscapes_format](https://github.com/open-mmlab/mmsegmentation.git). 
```
You should change: ./local_configs/_base_/datasets/cityscapes_1024x1024_repeat.py #3 
My data path
/HDD_DISK/datasets/data/cityscapes/
```
![](./example1.jpg)
Please refer to ```https://github.com/open-mmlab/mmsegmentation.git```( using txt file to select)

I provide a simple example to transform different dataset.In this repo: 

```
python mmseg_write_imgs.py
```
and label definition in ./label_test/cityscapes.json

Please change ./tool/test.py or ./tool/train.py. I modify their args to select 'scenes', because I want to do traing or test on different section of whole dataset.

If you use mmseg lastest repo or this repo, you should check
```
./mmseg/datasets/pipelines/loading.py #60
```
Be sure that you are loading the img file you need

### Installation
Most of my env are from [Segformer](https://github.com/NVlabs/SegFormer.git).

Pytorch version
```
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
mmcv-full=1.3.0
```
And you may check the ```./requirements.txt```, if you have some env error. My system is linux 20.04 with cuda version 11.8.

I doing another new in the 'on 13' commit. Hope this commends can help you.
```
conda create -n seg_temp python=3.7 -y # you can change 'seg_temp' into another name
conda activate seg_temp
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
git clone https://github.com/swimmerQAQ/Segformer_with_different_dataset.git
cd Segformer_with_different_dataset/
pip install mmcv-full==1.3.0 , ipython , attrs , timm , imageio
pip install -e . --user
# change your dataset path : data_root = '/HDD_DISK/datasets/data/cityscapes/ '
```
./local_configs/_base_/datasets/cityscapes_1024x1024_repeat.py # 3
```
chmod +x example.sh 

```
### test example
Example:  test.sh config ckpt gpu_nodes --scene will(select your scene doing test)


```
./tools/dist_test.sh local_configs/segformer/B5/segformer.b5.1024x1024.city.160k.py ./work_dirs/first_sidewalk_curb/iter_156000.pth 1  --scene waymo_test
```

![](./example2.jpg)

output: ./output_temp/ and You can modify here './mmseg/datasets/cityscapes.py #243' for your own output

Pretrained model I only used is mit-b5

Trained on four dataset (mapillary bdd idd cityscapes)
[model](https://drive.google.com/file/d/1upd5UJmH7ywloEyJZifY_Frs4IxkRojb/view?usp=drive_link)/
[log](https://drive.google.com/file/d/1Z8uFO_bprKqVndSfUpKu8qdPADYDp9ID/view?usp=drive_link)