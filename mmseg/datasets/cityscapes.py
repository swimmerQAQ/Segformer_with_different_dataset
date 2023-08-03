from distutils.archive_util import make_archive
import os.path as osp
import tempfile

import mmcv
import numpy as np
from mmcv.utils import print_log
from PIL import Image

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class CityscapesDataset(CustomDataset):
    """Cityscapes dataset.

    The ``img_suffix`` is fixed to '_leftImg8bit.png' and ``seg_map_suffix`` is
    fixed to '_gtFine_labelTrainIds.png' for Cityscapes dataset.
    """

    CLASSES = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
               'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
               'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
               'bicycle')

    PALETTE = [[128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156],
               [190, 153, 153], [153, 153, 153], [250, 170, 30], [220, 220, 0],
               [107, 142, 35], [152, 251, 152], [70, 130, 180], [220, 20, 60],
               [255, 0, 0], [0, 0, 142], [0, 0, 70], [0, 60, 100],
               [0, 80, 100], [0, 0, 230], [119, 11, 32]]

    def __init__(self, **kwargs):
        super(CityscapesDataset, self).__init__(
            img_suffix='_leftImg8bit.png',
            seg_map_suffix='_gtFine_labelTrainIds.png',
            **kwargs)

    @staticmethod
    def _convert_to_label_id(result):
        """Convert trainId to id for cityscapes."""
        if isinstance(result, str):
            result = np.load(result)
        import cityscapesscripts.helpers.labels as CSLabels
        result_copy = result.copy()
        for trainId, label in CSLabels.trainId2label.items():
            result_copy[result == trainId] = label.id

        return result_copy

    def results2img(self, results, imgfile_prefix, to_label_id):
        """Write the segmentation results to images.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]
            if to_label_id:
                result = self._convert_to_label_id(result)
            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            output = Image.fromarray(result.astype(np.uint8)).convert('P')
            import cityscapesscripts.helpers.labels as CSLabels
            palette = np.zeros((len(CSLabels.id2label), 3), dtype=np.uint8)
            for label_id, label in CSLabels.id2label.items():
                palette[label_id] = label.color

            output.putpalette(palette)
            output.save(png_filename)
            result_files.append(png_filename)
            prog_bar.update()

        return result_files

    def format_results(self, results, imgfile_prefix=None, to_label_id=True):
        """Format the results into dir (standard format for Cityscapes
        evaluation).

        Args:
            results (list): Testing results of the dataset.
            imgfile_prefix (str | None): The prefix of images files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            to_label_id (bool): whether convert output to label_id for
                submission. Default: False

        Returns:
            tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                for saving json/png files when img_prefix is not specified.
        """

        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        if imgfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            imgfile_prefix = tmp_dir.name
        else:
            tmp_dir = None
        result_files = self.results2img(results, imgfile_prefix, to_label_id)

        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='mIoU',
                 logger=None,
                 imgfile_prefix=None,
                 efficient_test=False):
        """Evaluation in Cityscapes/default protocol.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.
            imgfile_prefix (str | None): The prefix of output image file,
                for cityscapes evaluation only. It includes the file path and
                the prefix of filename, e.g., "a/b/prefix".
                If results are evaluated with cityscapes protocol, it would be
                the prefix of output png files. The output files would be
                png images under folder "a/b/prefix/xxx.png", where "xxx" is
                the image name of cityscapes. If not specified, a temp file
                will be created for evaluation.
                Default: None.

        Returns:
            dict[str, float]: Cityscapes/default metrics.
        """

        eval_results = dict()
        metrics = metric.copy() if isinstance(metric, list) else [metric]
        if 'cityscapes' in metrics:
            eval_results.update(
                self._evaluate_cityscapes(results, logger, imgfile_prefix))
            metrics.remove('cityscapes')
        if len(metrics) > 0:
            eval_results.update(
                super(CityscapesDataset,
                      self).evaluate(results, metrics, logger, efficient_test))

        return eval_results

    def _evaluate_cityscapes(self, results, logger, imgfile_prefix):
        """Evaluation in Cityscapes protocol.

        Args:
            results (list): Testing results of the dataset.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            imgfile_prefix (str | None): The prefix of output image file

        Returns:
            dict[str: float]: Cityscapes evaluation results.
        """
        try:
            import cityscapesscripts.evaluation.evalPixelLevelSemanticLabeling as CSEval  # noqa
        except ImportError:
            raise ImportError('Please run "pip install cityscapesscripts" to '
                              'install cityscapesscripts first.')
        msg = 'Evaluating in Cityscapes style'
        if logger is None:
            msg = '\n' + msg
        print_log(msg, logger=logger)

        result_files, tmp_dir = self.format_results(results, imgfile_prefix)

        if tmp_dir is None:
            result_dir = imgfile_prefix
        else:
            result_dir = tmp_dir.name

        eval_results = dict()
        print_log(f'Evaluating results under {result_dir} ...', logger=logger)

        CSEval.args.evalInstLevelScore = True
        CSEval.args.predictionPath = osp.abspath(result_dir)
        CSEval.args.evalPixelAccuracy = True
        CSEval.args.JSONOutput = False

        seg_map_list = []
        pred_list = []

        # when evaluating with official cityscapesscripts,
        # **_gtFine_labelIds.png is used
        for seg_map in mmcv.scandir(
                self.ann_dir, 'gtFine_labelIds.png', recursive=True):
            seg_map_list.append(osp.join(self.ann_dir, seg_map))
            pred_list.append(CSEval.getPrediction(CSEval.args, seg_map))

        eval_results.update(
            CSEval.evaluateImgLists(pred_list, seg_map_list, CSEval.args))

        if tmp_dir is not None:
            tmp_dir.cleanup()

        return eval_results
    def test_img_lab(self, results, imgfile_prefix=None, to_label_id=True):
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        if imgfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            imgfile_prefix = tmp_dir.name
        else:
            tmp_dir = None
        predicts = []
        store_score = []
        store_name = []
        for idx in range(len(self)):
            result = results[idx]
            predict = np.uint8(np.load(result))
            
            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]
            # print(basename,filename)
            npy_name = filename[:4] # 原来是7
            import glob, cv2, imageio, os
            write_path = '/SSD_DISK/users/kuangshaochen/SegFormer/junge_temp/100scenes/'
            #################################### 2023.1.23 chenyurui test 直接输出png
            os.makedirs(write_path+ filename.split('/')[0] + '/labels' , exist_ok = True)
            number = filename.split('/')[-1][:4]
            # print(write_path+ filename.split('/')[0] + '/labels/' + number + '.png')
            # print(predict.shape)
            # imageio.imwrite(write_path+ filename.split('/')[0] + '/labels/' + number + '.png' , predict)
            imageio.imwrite(write_path+ filename.split('/')[0] + '/labels/' + filename.split('/')[-1] , predict)
            ###############################################33
            ########################################using miou lidar  scene010_0_gtFine_labelTrainIds
            # mmseg_mask = np.array([0, 1, 7, 7, 7, 7, 7, 7, 2, 3, 255, 255, 255, 4, 5, 6, 255, 255, 255,])
            # pre_name = basename.replace("_leftImg8bit" , ".png")
            # # trans_pre = mmseg_mask[predict]
            # cv2.imwrite("/SSD_DISK/users/kuangshaochen/all_scene_6camlidar/" + pre_name[:10] + "_mmseg_old/"+pre_name , predict)
            # print(pre_name , )
            ##################################################################可注释代码块
            # predicts.append(predict)
            ########################333

            ############################################################## liwenye train
            ###################################################################333
            label = imageio.imread('/HDD_DISK/datasets/data/cityscapes/gtFine/val/'+ \
                filename.replace('leftImg8bit' , 'gtFine_labelTrainIds'))
            os.makedirs(write_path+ filename.split('/')[0] + '/gt' , exist_ok = True)
            imageio.imwrite(write_path+ filename.split('/')[0] + '/gt/' + filename.split('/')[-1] , label)
            os.makedirs(write_path+ filename.split('/')[0] + '/images/' , exist_ok = True)
            os.system('cp '+'/HDD_DISK/datasets/data/cityscapes/leftImg8bit/val/'+ filename+ ' ' + write_path+ filename.split('/')[0] + '/images/')
            
            
            # os.makedirs(write_path+ filename.split('/')[0] + '/logits' , exist_ok = True)
            # predicts = np.stack(predicts, axis=0)
            # print(predicts.shape)
            # np.save(write_path+ filename.split('/')[0] + '/logits/'+filename.split('/')[-1]+".npy" , predict)
            # mask = label != 255
            # new_label = label[mask]
            # new_predict = predict[mask]
            # def genConfusionMatrix(pred, label, n):
            #     '''
            #     Parameters
            #     ----------
            #     pred : 预测数组.
            #     label : 标签数组.
            #     n : 类别数(不包括背景).
            #     '''
            #     return np.bincount(n*label+pred, minlength=n**2).reshape(n, n)

            # def per_class_iu(hist):
            #     return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
            # hist = genConfusionMatrix(new_predict , new_label , 19)
            # temp = np.nan_to_num( per_class_iu(hist) )[np.unique(label)[:-2]]
            # # print(temp, temp.mean() , np.unique(label))
            # accuracy = (new_predict == new_label).sum()/(new_label.shape[0])
            # store_name.append(filename)
            # store_score.append(str(accuracy))

            ####################################################################3
        # quit()

        ##################3 lwy
        # store_score , store_name = zip(*sorted(zip(store_score , store_name)))
        # with open("/SSD_DISK/users/kuangshaochen/6cam_base/simu_lwy/accuracy.txt" , 'w') as f :
        #     for i, item in enumerate(store_name) :
        #         f.write(item +" " + store_score[i] + "\n")
        #################
        # for i in range(len(predicts)) :
        #     print( predicts[i].shape)
        