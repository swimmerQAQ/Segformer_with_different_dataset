import argparse
import os

import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction

from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
from IPython import embed


#################
import numpy as np
def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', default='work_dirs/res.pkl', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        default='mIoU',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    ###########################################################3
    #self use
    parser.add_argument('--scene',default= None)
    #################################################################
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main(myscene = None):
    args = parse_args()
    # assert args.scene != None , " 没有scene 参数读入 "
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if 'None' in args.eval:
        args.eval = None
    if args.eval and args.format_only:

        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)
##############################################################about args.scene ----------self use
    # cfg.data.test.split = 'val' + ".txt"
    if args.scene != None :
        cfg.data.test.split = args.scene + ".txt"
    if myscene != None :
        cfg.data.test.split = myscene + ".txt"

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        if cfg.data.test.type == 'CityscapesDataset':
            # hard code index
            cfg.data.test.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0
            ]
            cfg.data.test.pipeline[1].flip = True
        elif cfg.data.test.type == 'ADE20KDataset':
            # hard code index
            cfg.data.test.pipeline[1].img_ratios = [
                0.75, 0.875, 1.0, 1.125, 1.25
            ]
            cfg.data.test.pipeline[1].flip = True
        else:
            # hard code index
            cfg.data.test.pipeline[1].img_ratios = [
                0.5, 0.75, 1.0, 1.25, 1.5, 1.75
            ]
            cfg.data.test.pipeline[1].flip = True

    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        print(myscene)
        if myscene == '0066050' or myscene == None:
            # print("没有？")
            init_dist(args.launcher, **cfg.dist_params)
    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    model.PALETTE = checkpoint['meta']['PALETTE']

    efficient_test = True #False
    if args.eval_options is not None:
        efficient_test = args.eval_options.get('efficient_test', False)

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  efficient_test)
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir,
                                 args.gpu_collect, efficient_test)
    # num = 0
    # path = "/SSD_DISK/users/kuangshaochen/SegFormer/testTemp_imgs/numpy_npy_dir/"
    # for output in outputs :
    #     temp = np.load(output)
    #     np.save(path+"{num}.npy".format(num = num),temp)
    #     num += 1
    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            os.makedirs("./work_dirs" , exist_ok=True)
            mmcv.dump(outputs, args.out)
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            print("give format data")
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            print("no evaluate: test_img_lab ")
            dataset.test_img_lab(outputs, **kwargs)

            print("evaluate")
            dataset.evaluate(outputs, **kwargs)

if __name__ == '__main__':
    # scenes = ['0231', '0232', '0233', '0234', '0235',
    #  '0236', '0237', '0238', '0239', '0240', '0241',
    #   '0242', '0243', '0244', '0245', '0246', '0247',
    #    '0248', '0249', '0250', '0251', '0252', '0253',
    #     '0254', '0255', '0256', '0257', '0258', '0259',
    #      '0260', '0261', '0262', '0263', '0264', '0265',
    #       '0266', '0267', '0268', '0269']

    # '0158150', '0146130', '0196030', '0142100', '0124100',
    #  '0147030', '0162035', '0146050', '0153110', '0129170',
    #   '0149060', '0134035', '0100035', '0137075', '0187030',
    #    '0198080', '0173125', '0158030', '0133035',
    # scenes = [ '0198170',
    #     '0160065', '0159100', '0121145', '0100160', '0012065', 
    #     '0139100', '0140170', '0148045', '0131170', '0199170',
    #      '0172120', '0122085', '0113135', '0200030', '0182155',
    #       '0142030', '0119045', '0105155', '0120150', '0177030',
    #        '0188030', '0103055', '0129125', '0197030', '0167120',
    #         '0131035', '0167030', '0193100', '0120035', '0174150', '0145050']
    # '0200', '0201', '0202', '0203', '0204', '0205', '0206',
    #  '0207', '0208', '0209', '0210', '0211', '0212', '0213', '0214',
    #   '0215', '0216', '0217', '0218', '0219', '0220', '0221', '0222',
    #    '0223', '0224', '0225', '0226', '0227', '0228', '0229', '0230',
    #     '0231', '0232', '0233', '0234',
    # '0235', '0236', '0237', '0238'
    # scenes = [ '0239']
    # scenes = ['0066050', '0489125', '0028040', '0158150', '0146130', 
    # '0554035', '0196030', '0142100', '0124100', '0092110', '0042100', 
    # '0476065', '0147030', '0014075', '0032150', '0162035', '0074170', 
    # '0146050', '0453145', '0153110', '0129170', '0149060', '0079050', 
    # '0081120', '0069100', '0134035', '0100035', '0137075', '0039155', 
    # '0187030', '0527155', '0465095', '0485105', '0087070', '0173125', 
    # '0158030', '0074085', '0133035', '0053155', '0198170', '0160065', 
    # '0159100', '0036100', '0121145', '0508085', '0015155', '0100160', 
    # '0026085', '0012065', '0031150', '0084030', '0097050', '0059070', 
    # '0052080', '0005145', '0139100', '0140170', '0148045', '0078150', 
    # '0409175', '0040100', '0131170', '0199170', '0172120', '0122085', 
    # '0047140', '0220150', '0043155', '0179130', '0023120', '0027055', 
    # '0017085', '0054100', '0020150', '0113135', '0541125', '0028130', 
    # '0050100', '0029075', '0200030', '0095050', '0182155', '0142030', 
    # '0569140', '0570155', '0119045', '0105155', '0120150', '0072050', 
    # '0177030', '0188030', '0103055', '0000175', '0031030', '0020075', 
    # '0129125', '0076070', '0197030', '0167120', '0524150', '0131035', 
    # '0045125', '0167030', '0193100', '0238025', '0120035', '0101165', 
    # '0478115', '0061160', '0174150', '0067130', '0032030']
    # for scene in scenes :
    #     # scene = "{:04d}".format(scene + 100)
    #     main(myscene = scene)
    main()



    