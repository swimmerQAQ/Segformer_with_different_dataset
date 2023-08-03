import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .base import BaseSegmentor
#####################
import imageio
import numpy as np
import os.path as osp
##################
@SEGMENTORS.register_module()
class EncoderDecoder(BaseSegmentor):
    """Encoder Decoder segmentors.

    EncoderDecoder typically consists of backbone, decode_head, auxiliary_head.
    Note that auxiliary_head is only used for deep supervision during training,
    which could be dumped during inference.
    """

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(EncoderDecoder, self).__init__()
        self.backbone = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        self._init_decode_head(decode_head)
        self._init_auxiliary_head(auxiliary_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.init_weights(pretrained=pretrained)

        assert self.with_decode_head

    def _init_decode_head(self, decode_head):
        """Initialize ``decode_head``"""
        self.decode_head = builder.build_head(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes

    def _init_auxiliary_head(self, auxiliary_head):
        """Initialize ``auxiliary_head``"""
        if auxiliary_head is not None:
            if isinstance(auxiliary_head, list):
                self.auxiliary_head = nn.ModuleList()
                for head_cfg in auxiliary_head:
                    self.auxiliary_head.append(builder.build_head(head_cfg))
            else:
                self.auxiliary_head = builder.build_head(auxiliary_head)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        super(EncoderDecoder, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        self.decode_head.init_weights()
        if self.with_auxiliary_head:
            if isinstance(self.auxiliary_head, nn.ModuleList):
                for aux_head in self.auxiliary_head:
                    aux_head.init_weights()
            else:
                self.auxiliary_head.init_weights()

    def extract_feat(self, img):
        """Extract features from images."""
        # print("提取特征", self.backbone)
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out

    def _decode_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for decode head in
        training."""
        losses = dict()
        loss_decode = self.decode_head.forward_train(x, img_metas,
                                                     gt_semantic_seg,
                                                     self.train_cfg)

        losses.update(add_prefix(loss_decode, 'decode'))
        return losses

    def _decode_head_forward_test(self, x, img_metas):
        """Run forward function and calculate loss for decode head in
        inference."""
        seg_logits = self.decode_head.forward_test(x, img_metas, self.test_cfg)
        return seg_logits

    def _auxiliary_head_forward_train(self, x, img_metas, gt_semantic_seg):
        """Run forward function and calculate loss for auxiliary head in
        training."""
        losses = dict()
        if isinstance(self.auxiliary_head, nn.ModuleList):
            for idx, aux_head in enumerate(self.auxiliary_head):
                loss_aux = aux_head.forward_train(x, img_metas,
                                                  gt_semantic_seg,
                                                  self.train_cfg)
                losses.update(add_prefix(loss_aux, f'aux_{idx}'))
        else:
            loss_aux = self.auxiliary_head.forward_train(
                x, img_metas, gt_semantic_seg, self.train_cfg)
            losses.update(add_prefix(loss_aux, 'aux'))

        return losses

    def forward_dummy(self, img):
        """Dummy forward function."""
        seg_logit = self.encode_decode(img, None)

        return seg_logit

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # print("in forward train")
#         4
# torch.Size([2, 128, 256, 256])
# torch.Size([2, 256, 128, 128])
# torch.Size([2, 512, 64, 64])
# torch.Size([2, 1024, 32, 32])
        # print(img.shape)
        x = self.extract_feat(img)
        # print(len(x))
        # for i in range(len(x)):
        #     print(x[i].shape)
        # print(img_metas)
        # quit()
        losses = dict()
        # quit()
        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        losses.update(loss_decode)

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)

        return losses

    # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        """Inference by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        # print("using slide infer")
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, num_classes, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        return preds

    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""
        # print("using whole infer")
        seg_logit = self.encode_decode(img, img_meta)
        if rescale:
            seg_logit = resize(
                seg_logit,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit

    def inference(self, img, img_meta, rescale):
        """Inference with slide/whole style.

        Args:
            img (Tensor): The input image of shape (N, 3, H, W).
            img_meta (dict): Image info dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            rescale (bool): Whether rescale back to original shape.

        Returns:
            Tensor: The output segmentation map.
        """
        # print("using inference")
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit = self.whole_inference(img, img_meta, rescale)
        output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))
        return output

    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        # print("\nusing simple_test !!!")
        # print(img.shape,img_meta)
        seg_logit = self.inference(img, img_meta, rescale)
        seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        ############################################################################
        # print(seg_pred.shape)
        # label = seg_pred.squeeze(axis=0)
        # imageio.imwrite( "/SSD_DISK/users/kuangshaochen/my_segformer/test_imgs/label_from_mmseg/" + img_meta[0]['ori_filename'][6:],label)
        #############################################################################
        ############################################carmask
        # predict = seg_pred.squeeze()
        # temp_logits = seg_logit.cpu().numpy().squeeze().copy()
        # mask = predict == 13
        # confidence_mask = temp_logits[13] < 0.4
        # predict[:,:] = 0
        # predict[mask] = 255
        # predict[confidence_mask] = 0
        ####        # # import pdb;pdb.set_trace()
        ###### self write
        # carmask3 = img_meta[0]['filename'][49:-16]
        # imageio.imwrite("/SSD_DISK/users/kuangshaochen/carmask/mask4/00"+carmask3+".png",predict)
        ####################################################################
        ##### scene
        # scene = img_meta[0]['filename'][32:-21]
        # path = "./SSD_DISK/users/kuangshaochen/6cam_"+ scene +"_revision"
        # temp = seg_logit
        # temp = temp.cpu().numpy().squeeze()                     # scene1 = 39:43
        #                                                         # scene52 = 40:44
        # type_name = osp.join(path, "type19", img_meta[0]['filename'][40:44])
        # print(scene , type_name)
        # np.save(type_name + ".npy", temp)
        
        ###############################################################
        #别动
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred


    def aug_test(self, imgs, img_metas, rescale=True):
        """Test with augmentations.

        Only rescale=True is supported.
        """
        # print("\nusing aug_test")
        # aug_test rescale all imgs back to ori_shape for now
        assert rescale
        # to save memory, we get augmented seg logit inplace
        seg_logit = self.inference(imgs[0], img_metas[0], rescale)
        for i in range(1, len(imgs)):
            cur_seg_logit = self.inference(imgs[i], img_metas[i], rescale)
            seg_logit += cur_seg_logit
        seg_logit /= len(imgs)

        self_test_pre = seg_logit.cpu().numpy()
        self_test_pre = np.squeeze((self_test_pre))

        seg_pred = seg_logit.argmax(dim=1)
        seg_pred = seg_pred.cpu().numpy()
        ############################################carmask
        import os
        write_path = '/EXT_DISK/users/kuangshaochen/junge_temp/'
        # self_test_pre
        # print(img_metas);quit()
        filename = img_metas[0][0]['ori_filename'][:-4]
        os.makedirs(write_path+ filename.split('/')[-2] + '/logits' , exist_ok = True)
            
        np.save(write_path+ filename.split('/')[-2] + '/logits/'+filename.split('/')[-1]+".npy" , self_test_pre)
        # print(img_metas , self_test_pre.shape)
        ####################################################################
        # confidence_mask = self_test_pre[13] < 0.7
        # predict[:,:] = 0
        # # # import pdb;pdb.set_trace()
        # predict[mask] = 255
        # predict[confidence_mask] = 0

        ##################----------no threshold
        # predict = predict.astype(np.uint8)
        # contours, hierarchy = cv2.findContours(predict,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        # area = []
        # # 找到最大的轮廓
        # for k in range(len(contours)):
        #     area.append(cv2.contourArea(contours[k]))
        # max_idx = np.argmax(np.array(area))
        # # 填充最大的轮廓
        # temp = np.zeros_like(predict)
        # predict = cv2.drawContours(temp, contours, max_idx, 255, cv2.FILLED)
        # ###################

        # carmask3 = img_metas[0][0]['filename'][49:-16]
        # imageio.imwrite("/SSD_DISK/users/kuangshaochen/carmask/mask4/00"+carmask3+".png",predict)
        ####################################################################
        ##### scene
        # import os
        # scene = img_metas[0][0]['filename'][32:-21]
        # path = "/SSD_DISK/users/kuangshaochen/6cam_"+ scene +"_revision"
        # temp = seg_logit
        # temp = temp.cpu().numpy().squeeze()                               # scene52 40:44
        #                                                             # scene1 39:43
        # if os.path.exists(path + "/type19") :
        #     pass
        # else :
        #     os.mkdir(path + "/type19/")
        # type_name = path + osp.join( "/type19", img_metas[0][0]['filename'][40:44])
        # print(scene , type_name)
        # np.save(type_name + ".npy", temp)
        
        ###############################################################

        seg_pred = list(seg_pred)
        return seg_pred


# def visualize_gray(im_mat):
#     '''
#     Plot the velodyne points in the im image.
#     args:
#         im_mat: An numpy array.
#     returns:
#         im_mat: An Image instance.
#     '''
#     gray_values = np.arange(256, dtype=np.uint8)
#     color_values = map(tuple, cv2.applyColorMap(gray_values, cv2.COLORMAP_HOT).reshape(256, 3))
#     grey_to_color_map = dict(zip(gray_values, color_values))

#     im_mat = im_mat / im_mat.max() * 255

#     h, w = im_mat.shape
#     canvas = np.zeros((h, w, 3))
#     for i in range(h):
#         for j in range(w):
#             canvas[i, j] = np.array(grey_to_color_map[round(im_mat[i, j])])

#     return canvas[:,:,::-1]