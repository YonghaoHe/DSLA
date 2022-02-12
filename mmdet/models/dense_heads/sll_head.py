import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale, normal_init
from mmcv.runner import force_fp32
import numpy

from mmdet.core import distance2bbox, multi_apply, multiclass_nms, reduce_mean, bbox_overlaps
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead

INF = 1e8


@HEADS.register_module()
class SmoothedLabelLearningHead(AnchorFreeHead):
    """
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((0, 64), (64, 128), (128, 256), (256, 512), (512, INF)),

                 enable_interval_relaxation=True,
                 interval_relaxation_factor=0.2,
                 enable_centerness_core_zone=True,
                 enable_iou_score_coupling=True,
                 enable_iou_score_only=False,

                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,

                 loss_cls=dict(
                     type='QualityFocalLoss',
                     use_sigmoid=True,
                     beta=2.0,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.interval_relaxation_factor = interval_relaxation_factor
        assert 0.0 <= self.interval_relaxation_factor < 1.0, 'interval_relaxation_factor must be in [0, 1) !!!'
        assert loss_cls['type'] in ['QualityFocalLoss'], 'cls loss only support "QualityFocalLoss" !!!'
        self.cls_loss = loss_cls
        # get gray ranges ----------------------------------------------------
        self.interval_relaxation_ranges = [(int(low * (1 - self.interval_relaxation_factor)), int(up * (1 + self.interval_relaxation_factor))) for (low, up) in self.regress_ranges]
        self.enable_interval_relaxation = enable_interval_relaxation
        self.enable_iou_score_coupling = enable_iou_score_coupling
        self.enable_centerness_core_zone = enable_centerness_core_zone
        self.enable_iou_score_only = enable_iou_score_only

        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            **kwargs)

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])

    def init_weights(self):
        """Initialize weights of the head."""
        super().init_weights()

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:scale_relaxation_ranges = [(int(low * (1 - self.scale_relaxation_factor)), int(up * (1 + self.scale_relaxatio
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \scale_relaxation_ranges = [(int(low * (1 - self.scale_relaxation_factor)), int(up * (1 + self.scale_relaxatio
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): Centerss for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale levle.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)

        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        return cls_score, bbox_pred

    @force_fp32(apply_to=('cls_score_preds', 'bbox_reg_preds'))
    def loss(self,
             cls_score_preds,
             bbox_reg_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_score_preds (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_reg_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert len(cls_score_preds) == len(bbox_reg_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_score_preds]
        all_level_points = self.get_points(featmap_sizes, bbox_reg_preds[0].dtype, bbox_reg_preds[0].device)

        cls_score_targets, bbox_reg_targets = self.get_targets(all_level_points, gt_bboxes, gt_labels)

        num_imgs = cls_score_preds[0].size(0)
        # flatten preds
        flatten_cls_score_preds = [cls_score_pred.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
                                   for cls_score_pred in cls_score_preds]
        flatten_bbox_reg_preds = [bbox_reg_pred.permute(0, 2, 3, 1).reshape(-1, 4)
                                  for bbox_reg_pred in bbox_reg_preds]

        flatten_cls_score_preds = torch.cat(flatten_cls_score_preds)
        flatten_bbox_reg_preds = torch.cat(flatten_bbox_reg_preds)
        flatten_cls_score_targets = torch.cat(cls_score_targets)
        flatten_bbox_reg_targets = torch.cat(bbox_reg_targets)
        # repeat points to align with bbox_preds
        flatten_points = torch.cat([points.repeat(num_imgs, 1) for points in all_level_points])

        # save to file for drawing images
        # flatten_cls_score_preds_numpy = flatten_cls_score_preds.sigmoid().detach().cpu().numpy()
        # numpy.savetxt('/home/yonghaohe/projects/mmdetection-2.10.0/local_test/PR-journal_related/fcos_sll_save/cls_score_pred.txt',
        #               flatten_cls_score_preds_numpy,
        #               fmt='%.02f')
        #
        # flatten_cls_score_targets_numpy = flatten_cls_score_targets.detach().cpu().numpy()
        # numpy.savetxt('/home/yonghaohe/projects/mmdetection-2.10.0/local_test/PR-journal_related/fcos_sll_save/cls_score_target.txt',
        #               flatten_cls_score_targets_numpy,
        #               fmt='%.02f')
        #
        # flatten_points_numpy = flatten_points.detach().cpu().numpy()
        # numpy.savetxt('/home/yonghaohe/projects/mmdetection-2.10.0/local_test/PR-journal_related/fcos_sll_save/points.txt',
        #               flatten_points_numpy,
        #               fmt='%d')

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        # 此时labels是PxC的，C为类别数，首先需要获取每个point的最大值
        max_cls_score_targets, max_cls_score_indexes = flatten_cls_score_targets.max(dim=-1)
        pos_indexes = (max_cls_score_targets > 0).nonzero().reshape(-1)

        pos_bbox_reg_preds = flatten_bbox_reg_preds[pos_indexes]
        iou_score_targets = flatten_cls_score_preds.new_zeros(max_cls_score_targets.shape)

        if len(pos_indexes) > 0:
            pos_bbox_reg_targets = flatten_bbox_reg_targets[pos_indexes]
            pos_points = flatten_points[pos_indexes]
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_reg_preds)
            pos_decoded_bbox_targets = distance2bbox(pos_points, pos_bbox_reg_targets)

            # bbox_reg_weights = flatten_cls_score_preds.detach().sigmoid()
            # bbox_reg_weights = bbox_reg_weights[pos_indexes][range(len(pos_indexes)), max_cls_score_indexes[pos_indexes]]
            # bbox_reg_weights_denorm = max(reduce_mean(bbox_reg_weights.sum()), 1.0)

            bbox_reg_weights = max_cls_score_targets[pos_indexes]
            bbox_reg_weights_denorm = max(reduce_mean(bbox_reg_weights.sum()), 1.0)

            loss_bbox = self.loss_bbox(
                pos_decoded_bbox_preds,
                pos_decoded_bbox_targets,
                weight=bbox_reg_weights,
                avg_factor=bbox_reg_weights_denorm)

            iou_score_targets[pos_indexes] = bbox_overlaps(pos_decoded_bbox_preds.detach(), pos_decoded_bbox_targets, is_aligned=True)

        else:
            loss_bbox = pos_bbox_reg_preds.sum()

        if self.enable_iou_score_coupling:
            max_cls_coupled_score_targets = iou_score_targets * max_cls_score_targets
        else:
            max_cls_coupled_score_targets = max_cls_score_targets

        if self.enable_iou_score_only:
            max_cls_coupled_score_targets = iou_score_targets

        label_targets = max_cls_score_indexes * (max_cls_coupled_score_targets > 0) + self.num_classes * (max_cls_coupled_score_targets <= 0)
        cls_weights_denorm = max(reduce_mean(max_cls_coupled_score_targets.sum()), 1.0)
        loss_cls = self.loss_cls(flatten_cls_score_preds,
                                 [label_targets, max_cls_coupled_score_targets],
                                 avg_factor=cls_weights_denorm)

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox, )

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride),
                             dim=-1)  # + stride // 2
        return points

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerss targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        # expand regress ranges to align with points
        expanded_regress_ranges = [points[i].new_tensor(self.regress_ranges[i])[None].expand_as(points[i]) for i in range(num_levels)]
        # expand gray ranges
        expanded_scale_relaxation_ranges = [points[i].new_tensor(self.interval_relaxation_ranges[i])[None].expand_as(points[i]) for i in range(num_levels)]
        # expand stride
        expanded_strides_list = [points[i].new_tensor(self.strides[i]).expand(points[i].size(0)) for i in range(num_levels)]

        # concat all levels points and regress ranges
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_scale_relaxation_ranges = torch.cat(expanded_scale_relaxation_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)
        concat_strides = torch.cat(expanded_strides_list, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=concat_regress_ranges,
            interval_relaxation_ranges=concat_scale_relaxation_ranges,
            num_points_per_lvl=num_points,
            strides=concat_strides)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [bbox_targets.split(num_points, 0) for bbox_targets in bbox_targets_list]

        # concat per level image
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat([bbox_targets[i] for bbox_targets in bbox_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
        return concat_lvl_labels, concat_lvl_bbox_targets

    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges, interval_relaxation_ranges,
                           num_points_per_lvl, strides):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)

        classification_targets = gt_bboxes.new_zeros((num_points, self.num_classes))
        regression_targets = gt_bboxes.new_zeros((num_points, 4))

        if num_gts == 0:
            return classification_targets, regression_targets

        regress_ranges = regress_ranges[:, None, :].expand(num_points, num_gts, 2)
        interval_relaxation_ranges = interval_relaxation_ranges[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        gt_labels = gt_labels[None].expand(num_points, num_gts)

        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        if self.center_sampling:
            # condition1: inside a `center bbox`
            radius = self.center_sample_radius
            center_xs = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            center_ys = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            center_gts = torch.zeros_like(gt_bboxes)
            stride = center_xs.new_zeros(center_xs.shape)

            # project the points on current lvl back to the `original` sizes
            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            x_mins = center_xs - stride
            y_mins = center_ys - stride
            x_maxs = center_xs + stride
            y_maxs = center_ys + stride
            center_gts[..., 0] = torch.where(x_mins > gt_bboxes[..., 0],
                                             x_mins, gt_bboxes[..., 0])
            center_gts[..., 1] = torch.where(y_mins > gt_bboxes[..., 1],
                                             y_mins, gt_bboxes[..., 1])
            center_gts[..., 2] = torch.where(x_maxs > gt_bboxes[..., 2],
                                             gt_bboxes[..., 2], x_maxs)
            center_gts[..., 3] = torch.where(y_maxs > gt_bboxes[..., 3],
                                             gt_bboxes[..., 3], y_maxs)

            cb_dist_left = xs - center_gts[..., 0]
            cb_dist_right = center_gts[..., 2] - xs
            cb_dist_top = ys - center_gts[..., 1]
            cb_dist_bottom = center_gts[..., 3] - ys
            center_bbox = torch.stack(
                (cb_dist_left, cb_dist_top, cb_dist_right, cb_dist_bottom), -1)
            inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        else:
            # condition1: inside a gt bbox
            inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0

        # centerness score
        filtered_bbox_targets = bbox_targets * inside_gt_bbox_mask[..., None].expand((num_points, num_gts, 4))  # 过滤掉不在bbox中的point
        point_centerness_scores = self.centerness_score(filtered_bbox_targets)  # PxN

        if self.enable_centerness_core_zone:
            gt_bboxes_center_x = (gt_bboxes[..., 0] + gt_bboxes[..., 2]) / 2
            gt_bboxes_center_y = (gt_bboxes[..., 1] + gt_bboxes[..., 3]) / 2
            strides = strides[..., None].expand((num_points, num_gts))
            core_zone_left = gt_bboxes_center_x - strides / 2
            core_zone_right = gt_bboxes_center_x + strides / 2
            core_zone_top = gt_bboxes_center_y - strides / 2
            core_zone_bottom = gt_bboxes_center_y + strides / 2

            inside_core_zone = (xs >= core_zone_left) & (xs <= core_zone_right) & (ys >= core_zone_top) & (ys <= core_zone_bottom)
            inside_core_zone = inside_core_zone & inside_gt_bbox_mask  # in case that the point is out of bbox

            point_centerness_scores = point_centerness_scores * (~inside_core_zone) + inside_core_zone

        # scale multiplier
        assign_measure = bbox_targets.max(-1)[0]

        if self.enable_interval_relaxation:
            scale_scores = self.interval_relaxation_score(assign_measure, regress_ranges, interval_relaxation_ranges)
        else:
            scale_scores = (regress_ranges[..., 0] <= assign_measure) & (assign_measure <= regress_ranges[..., 1])

        final_scores = point_centerness_scores * scale_scores

        positive_condition = final_scores > 0.0

        # sort scores
        sorted_final_scores, sorted_indexes = final_scores.sort(dim=1)
        intermediate_indexes = sorted_indexes.new_tensor(range(sorted_indexes.size(0)))[..., None].expand(sorted_indexes.size(0), sorted_indexes.size(1))

        # reranking
        sorted_gt_labels = gt_labels[intermediate_indexes, sorted_indexes]
        sorted_positive_condition = positive_condition[intermediate_indexes, sorted_indexes]

        indexes_1, indexes_2 = torch.where(sorted_positive_condition)
        positive_label_indexes = sorted_gt_labels[indexes_1, indexes_2]
        classification_targets[indexes_1, positive_label_indexes] = sorted_final_scores[indexes_1, indexes_2]

        # if there are more than one objects for a location,
        # we choose the one with the highest score
        _, select_indexes = sorted_final_scores.max(dim=1)
        sorted_bbox_targets = bbox_targets[intermediate_indexes, sorted_indexes]
        regression_targets = sorted_bbox_targets[range(num_points), select_indexes]

        return classification_targets, regression_targets

    def centerness_score(self, bbox_targets):
        """Compute centerness targets.

        Args:
            bbox_targets (Tensor): BBox targets of all bboxes in shape
                (num_pos, num_bbox, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = bbox_targets[..., [0, 2]]
        top_bottom = bbox_targets[..., [1, 3]]
        # clamp to avoid zero-divisor
        centerness_targets = ((left_right.min(dim=-1)[0]).clamp(min=0.0) / (left_right.max(dim=-1)[0]).clamp(min=0.01)) * \
                             ((top_bottom.min(dim=-1)[0]).clamp(min=0.0) / (top_bottom.max(dim=-1)[0]).clamp(min=0.01))
        return torch.sqrt(centerness_targets)

    def interval_relaxation_score(self, measure, regress_ranges, gray_ranges):
        #  linear
        left_gray_multiplier = (measure - gray_ranges[..., 0]) / (regress_ranges[..., 0] - gray_ranges[..., 0]).clamp(min=0.01)
        left_gray_indicator = (gray_ranges[..., 0] <= measure) & (measure < regress_ranges[..., 0])

        green_indicator = (regress_ranges[..., 0] <= measure) & (measure <= regress_ranges[..., 1])

        right_gray_multiplier = (gray_ranges[..., 1] - measure) / (gray_ranges[..., 1] - regress_ranges[..., 1]).clamp(min=0.01)
        right_gray_indicator = (regress_ranges[..., 1] < measure) & (measure <= gray_ranges[..., 1])

        relaxation_score = left_gray_multiplier * left_gray_indicator + green_indicator + right_gray_multiplier * right_gray_indicator

        return relaxation_score

    @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            bbox_pred_list = [bbox_preds[i][img_id].detach() for i in range(num_levels)]

            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(
                cls_score_list,
                bbox_pred_list,
                mlvl_points,
                img_shape,
                scale_factor,
                cfg,
                rescale,
                with_nms)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, points in zip(cls_scores, bbox_preds, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)

            nms_pre = cfg.get('nms_pre', -1)
            if 0 < nms_pre < scores.shape[0]:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)

        # Set max number of box to be feed into nms in deployment
        deploy_nms_pre = cfg.get('deploy_nms_pre', -1)
        if deploy_nms_pre > 0 and torch.onnx.is_in_onnx_export():
            max_scores, _ = mlvl_scores.max(dim=1)
            _, topk_inds = max_scores.topk(deploy_nms_pre)
            mlvl_scores = mlvl_scores[topk_inds, :]
            mlvl_bboxes = mlvl_bboxes[topk_inds, :]
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(
                mlvl_bboxes,
                mlvl_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=None)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores
