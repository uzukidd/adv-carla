from abc import ABC, abstractmethod

from pcdet.models import dense_heads, detectors


class encode_adversarial_target(ABC):

    def __init__(self, model: detectors.Detector3DTemplate):
        super().__init__()
        self.model = model

    # data_dict:dict, gt_boxes
    @abstractmethod
    def encode_target(self, batch_dict: dict, pred_dicts: dict):
        """
        Args:
            batch_dict: dict
            pred_dicts: dict
        Returns:
            data_dict["pred_scores"]
            data_dict["pred_boxes"]
            data_dict["pred_labels"]
        """
        raise NotImplementedError


class end_to_end_detector(encode_adversarial_target):

    def __init__(self, model: detectors.Detector3DTemplate):
        super().__init__(model=model)

    def encode_target(self, batch_dict, pred_dicts):
        return pred_dicts

class pointrcnn_rpn(encode_adversarial_target):

    def __init__(self, model: detectors.PointRCNN):
        super().__init__(model=model)
        self.point_head: dense_heads.PointHeadBox = model.get_submodule("point_head")

    def encode_target(self, batch_dict, pred_dicts):
        point_cls_preds = self.point_head.forward_ret_dict["point_cls_preds"]
        point_box_preds = self.point_head.forward_ret_dict["point_box_preds"]

        point_cls_preds, point_box_preds = self.point_head.generate_predicted_boxes(
            points=batch_dict["point_coords"][:, 1:4],
            point_cls_preds=point_cls_preds,
            point_box_preds=point_box_preds,
        )

        point_cls_preds.argmax(-1)
        # (Pdb) self.point_head.forward_ret_dict['point_cls_preds'].size()
        # torch.Size([131072, 3])
        # (Pdb) self.point_head.forward_ret_dict['point_box_preds'].size()
        # torch.Size([131072, 8])
        import pdb

        pdb.set_trace()
        return pred_dicts


#
