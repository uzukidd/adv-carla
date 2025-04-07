import torch
import torch.nn as nn
import torch.nn.functional as F

from cudaext.ops.Rotated_IoU.oriented_iou_loss import cal_iou_3d
from cudaext.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu

class Deepfool(nn.Module):
    
    def __init__(self):
        super().__init__()
        
    def logitwise_pertubate(self):
        pass
    
    def forward(self, input:torch.Tensor, logit: torch.Tensor):
        logit_label = torch.argsort(logit)
        
class objectwise_deepfool(nn.Module):
    
    def __init__(self, model: nn.Module, overshoot:float = 0.02, freezed_iou: bool = False, normalized: bool = False, verbose: bool = False):
        super().__init__()
        assert not freezed_iou
        assert not normalized
        self.model = model
        self.overshoot = overshoot
        self.freezed_iou = freezed_iou
        self.normalized = normalized
        self.verbose = verbose
        self.debug_msg = None
    
    @torch.no_grad()
    def objectwise_assign(self, cls_pred: torch.Tensor, box_preds: torch.Tensor, target_class_logit: int):
        max_logit_idx = -1
        empty_flag = False
        
        cls_pred = cls_pred.clone()
        pred_classes = cls_pred.argmax(dim=-1)
        target_idx_mask = (pred_classes != target_class_logit)
        
        if target_idx_mask.all():
            empty_flag = True
        else:
            cls_pred[target_idx_mask, target_class_logit] = -1.0
            max_logit_idx = cls_pred[:, target_class_logit].argmax()

        return max_logit_idx, empty_flag

    def forward(self, batch_dict, point_coords,
                deform_vert: torch.Tensor,
                gt_boxes: torch.Tensor, 
                target_class: int, 
                ret_part_loss: bool = False):
        """

        Args:
        - batch_cls_preds: [N, 3]
        - batch_box_preds: [N, 7]
        - deform_vert: [K, 3] (to get gradient)
        - gt_boxes: [N, 8]
        - point_coords: [N, 3]
        - ret_part_loss: bool

        Returns:
        - iou_3d: 
        - assign_idx: 
        - mesh_loss:
        """
        batch_cls_preds:torch.Tensor = batch_dict["batch_cls_preds"] # [N, 3]
        batch_box_preds:torch.Tensor = batch_dict["batch_box_preds"] # [N, 7]
        
        gt_boxes, gt_labels = torch.split(gt_boxes.squeeze(dim=0), [7, 1], dim=1)  # [N, 7], [N, 1]
        gt_boxes = gt_boxes[gt_labels[:, 0] == target_class]
        gt_labels = gt_labels[gt_labels[:, 0] == target_class]
        
        target_class_logit = target_class - 1
        logit_size = batch_cls_preds.size(-1)

        assert gt_boxes.size(0), "at leaset one gt box exists."
        
        box_idxs_of_pts = points_in_boxes_gpu(
                point_coords.unsqueeze(dim=0), gt_boxes.unsqueeze(dim=0)
            ).long().squeeze(dim=0)
        
        deepfooled_grad = torch.zeros_like(deform_vert)
        iou_grad = torch.zeros_like(deform_vert)
        
        for box_idx_mask in range(gt_boxes.size(0)):
            pts_idx_mask = (box_idxs_of_pts == box_idx_mask)
            
            gt_cls_preds = batch_cls_preds[pts_idx_mask]
            gt_cls_preds = F.softmax(gt_cls_preds, dim=1) # [N, 3]
            
            gt_box_preds = batch_box_preds[pts_idx_mask]
            
            max_logit_idx, empty_flag = self.objectwise_assign(cls_pred = gt_cls_preds, 
                                 box_preds = gt_box_preds,
                                 target_class_logit = target_class_logit)
                        
            pert = torch.inf
            w = torch.zeros_like(deform_vert)

            if not empty_flag:
                self.model.zero_grad()
                if deform_vert.grad is not None:
                    deform_vert.grad.zero_()
                
                target_logit = gt_cls_preds[max_logit_idx, target_class_logit]
                target_logit.backward(retain_graph=True)
                target_grad = deform_vert.grad.detach().clone()
                
                self.debug_msg = f"gtbox_(\t{box_idx_mask})_(\t{max_logit_idx}): {gt_cls_preds}"
                
                for i in range(logit_size):
                    self.model.zero_grad()
                    if deform_vert.grad is not None:
                        deform_vert.grad.zero_()
                    if i == target_class_logit:
                        continue
                    
                    cur_logit = gt_cls_preds[max_logit_idx, i]
                    cur_logit.backward(retain_graph=True)
                    cur_grad = deform_vert.grad.detach().clone()
                    
                    w_k = cur_grad - target_grad
                    f_k = cur_logit - target_logit
                    
                    if torch.norm(w_k) > 1e-5:
                        pert_k = torch.abs(f_k) / torch.norm(w_k)
                        
                        if pert_k.item() < pert:
                            pert = pert_k.item()
                            w = w_k
                
                if pert != torch.inf:
                    r_i = (pert+1e-4) * w / torch.norm(w)
                    deepfooled_grad += r_i
                
                box_selected = gt_box_preds[max_logit_idx]
                gt_box = gt_boxes[box_idx_mask]
                iou_3d = cal_iou_3d(box_selected.view(1, 1, -1), gt_box.view(1, 1, -1))
                
                self.model.zero_grad()
                if deform_vert.grad is not None:
                    deform_vert.grad.zero_()
                    
                iou_3d.backward(retain_graph=True)
                iou_grad += deform_vert.grad
                
            else:
                pass
                # print(f"gtbox_({box_idx_mask}): EMPTY")
            
        # if deepfooled_grad.__len__() > 0:
        #     deepfooled_grad = torch.stack(deepfooled_grad).mean(dim=0)
        # else:
        #     deepfooled_grad = torch.zeros_like(deform_vert)
        
        return deepfooled_grad, iou_grad
        
        
def deepfool(image, net, num_classes=10, overshoot=0.02, max_iter=50):

    """
       :param image: Image of size HxWx3
       :param net: network (input: images, output: values of activation **BEFORE** softmax).
       :param num_classes: num_classes (limits the number of classes to test against, by default = 10)
       :param overshoot: used as a termination criterion to prevent vanishing updates (default = 0.02).
       :param max_iter: maximum number of iterations for deepfool (default = 50)
       :return: minimal perturbation that fools the classifier, number of iterations that it required, new estimated_label and perturbed image
    """

    label = None

    input_shape = image.cpu().numpy().shape
    pert_image = copy.deepcopy(image)
    w = np.zeros(input_shape)
    r_tot = np.zeros(input_shape)

    loop_i = 0

    x = Variable(pert_image[None, :], requires_grad=True)
    fs = net.forward(x)
    fs_list = [fs[0,I[k]] for k in range(num_classes)]
    k_i = label

    while k_i == label and loop_i < max_iter:

        pert = np.inf
        fs[0, I[0]].backward(retain_graph=True)
        grad_orig = x.grad.data.cpu().numpy().copy()

        for k in range(1, num_classes):
            x.grad.zero_()

            fs[0, I[k]].backward(retain_graph=True)
            cur_grad = x.grad.data.cpu().numpy().copy()

            # set new w_k and new f_k
            w_k = cur_grad - grad_orig
            f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

            pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

            # determine which w_k to use
            if pert_k < pert:
                pert = pert_k
                w = w_k

        # compute r_i and r_tot
        # Added 1e-4 for numerical stability
        r_i =  (pert+1e-4) * w / np.linalg.norm(w)
        r_tot = np.float32(r_tot + r_i)

        if is_cuda:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
        else:
            pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

        x = Variable(pert_image, requires_grad=True)
        fs = net.forward(x)
        k_i = np.argmax(fs.data.cpu().numpy().flatten())

        loop_i += 1

    r_tot = (1+overshoot)*r_tot

    return r_tot, loop_i, label, k_i, pert_image
