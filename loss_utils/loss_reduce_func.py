import torch

def monocular_logit(iou3d: torch.Tensor, 
                        masked_cls_preds: torch.Tensor, 
                        masked_cls_preds_extended: torch.Tensor):
    """
    ## monocular_logit
    Loss form: 
    $$
    \text{Loss} = \text{Logit}
    $$
    """
    
    total_loss = masked_cls_preds
    total_loss = total_loss.sum()
    return total_loss

def monocular_iou(iou3d: torch.Tensor, 
                        masked_cls_preds: torch.Tensor, 
                        masked_cls_preds_extended: torch.Tensor):
    """
    ## monocular_iou
    
    Paper index: (7)
    
    Loss form: 
    $$
    \text{Loss} = \text{IoU}
    $$
    """
    
    total_loss = iou3d
    total_loss = total_loss.sum()
    return total_loss

def logit_multiply_iou3d(iou3d: torch.Tensor, 
                        masked_cls_preds: torch.Tensor, 
                        masked_cls_preds_extended: torch.Tensor):
    """
    ## logit_multiply_iou3d
    
    Paper index: (10)
    
    Loss form: 
    $$
    \text{Loss} = \text{IoU} \cdot \text{Logit}
    $$
    """
    
    total_loss = masked_cls_preds_extended * iou3d
    total_loss = total_loss.sum()
    return total_loss

def score_multiply_iou3d(iou3d: torch.Tensor, 
                        masked_cls_preds: torch.Tensor, 
                        masked_cls_preds_extended: torch.Tensor):
    """
    ## score_multiply_iou3d
    Loss form: (9)
    $$
    \text{Loss} = \text{IoU} \cdot \text{score}\\ \text{w.r.t}\quad \text{score} = \sigma (\text{Logit})
    $$
    """

    total_loss = torch.sigmoid(masked_cls_preds_extended) * iou3d
    total_loss = total_loss.sum()
    return total_loss

def entropy_score_loss(iou3d: torch.Tensor, 
                        masked_cls_preds: torch.Tensor, 
                        masked_cls_preds_extended: torch.Tensor):
    """
    ## physical_loss
    
    Paper index: (8)
    
    Loss form:
    $$
    \text{Loss} = -\log(1 - \text{score})\\ \text{w.r.t}\quad \text{score} = \sigma (\text{Logit})
    $$
    """
    masked_cls_preds_extended = torch.sigmoid(masked_cls_preds_extended)
    total_loss = -1.0 * torch.log(1.0 - masked_cls_preds_extended)
    total_loss = total_loss.sum()
    return total_loss

def physical_loss(iou3d: torch.Tensor, 
                        masked_cls_preds: torch.Tensor, 
                        masked_cls_preds_extended: torch.Tensor):
    """
    ## physical_loss
    
    Paper index: (4)
    
    Loss form:
    $$
    \text{Loss} = -\text{IoU} \cdot \log(1 - \text{score})\\ \text{w.r.t}\quad \text{score} = \sigma (\text{Logit})
    $$
    """
    masked_cls_preds_extended = torch.sigmoid(masked_cls_preds_extended)
    total_loss = -1.0 * torch.log(1.0 - masked_cls_preds_extended) * iou3d
    total_loss = total_loss.sum()
    return total_loss