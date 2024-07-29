import torch

def log1m_sigmoid(x):
    """
    ## log1m_sigmoid
    Form: 
    $$
    \text{log1m_sigmoid}(x) = -x - \log(1+\exp(-x)) = \log(1 - \text{sigmoid}(x)) = \log(1 - \text{score})
    $$
    """
    return -x - torch.log1p(torch.exp(-x))

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
    total_loss = log1m_sigmoid(masked_cls_preds)
    
    overflow_mask = torch.logical_or(torch.isnan(total_loss),  
                                     torch.isinf(total_loss))
    total_loss = total_loss[~overflow_mask]
    
    if total_loss.numel() != 0:
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
    total_loss = log1m_sigmoid(masked_cls_preds_extended) * iou3d
    
    overflow_mask = torch.logical_or(torch.isnan(total_loss),  
                                     torch.isinf(total_loss))
    total_loss = total_loss[~overflow_mask]

    if total_loss.numel() != 0:
        total_loss = total_loss.sum()
    return total_loss