import torch
from sklearn.metrics import roc_auc_score

def calc_metrics(ground_truth, pred):
    """
    Calculate various metrics for each sample in the batch, including Accuracy, F1-score, Precision, Sensitivity, Specificity, Recall, AUC, IoU, and Dice coefficient.
    
    Parameters:
    - ground_truth (torch.Tensor): Ground truth labels with shape [channel, height, width] or [height, width]
    - pred (torch.Tensor): Predicted labels with shape [channel, height, width] or [height, width]

    Returns:
    - aggregated_metrics (dict): A dictionary containing the mean of all the calculated metrics across the batch
    """
    
    # Initialize accumulators for each metric
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_specificity = 0
    total_f1_score = 0
    total_iou = 0
    total_dice = 0
    total_auc = 0
    
    gt = ground_truth.contiguous().view(-1)
    pr = pred.contiguous().view(-1)
    
    # Convert predictions to binary labels (0 or 1)
    pr_labels = (pr > 0.5).float()
    
    # Calculate TP, TN, FP, FN
    TP = ((pr_labels == 1) & (gt == 1)).sum().item()
    TN = ((pr_labels == 0) & (gt == 0)).sum().item()
    FP = ((pr_labels == 1) & (gt == 0)).sum().item()
    FN = ((pr_labels == 0) & (gt == 1)).sum().item()
    
    # Calculate each metric
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    specificity = TN / (TN + FP + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
    iou = TP / (TP + FP + FN + 1e-8)
    dice = 2 * TP / (2 * TP + FP + FN + 1e-8)
    
    # Calculate AUC
    gt_np = gt.cpu().numpy()
    pr_np = pr.cpu().numpy()
    auc = roc_auc_score(gt_np, pr_np)
    
    # Accumulate metrics
    total_accuracy += accuracy
    total_precision += precision
    total_recall += recall
    total_specificity += specificity
    total_f1_score += f1_score
    total_iou += iou
    total_dice += dice
    total_auc += auc
    
    # Calculate mean of each metric
    aggregated_metrics = {
        'Accuracy': total_accuracy,
        'F1-score': total_f1_score,
        'Precision': total_precision,
        'Specificity': total_specificity,
        'Recall': total_recall,
        'AUC': total_auc,
        'IoU': total_iou,
        'Dice': total_dice
    }

    return aggregated_metrics

def calc_batch_metrics(ground_truth, pred):
    """
    Calculate various metrics for each sample in the batch, including Accuracy, F1-score, Precision, Sensitivity, Specificity, Recall, AUC, IoU, and Dice coefficient.
    
    Parameters:
    - ground_truth (torch.Tensor): Ground truth labels with shape [batch_size, channel, height, width] or [batch_size, height, width]
    - pred (torch.Tensor): Predicted labels with shape [batch_size, channel, height, width] or [batch_size, height, width]

    Returns:
    - aggregated_metrics (dict): A dictionary containing the mean of all the calculated metrics across the batch
    """
    batch_size = ground_truth.shape[0]
    
    # Initialize accumulators for each metric
    total_accuracy = 0
    total_precision = 0
    total_recall = 0
    total_specificity = 0
    total_f1_score = 0
    total_iou = 0
    total_dice = 0
    total_auc = 0
    
    for i in range(batch_size):
        gt = ground_truth[i].view(-1)
        pr = pred[i].view(-1)
        
        # Convert predictions to binary labels (0 or 1)
        pr_labels = (pr > 0.5).float()
        
        # Calculate TP, TN, FP, FN
        TP = ((pr_labels == 1) & (gt == 1)).sum().item()
        TN = ((pr_labels == 0) & (gt == 0)).sum().item()
        FP = ((pr_labels == 1) & (gt == 0)).sum().item()
        FN = ((pr_labels == 0) & (gt == 1)).sum().item()
        
        # Calculate each metric
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        specificity = TN / (TN + FP + 1e-8)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)
        iou = TP / (TP + FP + FN + 1e-8)
        dice = 2 * TP / (2 * TP + FP + FN + 1e-8)
        
        # Calculate AUC
        gt_np = gt.cpu().numpy()
        pr_np = pr.cpu().numpy()
        auc = roc_auc_score(gt_np, pr_np)
        
        # Accumulate metrics
        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall
        total_specificity += specificity
        total_f1_score += f1_score
        total_iou += iou
        total_dice += dice
        total_auc += auc
    
    # Calculate mean of each metric
    aggregated_metrics = {
        'Accuracy': total_accuracy / batch_size,
        'F1-score': total_f1_score / batch_size,
        'Precision': total_precision / batch_size,
        'Specificity': total_specificity / batch_size,
        'Recall': total_recall / batch_size,
        'AUC': total_auc / batch_size,
        'IoU': total_iou / batch_size,
        'Dice': total_dice / batch_size
    }

    return aggregated_metrics

if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ground_truth = torch.tensor([[[[0, 1], [1, 0]]], [[[1, 0], [0, 1]]]], dtype=torch.float32).to(device)  # shape [2, 1, 2, 2]
    pred = torch.tensor([[[[0.1, 0.9], [0.8, 0.2]]], [[[0.9, 0.1], [0.2, 0.8]]]], dtype=torch.float32).to(device)  # shape [2, 1, 2, 2]

    metrics = calc_batch_metrics(ground_truth, pred)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")