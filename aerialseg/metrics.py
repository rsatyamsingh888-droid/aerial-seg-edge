import torch


def confusion_matrix(pred, target, num_classes, ignore_index):
    keep = target != ignore_index
    index = target[keep] * num_classes + pred[keep]
    counts = torch.bincount(index, minlength=num_classes ** 2)
    return counts.reshape(num_classes, num_classes)


def iou_per_class(matrix):
    correct = matrix.diag()
    union = matrix.sum(dim=0) + matrix.sum(dim=1) - correct
    return correct / union.clamp(min=1)