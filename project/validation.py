import numpy as np
import utils
from torch import nn
import torch
import matplotlib.pyplot as  plt


def validation_binary(model, criterion, valid_loader, num_classes=None):
    with torch.no_grad():
        num_classes = 2
        model.eval()
        losses = []
        # jaccard = []
        confusion_matrix = np.zeros(
            (num_classes, num_classes), dtype=np.uint32)

        for inputs, targets in valid_loader:
            inputs = utils.cuda(inputs)
            targets = utils.cuda(targets)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            # jaccard += get_jaccard(targets, (outputs > 0).float())
            output_classes = (outputs > 0).float()
            output_classes = output_classes.data.cpu().numpy()
            target_classes = targets.data.cpu().numpy()
            target_classes = target_classes[0]
            confusion_matrix += calculate_confusion_matrix_from_arrays(
                output_classes, target_classes, num_classes)

        valid_loss = np.mean(losses)  # type: float
        iou = calculate_binary_iou(confusion_matrix)
        dice = calculate_binary_dice(confusion_matrix)
        # valid_jaccard = np.mean(jaccard).astype(np.float64)

        # print('Valid loss: {:.5f}, jaccard: {:.5f}'.format(valid_loss, valid_jaccard))
        print('Valid loss: {:.4f}, IoU: {:.4f}, Dice: {:.4f}'.format(valid_loss, iou, dice))
        metrics = {'valid_loss': valid_loss, 'iou': iou, 'dice': dice}
        # metrics = {'valid_loss': valid_loss, 'jaccard_loss': valid_jaccard}
        return metrics


def calculate_binary_iou(confusion_matrix):
    return confusion_matrix[1][1] / (confusion_matrix[0][1] + confusion_matrix[1][0] + confusion_matrix[1][1])


def calculate_binary_dice(confusion_matrix):
    return 2 * confusion_matrix[1][1] / (confusion_matrix[0][1] + confusion_matrix[1][0] + 2 * confusion_matrix[1][1])


def get_jaccard(y_true, y_pred):
    epsilon = 1e-15
    intersection = (y_pred * y_true).sum(dim=-2).sum(dim=-1)
    union = y_true.sum(dim=-2).sum(dim=-1) + y_pred.sum(dim=-2).sum(dim=-1)

    return list(((intersection + epsilon) / (union - intersection + epsilon)).data.cpu().numpy())


def validation_multi(model: nn.Module, criterion, valid_loader, num_classes):
    with torch.no_grad():
        model.eval()
        losses = []

        confusion_matrix = np.zeros(
            (num_classes, num_classes), dtype=np.uint32)
        # print(len(valid_loader),num_classes)

        for inputs, targets in valid_loader:
            inputs = utils.cuda(inputs)
            targets = utils.cuda(targets)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            losses.append(loss.item())
            output_classes = outputs.data.cpu().numpy().argmax(axis=1)
            target_classes = targets.data.cpu().numpy()
            # plt.figure()
            # plt.subplot(121)
            # plt.imshow(output_classes[0], cmap='gray')
            # plt.subplot(122)
            # plt.imshow(target_classes[0], cmap='gray')
            # plt.show()

            confusion_matrix += calculate_confusion_matrix_from_arrays(
                output_classes, target_classes, num_classes)

        confusion_matrix = confusion_matrix[1:, 1:]  # exclude background
        valid_loss = np.mean(losses)  # type: float
        ious = {'iou_{}'.format(cls + 1): iou
                for cls, iou in enumerate(calculate_iou(confusion_matrix))}

        dices = {'dice_{}'.format(cls + 1): dice
                 for cls, dice in enumerate(calculate_dice(confusion_matrix))}

        average_iou = np.nanmean(list(ious.values()))
        average_dices = np.nanmean(list(dices.values()))
        metrics = {'valid_loss': valid_loss, 'iou': average_iou, 'dice': average_dices}
        metrics.update(ious)
        metrics.update(dices)
        print(
            'Valid loss: {:.4f}, average IoU: {:.4f}, average Dice: {:.4f}'.format(valid_loss,
                                                                                   average_iou,
                                                                                   average_dices))

        return metrics


def validation_all(model: nn.Module, criterion, valid_loader, num_classes):
    with torch.no_grad():
        model.eval()
        losses = []
        confusion_matrix_binary = np.zeros((2, 2), dtype=np.uint32)
        confusion_matrix_parts = np.zeros((4, 4), dtype=np.uint32)
        confusion_matrix_instruments = np.zeros((8, 8), dtype=np.uint32)
        # print(len(valid_loader),num_classes)

        for inputs, targets_binary, targets_parts, targets_instruments in valid_loader:
            inputs = utils.cuda(inputs)
            targets_binary = utils.cuda(targets_binary)
            targets_parts = utils.cuda(targets_parts)
            targets_instruments = utils.cuda(targets_instruments)

            outputs_binary, outputs_parts, outputs_instruments = model(inputs)
            loss = criterion(outputs_binary, targets_binary, outputs_parts, targets_parts, outputs_instruments,
                             targets_instruments)
            losses.append(loss.item())

            outputs_binary_classes = (outputs_binary > 0).float()
            outputs_binary_classes = outputs_binary.data.cpu().numpy()
            targets_binary_classes = targets_binary.data.cpu().numpy()

            outputs_parts_classes = outputs_parts.data.cpu().numpy().argmax(axis=1)
            targets_parts_classes = targets_parts.data.cpu().numpy()

            outputs_instruments_classes = outputs_instruments.data.cpu().numpy().argmax(axis=1)
            targets_instruments_classes = targets_instruments.data.cpu().numpy()

            confusion_matrix_binary += calculate_confusion_matrix_from_arrays(
                outputs_binary_classes, targets_binary_classes, 2
            )
            confusion_matrix_parts+=calculate_confusion_matrix_from_arrays(
                outputs_parts_classes,targets_parts_classes,4
            )
            confusion_matrix_instruments+=calculate_confusion_matrix_from_arrays(
                outputs_instruments_classes,targets_instruments_classes,8
            )

        confusion_matrix_parts=confusion_matrix_parts[1:,1:]
        confusion_matrix_instruments=confusion_matrix_instruments[1:,1:]
        valid_loss=np.mean(losses)

        iou_binary = calculate_binary_iou(confusion_matrix_binary)
        dice_binary = calculate_binary_dice(confusion_matrix_binary)

        ious_parts = {'iou_{}'.format(cls + 1): iou
                for cls, iou in enumerate(calculate_iou(confusion_matrix_parts))}
        dices_parts = {'iou_{}'.format(cls + 1): iou
                      for cls, iou in enumerate(calculate_dice(confusion_matrix_parts))}
        iou_parts=np.nanmean(list(ious_parts))
        dice_parts=np.nanmean(list(dices_parts))

        ious_instruments = {'iou_{}'.format(cls + 1): iou
                      for cls, iou in enumerate(calculate_iou(confusion_matrix_instruments))}

        dices_instruments = {'iou_{}'.format(cls + 1): iou
                       for cls, iou in enumerate(calculate_dice(confusion_matrix_instruments))}
        iou_instruments = np.nanmean(list(ious_instruments))
        dice_instruments = np.nanmean(list(dices_instruments))

        metrics = {
            'valid_loss': valid_loss,
            'iou_binary': iou_binary, 'dice_binary': dice_binary,
            'iou_parts':iou_parts,'dice_parts':dice_parts,
            'iou_instruments':iou_instruments,'dice_instruments':dice_instruments
        }

        print(
            'Valid loss: {:.4f}, IoU_binary: {:.4f}, IoU_parts: {:.4f}, IoU_instruments'.format(
                valid_loss,iou_binary,iou_parts,iou_instruments)
        )
        return metrics



def calculate_confusion_matrix_from_arrays(prediction, ground_truth, nr_labels):
    replace_indices = np.vstack((
        ground_truth.flatten(),
        prediction.flatten())
    ).T
    confusion_matrix, _ = np.histogramdd(
        replace_indices,
        bins=(nr_labels, nr_labels),
        range=[(0, nr_labels), (0, nr_labels)]
    )
    confusion_matrix = confusion_matrix.astype(np.uint32)
    return confusion_matrix


def calculate_iou(confusion_matrix):
    ious = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = true_positives + false_positives + false_negatives
        if denom == 0:
            iou = 0
        else:
            iou = float(true_positives) / denom

        if true_positives + false_negatives == 0:
            iou = np.nan
        ious.append(iou)
    return ious


def calculate_dice(confusion_matrix):
    dices = []
    for index in range(confusion_matrix.shape[0]):
        true_positives = confusion_matrix[index, index]
        false_positives = confusion_matrix[:, index].sum() - true_positives
        false_negatives = confusion_matrix[index, :].sum() - true_positives
        denom = 2 * true_positives + false_positives + false_negatives
        if denom == 0:
            dice = 0
        else:
            dice = 2 * float(true_positives) / denom

        if true_positives + false_negatives == 0:
            dice = np.nan
        dices.append(dice)
    return dices
