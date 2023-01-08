import numpy as np
import torch
from tqdm import tqdm
from . import metrics


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def format_logs(logs):
    str_logs = ["{}={:.3}".format(k, v) for k, v in logs.items()]
    return ", ".join(str_logs)


def metric(input, target):
    """
    Args:
        input (tensor): prediction
        target (tensor): reference data

    Returns:
        float: harmonic fscore without including backgorund
    """
    input = torch.softmax(input, dim=1)
    scores = []

    for i in range(1, input.shape[1]):  # background is not included
        ypr = input[:, i, :, :].view(input.shape[0], -1)
        ygt = target[:, i, :, :].view(target.shape[0], -1)
        scores.append(metrics.iou(ypr, ygt).item())

    return np.mean(scores)


def train_epoch(model, optimizer, criterion, dataloader, device="cpu", floats=False):
    """_summary_

    Args:
        model (_type_): _description_
        optimizer (_type_): _description_
        criterion (_type_): _description_
        dataloader (_type_): _description_
        device (str, optional): _description_. Defaults to "cpu".

    Returns:
        _type_: _description_
    """

    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}

    model.to(device).train()

    iterator = tqdm(dataloader, desc="Train")
    for x, y, *_ in iterator:
        x = x.to(device)
        y = y.to(device)
        n = x.shape[0]

        optimizer.zero_grad()
        outputs = model.forward(x)

        if len(criterion) == 1:
            if floats:
                loss = criterion[0](outputs, y.float())
            else:
                loss = criterion[0](outputs, y)
            loss.backward()
        elif len(criterion) > 2:
            loss_0 = criterion[0](outputs, y)
            loss_0.backward(retain_graph=True)
            loss_1 = criterion[1](outputs, y.float())
            loss_1.backward(retain_graph=True)
            loss_2 = criterion[2](outputs, y)
            loss_2.backward()
            loss = loss_0 + loss_1 + loss_2
        else:
            loss_1 = criterion[0](outputs, y)
            loss_1.backward(retain_graph=True)
            loss_2 = criterion[1](outputs, y)
            loss_2.backward()
            loss = loss_1 + loss_2

        # loss_0 = criterion[0](outputs, y.float())
        # loss_0.backward(retain_graph=True)
        # loss_1 = criterion[1](outputs, y)
        # loss_1.backward()
        optimizer.step()

        # loss = loss_0 + loss_1
        # loss = loss_1

        loss_meter.update(loss.item(), n=n)

        with torch.no_grad():
            score_meter.update(metric(outputs, y), n=n)

        logs.update({"Loss": loss_meter.avg})
        logs.update({"Score": score_meter.avg})
        iterator.set_postfix_str(format_logs(logs))
    return logs


def valid_epoch(model=None, criterion=None, dataloader=None, device="cpu", floats=False):
    """_summary_

    Args:
        model (_type_, optional): _description_. Defaults to None.
        criterion (_type_, optional): _description_. Defaults to None.
        dataloader (_type_, optional): _description_. Defaults to None.
        device (str, optional): _description_. Defaults to "cpu".

    Returns:
        _type_: _description_
    """

    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    logs = {}
    model.to(device).eval()

    iterator = tqdm(dataloader, desc="Valid")
    for x, y, *_ in iterator:
        x = x.to(device)
        y = y.to(device)
        n = x.shape[0]

        with torch.no_grad():
            outputs = model.forward(x)

            # loss_0 = criterion[0](outputs, y.float())
            # loss_1 = criterion[1](outputs, y)

            # # loss = loss_0 + loss_1
            # loss = loss_1

            if len(criterion) == 1:
                if floats:
                    loss = criterion[0](outputs, y.float())
                else:
                    loss = criterion[0](outputs, y)
            elif len(criterion) > 2:
                loss_0 = criterion[0](outputs, y)
                loss_1 = criterion[1](outputs, y.float())
                loss_2 = criterion[2](outputs, y)
                loss = loss_0 + loss_1 + loss_2
            else:
                loss_1 = criterion[0](outputs, y)
                loss_2 = criterion[1](outputs, y)
                loss = loss_1 + loss_2

            loss_meter.update(loss.item(), n=n)
            score_meter.update(metric(outputs, y), n=n)

        logs.update({"Loss": loss_meter.avg})
        logs.update({"Score": score_meter.avg})
        iterator.set_postfix_str(format_logs(logs))
    return logs
