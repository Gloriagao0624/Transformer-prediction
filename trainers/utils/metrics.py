import torch


def topk(scores, labels, name='Metrics', ks=[1, 2, 3, 4], bias=None, ignore_idxs=None, replace_item=None):
    """
    计算当前 Batch 准确率TopK

    Args:
        scores (Tensor): 预测值
        labels (Tensor): 样本标签
        name (str): 指标的名称. Defaults to 'Metrics'.
        ks (list): Topk中的 K example：K=[1,2,3,4,...]. Defaults to [1, 2, 3, 4].
        bias (float): 微信占比. Defaults to None.
        ignore_idxs (Tensor, optional): 掩码Tensor，shape=[num_classes]，不加入排序的位置置 0
        replace_item (Tuple, optional): 候选组: (x, y)，将 x 用 y 替换. Defaults to None.

    Returns:
        [type]: [description]
    """
    k = max(ks)
    labels = labels.view(-1, 1)  # [n] --> [n,1]
    batch_size = scores.size(0)

    metrics = {}

    # 忽略一些类别，将这些类别的预测值置0，从而禁止他们参与排序
    if ignore_idxs is not None:
        assert scores.size(-1) == ignore_idxs.size(-1), '[!] ignore_idxs的类别数和scores的类别数不一致！'
        ignore_idxs = ignore_idxs.view(-1, ignore_idxs.size(-1))
        scores = scores * ignore_idxs

    _, max_k = torch.topk(scores, k=k, dim=-1)

    if replace_item is not None:
        assert isinstance(replace_item, tuple), '[!] candidate 传参错误，应该为元组！'
        x, y = replace_item
        y = y.view(-1, 1).repeat(1, k)
        max_k = torch.where(max_k == x, y, max_k)

    for k in ks:
        top = (labels == max_k[:, 0:k]).sum().item()
        metrics['{}@{}'.format(name, k)] = top / batch_size

    # 标记一个batch中分类正确和错误的样本
    wrong_flag = torch.sum((labels == max_k), dim=1)

    # 将微信纳入Topk，占比修正
    if bias is not None:
        metrics = {k: v * (1 - bias) + bias for (k, v) in metrics.items()}

    _return = {'metrics': metrics, 'wrong_flag': wrong_flag, 'topk': max_k}
    return _return


def accuracy_topk(scores, labels, name='Metrics', ks=[1, 2, 3, 4], ignore_idxs=None):
    """
    计算当前 Batch 准确率TopK

    Args:
        scores (Tensor): 预测值
        labels (Tensor): 样本标签
        name (str): 指标的名称
        metric_ks (list): Topk中的 K example：K=[1,2,3,4,...]
        ignore_idxs (Tensor, optional): 掩码Tensor，shape=[num_classes]，不加入排序的位置置 0

    Returns:
        [dict]: [description]
    """
    metrics = {}
    batch_size = scores.size(0)

    if ignore_idxs is not None:
        assert scores.size(-1) == ignore_idxs.size(-1), '[!] ignore_idxs的类别数和scores的类别数不一致！'
        ignore_idxs = ignore_idxs.view(-1, ignore_idxs.size(-1))
        scores = scores * ignore_idxs

    _, max_k = torch.topk(scores, k=max(ks), dim=-1)

    labels = labels.view(-1, 1)  # [n] --> [n,1]
    for k in ks:
        top = (labels == max_k[:, 0:k]).sum().item()
        metrics['{}@{}'.format(name, k)] = top / batch_size
    wrong_flag = torch.sum((labels == max_k), dim=1)

    _return = {'metrics': metrics, 'wrong_flag': wrong_flag, 'topk': max_k}

    return _return


def accuracy_topk_add_bias(scores, labels, name, ks, bias, ignore_idxs=None):
    """
    将当前 Batch 计算的topk + 微信占比

    Args:
        scores (Tensor): 预测值
        labels (Tensor): 样本标签
        ks (list): Topk中的 K example：K=[1,2,3,4,...]
        bias (float): 微信占比
        ignore_idxs (Tensor, optional): 掩码Tensor，shape=[num_classes]，不加入排序的位置置 0

    Returns:
        [dict]: [description]
    """
    _return = accuracy_topk(scores, labels, name, ks, ignore_idxs)
    metrics = {k: v * (1 - bias) + bias for (k, v) in _return['metrics'].items()}
    _return.update({'metrics': metrics})
    return _return


def accuracy_topk_add_bias_and_longtail(scores, labels, longtail, name, ks, bias):
    """
    将 scores 中的第 0 类用longtail中的类别替换，再计算当前 Batch Topk + 微信占比

    Args:
        scores (Tensor [batch size, n_class]): 预测值
        labels (Tensor [batch size]): 样本标签
        longtail (Tensor [batch size]): 样本对应的长尾App
        ks (list)): Topk中的 K example：K=[1,2,3,4,...]
        bias (float): 微信占比

    Returns:
        [dict]: [description]
    """
    metrics = {}
    batch_size = scores.size(0)

    # 如果该样本对应的用户没有长尾app，则将长尾app对应第 0 类的预测值 置 0
    scores[:, 0] = scores[:, 0] * torch.ne(longtail, 0).long()

    k = max(ks)
    _, max_k = torch.topk(scores, k=k, dim=-1)
    longtail = longtail.view(-1, 1).repeat(1, k)
    max_k = torch.where(max_k == 0, longtail, max_k)

    labels = labels.view(-1, 1)  # [n] --> [n,1]
    for k in ks:
        top = (labels == max_k[:, 0:k]).sum().item()
        metrics['{}@{}'.format(name, k)] = top / batch_size

    metrics = {k: v * (1 - bias) + bias for (k, v) in metrics.items()}
    wrong_flag = torch.sum((labels == max_k), dim=1)

    _return = {'metrics': metrics, 'wrong_flag': wrong_flag, 'topk': max_k}

    return _return


def accuracy_topk_add_bias_and_longtail_and_mask(scores, labels, longtail, name, ks, bias, ignore_idxs=None):
    """
    将 scores 先用保留列表mask之后，再计算当前 Batch Topk + 微信占比

    Args:
        scores (Tensor [batch size, n_class]): 预测值
        labels (Tensor [batch size]): 样本标签
        longtail (Tensor [batch size]): 样本对应的长尾App
        ks (list)): Topk中的 K example：K=[1,2,3,4,...]
        bias (float): 微信占比
        ignore_idxs (Tensor, optional): 掩码Tensor，shape=[num_classes]，不加入排序的位置置 0

    Returns:
        [dict]: [description]
    """
    metrics = {}
    batch_size = scores.size(0)

    if ignore_idxs is not None:
        assert scores.size(-1) == ignore_idxs.size(-1), '[!] ignore_idxs的类别数和scores的类别数不一致！'
        ignore_idxs = ignore_idxs.view(-1, ignore_idxs.size(-1))
        scores = scores * ignore_idxs

    # 如果该样本对应的用户没有长尾app，则将长尾app对应第 0 类的预测值 置 0
    scores[:, 0] = scores[:, 0] * torch.ne(longtail, 0).long()

    k = max(ks)
    _, max_k = torch.topk(scores, k=k, dim=-1)
    longtail = longtail.view(-1, 1).repeat(1, k)
    max_k = torch.where(max_k == 0, longtail, max_k)

    labels = labels.view(-1, 1)  # [n] --> [n,1]
    for k in ks:
        top = (labels == max_k[:, 0:k]).sum().item()
        metrics['{}@{}'.format(name, k)] = top / batch_size

    metrics = {k: v * (1 - bias) + bias for (k, v) in metrics.items()}
    wrong_flag = torch.sum((labels == max_k), dim=1)

    _return = {'metrics': metrics, 'wrong_flag': wrong_flag, 'topk': max_k}

    return _return