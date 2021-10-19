import torch


def generate_square_subsequent_mask(sz):
    """
    注意力掩码生成函数，生成 shape = [sz, sz] 的矩阵

    Args:
        sz (int): 矩阵尺寸

    Returns:
        [Tensor]: [sz, sz]
    """
    # float('-inf') -2**32+1
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, (-2**32+1)).masked_fill(mask == 1, float(0.0))
    return mask
