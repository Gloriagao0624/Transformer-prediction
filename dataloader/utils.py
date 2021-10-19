import numpy as np
import time


def random_neg_from_retention(s, _list, rng):
    """
    从给定的列表 _list 中随机取 不等于 s 的值
    Args:
        s (int): 例外
        _list (list): 取值列表
        rng (object): Random包实例化对象

    Returns:
        [int]: [description]
    """
    # neg_item = random.randint(0, self.item_num-1)
    neg_item = rng.choice(_list)
    while neg_item == s:
        neg_item = rng.choice(_list)
    return neg_item


def random_neg(s, item_num, rng):
    """
    从区间中随机取 不等于 s 的值

    Args:
        s (int): 例外
        item_num (int): 取值区间 [0, item_num)
        rng (object): Random包实例化对象

    Returns:
        [int]: [description]
    """
    neg_item = rng.randint(0, item_num - 1)
    while neg_item == s:
        neg_item = rng.randint(0, item_num - 1)
    return neg_item


def compute_time_interval_matrix(time_seq, time_span=None):
    size = np.array(time_seq).shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i] - time_seq[j])
            if time_span is not None and span > time_span:
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix


def strftime(time_list):
    _time = {
        'tm_year': [],
        'tm_mon': [],
        'tm_mday': [],
        'tm_hour': [],
        'tm_min': [],
        'tm_sec': [],
        'tm_wday': [],
        'timestamp': [],
        'timestamp_m': [],
    }
    for t in time_list:
        t = time.strptime(t, "%Y-%m-%d %H:%M:%S")
        time_s = time.mktime(t)
        print(t)
        _time['tm_year'].append(t[0])
        _time['tm_mon'].append(t[1])
        _time['tm_mday'].append(t[2])
        _time['tm_hour'].append(t[3])
        _time['tm_min'].append(t[4])
        _time['tm_sec'].append(t[5])
        _time['tm_wday'].append(t[6])
        _time['timestamp'].append(time_s)
        _time['timestamp_m'].append(time_s//60)

    return _time


def padding1D(data, need_len, v, _type):
    data_len = len(data)
    data_ = np.full([need_len], v, dtype=_type)
    if data_len <= need_len:  # padding序列
        data_[need_len - data_len:] = data
    else:  # 截断序列
        data_ = data[data_len - need_len:]
    return data_


def padding2D(data, need_len, v, _type):
    data_len = len(data)
    data_ = np.full([need_len, need_len], v, dtype=_type)
    if data_len <= need_len:  # padding序列
        data_[need_len - data_len:, need_len - data_len:] = data[:, :]
    else:  # 截断序列
        data_[:, :] = data[data_len - need_len:, data_len - need_len:]
    return data_

# a = strftime(['2016-05-05 20:28:54','2016-05-05 20:28:54'])
# print(a)