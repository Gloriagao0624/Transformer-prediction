import torch.nn as nn
from .single import ScaledDotProductAttention, TimeAwareScaledDotProductAttention


class MultiHeadedAttention(nn.Module):
    """
    attention is all your need 多头实现

    Args:
        __init__():
            d_model (int): App item 编码的隐向量维度
            n_head (int): head number
            dropout_p (int): dropout失活比例。

        forward():
            Q (Tensor [batch size, Q length, d_model]): Target sequence Embed 
            K (Tensor [batch size, K length, d_model]): Src sequence Embed 
            V (Tensor [batch size, V length, d_model]): Src sequence Embed 
            attn_mask (Tensor [Q length, K length], optional): 注意力掩码 . Defaults to None.
            padding_mask (Tensor [batch size, K length], optional): padding 掩码  Defaults to None.

    Returns:
        forward():
            [Tensor]: [batch size, Q len, d_model]

    """
    def __init__(self, d_model, n_heads, dropout_p):
        super().__init__()

        assert d_model % n_heads == 0, '[!] item 隐向量的维数必须能被head num 整除'

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.attention = ScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout_p)

        self.fc_o = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, Q, K, V, attn_mask=None, padding_mask=None):
        """
        Args:
            Q (Tensor): Target sequence Embed [batch size, Q length, d_model]
            K (Tensor): Src sequence Embed [batch size, K length, d_model]
            V (Tensor): Src sequence Embed [batch size, V length, d_model]
            attn_mask (Tensor, optional): 注意力掩码 [Q length, K length]. Defaults to None.
            padding_mask (Tensor, optional): padding 掩码 [batch size, K length] Defaults to None.

        Returns:
            [Tensor]: [batch size, Q len, d_model]
        """
        batch_size = Q.shape[0]

        # Q,K,V计算与变形：
        # query,key,value = [batch size, (query,key,value)'s len, head dim]
        Q, K, V = self.fc_q(Q), self.fc_k(K), self.fc_v(V)

        # Q,K,V = [batch size, n head, (query,key,value)'s len, head dim]
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        x, attn = self.attention(Q, K, V, attn_mask, padding_mask, self.dropout)
        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, Q len, hid dim]
        x = x.view(batch_size, -1, self.d_model)
        x = self.fc_o(x)

        # x = self.dropout2(x) # 效果貌似差一丢丢，pytorch官方实现带了 dropout

        # x = [batch size, Q len, d_model]
        return x


class TimeAwareMultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout_p):
        """
        TiSAS 时间感知的多头注意力机制实现

        Args:
            __init__():
                d_model (int): App item 编码的隐向量维度
                n_head (int): head number
                dropout_p (int): dropout失活比例。

            forward():
                Q (Tensor [batch size, Q length, d_model]): Target sequence Embed 
                K (Tensor [batch size, K length, d_model]): Src sequence Embed 
                V (Tensor [batch size, V length, d_model]): Src sequence Embed 
                time_matrix_K (Tensor [batch size, n head, query's len, key's len, head_dim]): Q 对 K 的时间间隔矩阵 Embed 
                time_matrix_V (Tensor [batch size, n head, query's len, value's len, head_dim]): Q 对 V 的时间间隔矩阵 Embed 
                attn_mask (Tensor [Q length, K length], optional): 注意力掩码 . Defaults to None.
                padding_mask (Tensor [batch size, K length], optional): padding 掩码  Defaults to None.

    Returns:
        forward():
            [Tensor]: [batch size, Q len, d_model]


        """
        super().__init__()

        assert d_model % n_heads == 0, '[!] item 隐向量的维数必须能被head num 整除'

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.fc_q = nn.Linear(d_model, d_model)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.attention = TimeAwareScaledDotProductAttention()
        self.dropout = nn.Dropout(dropout_p)

        self.fc_o = nn.Linear(d_model, d_model)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, Q, K, V, time_matrix_K, time_matrix_V, padding_mask=None, attn_mask=None):
        batch_size = Q.shape[0]
        q_len, k_len, v_len = Q.shape[1], K.shape[1], V.shape[1]

        # Q,K,V计算与变形：
        # query,key,value = [batch size, (query,key,value)'s len, head dim]
        Q, K, V = self.fc_q(Q), self.fc_k(K), self.fc_v(V)

        # Q,K,V = [batch size, n head, (query,key,value)'s len, head dim]
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)

        # time_matrix_K/V = [batch size, n head, query's len, (key,value)'s len, head_dim]
        time_matrix_K = time_matrix_K.view(batch_size, q_len, k_len, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        time_matrix_V = time_matrix_V.view(batch_size, q_len, v_len, self.n_heads, self.head_dim).permute(0, 3, 1, 2, 4)

        x, attn = self.attention(Q, K, V, time_matrix_K, time_matrix_V, attn_mask, padding_mask, self.dropout)

        x = x.permute(0, 2, 1, 3).contiguous()

        # x = [batch size, Q len, hid dim]
        x = x.view(batch_size, -1, self.d_model)
        x = self.fc_o(x)

        # x = self.dropout2(x) # 效果貌似差一丢丢，pytorch官方实现带了 dropout

        # x = [batch size, Q len, d_model]
        return x
