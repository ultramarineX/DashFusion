import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from einops import rearrange, repeat

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class PreNormForward(nn.Module):
    def __init__(self, d_model, fn):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PreNormAttention(nn.Module):
    def __init__(self, d_model, fn):
        super().__init__()
        self.norm_q = nn.LayerNorm(d_model)
        self.norm_k = nn.LayerNorm(d_model)
        self.norm_v = nn.LayerNorm(d_model)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):
        q = self.norm_q(q)
        k = self.norm_k(k)
        v = self.norm_v(v)

        return self.fn(q, k, v)

class PostNormForward(nn.Module):
    def __init__(self, d_model, fn):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fn = fn
    def forward(self, x, **kwargs):

        return self.norm(self.fn(x, **kwargs))


class PostNormAttention(nn.Module):
    def __init__(self, d_model, fn):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fn = fn

    def forward(self, q, k, v, **kwargs):

        return self.norm(self.fn(q, k, v))


class Attention(nn.Module):
    def __init__(self, d_model, n_heads = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.scale = d_model ** -0.5
        self.to_query = nn.Linear(d_model, d_model)
        self.to_key = nn.Linear(d_model, d_model)
        self.to_value = nn.Linear(d_model, d_model)
    
    def forward(self, queries, keys, values, mask = None):
        # b: batch_size、l,n: sequence_length, _: embedding_dim，h: attention heads
        b, l, _, h = *queries.shape, self.n_heads
        _, n, _, = keys.shape
        queries = self.to_query(queries)
        keys = self.to_key(keys)
        values = self.to_value(values)
        # (batch_size, sequence_length, num_heads, dim_per_head) 
        queries = queries.view(b, l, h, -1).transpose(1,2)
        keys = keys.view(b, n, h, -1).transpose(1,2)
        values = values.view(b, n, h, -1).transpose(1,2)
        # (batch_size, num_heads, sequence_length, sequence_length) 
        dots = torch.einsum('bhid,bhjd->bhij', queries, keys) * self.scale

        if mask is not None:
            
            mask = F.pad(mask.flatten(1), (1,0), value = True)
            assert mask.shape[-1] == dots.shape[-1], 'Mask has incorrect dimensions'
            mask = mask[:, None, :].expand(-1, l, -1)
            dots.masked_fill_(~mask, float('-inf'))

        attn = dots.softmax(dim = -1)
        out = torch.einsum('bhij,bhjd->bhid', attn, values)
        out = out.transpose(1, 2).contiguous().view(b,l,-1)
        return out


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, depth, n_heads, ff_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                #PreNormAttention(d_model, Attention(d_model, n_heads = n_heads)),
                #PreNormForward(d_model, FeedForward(d_model, ff_dim, dropout = dropout))
                PostNormAttention(d_model, Attention(d_model, n_heads = n_heads)),
                PostNormForward(d_model, FeedForward(d_model, ff_dim, dropout = dropout))
            ]))

    def forward(self, x, save_hidden=False):
        if save_hidden == True:
            hidden_list = []
            hidden_list.append(x)
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
                hidden_list.append(x)
            return hidden_list
        else:
            for attn, ff in self.layers:
                x = attn(x, x, x) + x
                x = ff(x) + x
            return x


class CrossTransformerEncoder(nn.Module):
    def __init__(self, d_model, depth, n_heads, ff_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                #PreNormAttention(d_model, Attention(d_model, n_heads = n_heads)),
                #PreNormForward(d_model, FeedForward(d_model, ff_dim, dropout = dropout))
                PostNormAttention(d_model, Attention(d_model, n_heads = n_heads)),
                PostNormForward(d_model, FeedForward(d_model, ff_dim, dropout = dropout))
            ]))

    def forward(self, target_x, source_x,):
        for attn, ff in self.layers:
            target_x_tmp = attn(target_x, source_x, source_x)
            target_x = target_x_tmp + target_x
            target_x = ff(target_x) + target_x
        return target_x


class Transformer(nn.Module):
    def __init__(self, num_frames, token_len, save_hidden, d_model, depth, n_heads, ff_dim, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.token_len = token_len
        self.save_hidden = save_hidden

        if token_len is not None:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_frames + token_len, d_model))
            self.extra_token = nn.Parameter(torch.zeros(1, token_len, d_model))
        else:
             self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, d_model))
             self.extra_token = None

        self.dropout = nn.Dropout(emb_dropout)

        self.encoder = TransformerEncoder(d_model, depth, n_heads, ff_dim, dropout)


    def forward(self, x):
        b, n, _ = x.shape

        if self.token_len is not None:
            extra_token = repeat(self.extra_token, '1 n d -> b n d', b = b)
            x = torch.cat((extra_token, x), dim=1)
            x = x + self.pos_embedding[:, :n+self.token_len]
        else:
            x = x + self.pos_embedding[:, :n]

        x = self.dropout(x)
        x = self.encoder(x, self.save_hidden)

        return x


class CrossTransformer(nn.Module):
    def __init__(self, *, source_num_frames, tgt_num_frames, d_model, depth, n_heads, ff_dim, dropout = 0., emb_dropout = 0.):
        super().__init__()

        self.pos_embedding_s = nn.Parameter(torch.randn(1, source_num_frames + 1, d_model))
        self.pos_embedding_t = nn.Parameter(torch.randn(1, tgt_num_frames + 1, d_model))
        self.extra_token = nn.Parameter(torch.zeros(1, 1, d_model))

        self.dropout = nn.Dropout(emb_dropout)

        self.CrossTransformerEncoder = CrossTransformerEncoder(d_model, depth, n_heads, dim_head, ff_dim, dropout)

    def forward(self, target_x, source_x):
        b, n_s, _ = source_x.shape
        b, n_t, _ = target_x.shape

        extra_token = repeat(self.extra_token, '1 1 d -> b 1 d', b = b)

        source_x = torch.cat((extra_token, source_x), dim=1)
        source_x = source_x + self.pos_embedding_s[:, : n_s+1]

        target_x = torch.cat((extra_token, target_x), dim=1)
        target_x = target_x + self.pos_embedding_t[:, : n_t+1]

        source_x = self.dropout(source_x)
        target_x = self.dropout(target_x)

        x_s2t = self.CrossTransformerEncoder(source_x, target_x)

        return x_s2t



class TemporalAlignment(nn.Module):
    def __init__(self, dim, heads=8, dim_head=32, dropout=0.):
        super().__init__()

        inner_dim = dim_head * heads
        project_out = not (dim == inner_dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_ta = nn.Linear(dim, inner_dim, bias=False)
        self.to_k_tv = nn.Linear(dim, inner_dim, bias=False)
        self.to_v_ta = nn.Linear(dim, inner_dim, bias=False)
        self.to_v_tv = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim, bias=True),  # output linear
            nn.Dropout(dropout)  # dropout 
        ) if project_out else nn.Identity()

        self.ffn = FeedForward(dim, dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, h_t, h_a, h_v):
        b, n, _, h = *h_t.shape, self.heads

        q = self.to_q(h_t)
        k_ta = self.to_k_ta(h_a)
        k_tv = self.to_k_tv(h_v)
        v_ta = self.to_v_ta(h_a)
        v_tv = self.to_v_tv(h_v)

        q, k_ta, k_tv, v_ta, v_tv = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k_ta, k_tv, v_ta, v_tv))

        dots_ta = einsum('b h i d, b h j d -> b h i j', q, k_ta) * self.scale

        attn_ta = self.attend(dots_ta)
        out_ta = einsum('b h i j, b h j d -> b h i d', attn_ta, v_ta)
        out_ta = rearrange(out_ta, 'b h n d -> b n (h d)')

        dots_tv = einsum('b h i d, b h j d -> b h i j', q, k_tv) * self.scale
        attn_tv = self.attend(dots_tv)
        out_tv = einsum('b h i j, b h j d -> b h i d', attn_tv, v_tv)
        out_tv = rearrange(out_tv, 'b h n d -> b n (h d)')

        out_tav = h_t + self.to_out(out_ta + out_tv)

        return out_tav


class Multi_CA(nn.Module):
    def __init__(self, dim, ff_dim, heads=8, dropout=0.):
        super().__init__()

        self.heads = heads
        dim_head = int(dim/heads)
        self.scale = dim_head ** -0.5

        self.softmax = nn.Softmax(dim=-1)

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k_t = nn.Linear(dim, dim, bias=False)
        self.to_k_a = nn.Linear(dim, dim, bias=False)
        self.to_k_v = nn.Linear(dim, dim, bias=False)
        self.to_v_t = nn.Linear(dim, dim, bias=False)
        self.to_v_a = nn.Linear(dim, dim, bias=False)
        self.to_v_v = nn.Linear(dim, dim, bias=False)

        self.ffn = FeedForward(dim, ff_dim, dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, query, h_t, h_a, h_v):
        
        b, n, _, h = *h_t.shape, self.heads

        q = self.to_q(query)
        k_t = self.to_k_t(h_t)
        k_a = self.to_k_a(h_a)
        k_v = self.to_k_v(h_v)
        v_t = self.to_v_t(h_t)
        v_a = self.to_v_a(h_a)
        v_v = self.to_v_v(h_v)

        q, k_t, k_a, k_v, v_t, v_a, v_v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k_t, k_a, k_v, v_t, v_a, v_v))

        dots_qt = einsum('b h i d, b h j d -> b h i j', q, k_t) * self.scale

        # attention
        attn_qt = self.softmax(dots_qt)
        out_qt = einsum('b h i j, b h j d -> b h i d', attn_qt, v_t)
        out_qt = rearrange(out_qt, 'b h n d -> b n (h d)')

        dots_qa = einsum('b h i d, b h j d -> b h i j', q, k_a) * self.scale
        attn_qa = self.softmax(dots_qa)
        out_qa = einsum('b h i j, b h j d -> b h i d', attn_qa, v_a)
        out_qa = rearrange(out_qa, 'b h n d -> b n (h d)')

        dots_qv = einsum('b h i d, b h j d -> b h i j', q, k_v) * self.scale
        attn_qv = self.softmax(dots_qv)
        out_qv = einsum('b h i j, b h j d -> b h i d', attn_qv, v_v)
        out_qv = rearrange(out_qv, 'b h n d -> b n (h d)')

        out_tav = query + out_qt + out_qa + out_qv
        out_tav = self.norm1(out_tav)

        out_tav = out_tav + self.ffn(out_tav)
        out_tav = self.norm2(out_tav)

        return out_tav


class HierarchicalBottleneckFusion(nn.Module):
    def __init__(self, d_model=128, n_heads=8, ff_dim=512, drop_out=0.5, depth=2):
        
        super(HierarchicalBottleneckFusion, self).__init__()
        assert depth >= 1, "depth must be >= 1"

        self.depth = depth
        self.initial_query_len = 8

        self.encoder_q = nn.ModuleList()
        self.encoder_q2tav = nn.ModuleList()
        self.encoder_t = nn.ModuleList()
        self.encoder_a = nn.ModuleList()
        self.encoder_v = nn.ModuleList()

        for i in range(depth):
            self.encoder_q.append(TransformerEncoder(d_model, 1, n_heads, ff_dim, drop_out))
            self.encoder_q2tav.append(Multi_CA(d_model, ff_dim, n_heads, drop_out))
            self.encoder_t.append(CrossTransformerEncoder(d_model, 1, n_heads, ff_dim, drop_out))
            self.encoder_a.append(CrossTransformerEncoder(d_model, 1, n_heads, ff_dim, drop_out))
            self.encoder_v.append(CrossTransformerEncoder(d_model, 1, n_heads, ff_dim, drop_out))

    def forward(self, x_m, x_t, x_a, x_v):
        
        m_t, m_a, m_v = x_t, x_a, x_v
        query = None
        keep = self.initial_query_len
        for level in range(self.depth):

            if level == 0:
                query = self.encoder_q[level](x_m)
            else:
                query = self.encoder_q[level](query)

            query = query[:, :keep]
            query = self.encoder_q2tav[level](query, m_t, m_a, m_v)

            m_t = self.encoder_t[level](m_t, query)
            m_a = self.encoder_a[level](m_a, query)
            m_v = self.encoder_v[level](m_v, query)

            keep = max(1, keep // 2)

        t_a_v = torch.cat((query[:, 0], m_t[:, 0], m_a[:, 0], m_v[:, 0]), dim=-1)

        return t_a_v