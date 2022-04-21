from typing import Optional, List
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


# Wraps the input tuple for a function to process a time x batch x features sequence in batch x features (assumes one output)
def bottle(f, x_tuple):
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
    y_size = y.size()
    return y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])


# Wraps the input tuple for a function to process a time x batch x chunk x features sequence in batch x features (assumes one output)
def bottle3(f, x_tuple):
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1] * x[1][2], *x[1][3:]), zip(x_tuple, x_sizes)))
    y_size = y.size()
    return y.view(x_sizes[0][0], x_sizes[0][1], x_sizes[0][2], *y_size[1:])


class TransitionModel(nn.Module):
    __constants__ = ['min_std_dev']

    def __init__(self, belief_size, state_size, action_size, hidden_size, embedding_size, activation_function='relu',
                 min_std_dev=0.1):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.fc_embed_state_action = nn.Linear(state_size + action_size, belief_size)
        self.rnn = nn.GRUCell(belief_size, belief_size)
        self.fc_embed_belief_prior = nn.Linear(belief_size, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
        self.fc_embed_belief_posterior = nn.Linear(belief_size + embedding_size, hidden_size)
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)

    # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
    # t :  0  1  2  3  4  5
    # o :    -X--X--X--X--X-
    # a : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -x--X--X--X--X--X-
    # s : -x--X--X--X--X--X-
    def forward(self, prev_state: torch.Tensor, actions: torch.Tensor, prev_belief: torch.Tensor,
                observations: Optional[torch.Tensor] = None, nonterminals: Optional[torch.Tensor] = None) -> List[torch.Tensor]:
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        T = actions.size(0) + 1
        beliefs, prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = [torch.empty(0)] * T, \
                                                                                                                    [torch.empty(0)] * T, \
                                                                                                                    [torch.empty(0)] * T, \
                                                                                                                    [torch.empty(0)] * T, \
                                                                                                                    [torch.empty(0)] * T, \
                                                                                                                    [torch.empty(0)] * T, \
                                                                                                                    [torch.empty(0)] * T
        beliefs[0], prior_states[0], posterior_states[0] = prev_belief, prev_state, prev_state
        # Loop over time sequence
        for t in range(T - 1):
            _state = prior_states[t] if observations is None else posterior_states[t]  # Select appropriate previous state
            _state = _state if nonterminals is None else _state * nonterminals[t]  # Mask if previous transition was terminal
            # Compute belief (deterministic hidden state)
            hidden = self.act_fn(self.fc_embed_state_action(torch.cat([_state, actions[t]], dim=1)))
            beliefs[t + 1] = self.rnn(hidden, beliefs[t])
            # Compute state prior by applying transition dynamics
            hidden = self.act_fn(self.fc_embed_belief_prior(beliefs[t + 1]))
            prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(hidden), 2, dim=1)
            prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])
            if observations is not None:
                # Compute state posterior by applying transition dynamics and using current observation
                t_ = t - 1  # Use t_ to deal with different time indexing for observations
                hidden = self.act_fn(
                    self.fc_embed_belief_posterior(torch.cat([beliefs[t + 1], observations[t_ + 1]], dim=1)))
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
                posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(
                    posterior_means[t + 1])
        # Return new hidden states
        hidden = [torch.stack(beliefs[1:], dim=0), torch.stack(prior_states[1:], dim=0),
                  torch.stack(prior_means[1:], dim=0), torch.stack(prior_std_devs[1:], dim=0)]
        if observations is not None:
            hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0),
                       torch.stack(posterior_std_devs[1:], dim=0)]
        return hidden


class SymbolicObservationModel(nn.Module):
    def __init__(self, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, observation_size)

    def forward(self, belief, state):
        hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        observation = self.fc3(hidden)
        return observation


class VisualObservationModel(nn.Module):
    __constants__ = ['embedding_size', 'image_dim']

    def __init__(self, belief_size, state_size, embedding_size, image_dim, activation_function='relu'):
        super().__init__()
        self.image_dim = image_dim
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        # Spatial broadcast decoder
        self.fc1 = nn.Linear(belief_size + state_size, embedding_size)
        self.conv1 = nn.Conv2d(embedding_size+2, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=1)

    def init_grid(self):
        x = torch.linspace(-3, 3, self.image_dim)
        y = torch.linspace(-3, 3, self.image_dim)
        self.x_grid, self.y_grid = torch.meshgrid(x, y)
    
    def broadcast(self, embedded_obs):
        x_grid = self.x_grid.expand(embedded_obs.size(0),1,-1,-1).to(embedded_obs.device)
        y_grid = self.y_grid.expand(embedded_obs.size(0),1,-1,-1).to(embedded_obs.device)
        embedded_obs_widen = embedded_obs.view((embedded_obs.size(0),-1,1,1)).expand(-1,-1,self.image_dim,self.image_dim)
        embedded_obs_widen = torch.cat([embedded_obs_widen, x_grid, y_grid], dim=1)
        return embedded_obs_widen


    def forward(self, belief, state):
        self.init_grid()
        hidden = self.fc1(torch.cat([belief, state], dim=1))  # No nonlinearity here
        hidden = self.broadcast(hidden)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        observation = self.conv3(hidden)
        return observation


def ObservationModel(symbolic, observation_size, belief_size, state_size, embedding_size, activation_function='relu'):
    if symbolic:
        return SymbolicObservationModel(observation_size, belief_size, state_size, embedding_size, activation_function)
    else:
        return VisualObservationModel(belief_size, state_size, embedding_size, observation_size[1], activation_function)


class RewardModel(nn.Module):
    def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, belief, state):
        hidden = self.act_fn(self.fc1(torch.cat([belief, state], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        reward = self.fc3(hidden).squeeze(dim=1)
        return reward


class SymbolicEncoder(nn.Module):
    def __init__(self, observation_size, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(observation_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, embedding_size)

    def forward(self, observation):
        hidden = self.act_fn(self.fc1(observation))
        hidden = self.act_fn(self.fc2(hidden))
        hidden = self.fc3(hidden)
        return hidden


class ValueModel(nn.Module):
    def __init__(self, belief_size, state_size, hidden_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.fc1 = nn.Linear(belief_size + state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, belief, state):
        hidden = self.act_fn(self.fc1(torch.cat([belief.detach(), state.detach()], dim=1)))
        hidden = self.act_fn(self.fc2(hidden))
        value = self.fc3(hidden).squeeze(dim=1)
        return value

def shift_dim(x, src_dim=-1, dest_dim=-1, make_contiguous=True):
    n_dims = len(x.shape)
    if src_dim < 0:
        src_dim = n_dims + src_dim
    if dest_dim < 0:
        dest_dim = n_dims + dest_dim

    assert 0 <= src_dim < n_dims and 0 <= dest_dim < n_dims

    dims = list(range(n_dims))
    del dims[src_dim]

    permutation = []
    ctr = 0
    for i in range(n_dims):
        if i == dest_dim:
            permutation.append(src_dim)
        else:
            permutation.append(dims[ctr])
            ctr += 1
    x = x.permute(permutation)
    if make_contiguous:
        x = x.contiguous()
    return x

# reshapes tensor start from dim i (inclusive)
# to dim j (exclusive) to the desired shape
# e.g. if x.shape = (b, thw, c) then
# view_range(x, 1, 2, (t, h, w)) returns
# x of shape (b, t, h, w, c)
def view_range(x, i, j, shape):
    shape = tuple(shape)

    n_dims = len(x.shape)
    if i < 0:
        i = n_dims + i

    if j is None:
        j = n_dims
    elif j < 0:
        j = n_dims + j

    assert 0 <= i < j <= n_dims

    x_shape = x.shape
    target_shape = x_shape[:i] + shape + x_shape[j:]
    return x.view(target_shape)


class MultiHeadAttention(nn.Module):
    def __init__(self, shape, dim_q, dim_kv, n_head, n_layer,
                 causal, attn_type, attn_kwargs):
        super().__init__()
        self.causal = causal
        self.shape = shape

        self.d_k = dim_q // n_head
        self.d_v = dim_kv // n_head
        self.n_head = n_head

        self.w_qs = nn.Linear(dim_q, n_head * self.d_k, bias=False) # q
        self.w_qs.weight.data.normal_(std=1.0 / np.sqrt(dim_q))

        self.w_ks = nn.Linear(dim_kv, n_head * self.d_k, bias=False) # k
        self.w_ks.weight.data.normal_(std=1.0 / np.sqrt(dim_kv))

        self.w_vs = nn.Linear(dim_kv, n_head * self.d_v, bias=False) # v
        self.w_vs.weight.data.normal_(std=1.0 / np.sqrt(dim_kv))

        self.fc = nn.Linear(n_head * self.d_v, dim_q, bias=True) # c
        self.fc.weight.data.normal_(std=1.0 / np.sqrt(dim_q * n_layer))

        if attn_type == 'full':
            self.attn = FullAttention(shape, causal, **attn_kwargs)
        elif attn_type == 'axial':
            assert not causal, 'causal axial attention is not supported'
            self.attn = AxialAttention(len(shape), **attn_kwargs)
        elif attn_type == 'sparse':
            self.attn = SparseAttention(shape, n_head, causal, **attn_kwargs)

        self.cache = None

    def forward(self, q, k, v, decode_step=None, decode_idx=None):
        """ Compute multi-head attention
        Args
            q, k, v: a [b, d1, ..., dn, c] tensor or
                     a [b, 1, ..., 1, c] tensor if decode_step is not None
        Returns
            The output after performing attention
        """

        # compute k, q, v
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        q = view_range(self.w_qs(q), -1, None, (n_head, d_k))
        k = view_range(self.w_ks(k), -1, None, (n_head, d_k))
        v = view_range(self.w_vs(v), -1, None, (n_head, d_v))

        # b x n_head x seq_len x d
        # (b, *d_shape, n_head, d) -> (b, n_head, *d_shape, d)
        q = shift_dim(q, -2, 1)
        k = shift_dim(k, -2, 1)
        v = shift_dim(v, -2, 1)

        # fast decoding
        if decode_step is not None:
            if decode_step == 0:
                if self.causal:
                    k_shape = (q.shape[0], n_head, *self.shape, self.d_k)
                    v_shape = (q.shape[0], n_head, *self.shape, self.d_v)
                    self.cache = dict(k=torch.zeros(k_shape, dtype=k.dtype, device=q.device),
                                    v=torch.zeros(v_shape, dtype=v.dtype, device=q.device))
                else:
                    # cache only once in the non-causal case
                    self.cache = dict(k=k.clone(), v=v.clone())
            if self.causal:
                idx = (slice(None, None), slice(None, None), *[slice(i, i+ 1) for i in decode_idx])
                self.cache['k'][idx] = k
                self.cache['v'][idx] = v
            k, v = self.cache['k'], self.cache['v']

        a = self.attn(q, k, v, decode_step, decode_idx)

        # (b, *d_shape, n_head, d) -> (b, *d_shape, n_head * d)
        a = shift_dim(a, 1, -2).flatten(start_dim=-2)
        a = self.fc(a) # (b x seq_len x embd_dim)

        return a

############## Attention #######################
class FullAttention(nn.Module):
    def __init__(self, shape, causal, attn_dropout):
        super().__init__()
        self.causal = causal
        self.attn_dropout = attn_dropout

        seq_len = np.prod(shape)
        if self.causal:
            self.register_buffer('mask', torch.tril(torch.ones(seq_len, seq_len)))

    def forward(self, q, k, v, decode_step, decode_idx):
        mask = self.mask if self.causal else None
        if decode_step is not None and mask is not None:
            mask = mask[[decode_step]]

        old_shape = q.shape[2:-1]
        q = q.flatten(start_dim=2, end_dim=-2)
        k = k.flatten(start_dim=2, end_dim=-2)
        v = v.flatten(start_dim=2, end_dim=-2)

        out = scaled_dot_product_attention(q, k, v, mask=mask,
                                           attn_dropout=self.attn_dropout,
                                           training=self.training)

        return view_range(out, 2, 3, old_shape)

class AxialAttention(nn.Module):
    def __init__(self, n_dim, axial_dim):
        super().__init__()
        if axial_dim < 0:
            axial_dim = 2 + n_dim + 1 + axial_dim
        else:
            axial_dim += 2 # account for batch, head, dim
        self.axial_dim = axial_dim

    def forward(self, q, k, v, decode_step, decode_idx):
        q = shift_dim(q, self.axial_dim, -2).flatten(end_dim=-3)
        k = shift_dim(k, self.axial_dim, -2).flatten(end_dim=-3)
        v = shift_dim(v, self.axial_dim, -2)
        old_shape = list(v.shape)
        v = v.flatten(end_dim=-3)

        out = scaled_dot_product_attention(q, k, v, training=self.training)
        out = out.view(*old_shape)
        out = shift_dim(out, -2, self.axial_dim)
        return out


class SparseAttention(nn.Module):
    ops = dict()
    attn_mask = dict()
    block_layout = dict()

    def __init__(self, shape, n_head, causal, num_local_blocks=4, block=32,
                 attn_dropout=0.): # does not use attn_dropout
        super().__init__()
        self.causal = causal
        self.shape = shape

        self.sparsity_config = StridedSparsityConfig(shape=shape, n_head=n_head,
                                                     causal=causal, block=block,
                                                     num_local_blocks=num_local_blocks)

        if self.shape not in SparseAttention.block_layout:
            SparseAttention.block_layout[self.shape] = self.sparsity_config.make_layout()
        if causal and self.shape not in SparseAttention.attn_mask:
            SparseAttention.attn_mask[self.shape] = self.sparsity_config.make_sparse_attn_mask()

def scaled_dot_product_attention(q, k, v, mask=None, attn_dropout=0., training=True):
    # Performs scaled dot-product attention over the second to last dimension dn

    # (b, n_head, d1, ..., dn, d)
    attn = torch.matmul(q, k.transpose(-1, -2))
    attn = attn / np.sqrt(q.shape[-1])
    if mask is not None:
        attn = attn.masked_fill(mask == 0, float('-inf'))
    attn_float = F.softmax(attn, dim=-1)
    attn = attn_float.type_as(attn) # b x n_head x d1 x ... x dn x d
    attn = F.dropout(attn, p=attn_dropout, training=training)

    a = torch.matmul(attn, v) # b x n_head x d1 x ... x dn x d

    return a
    
class StridedSparsityConfig(object):
    """
    Strided Sparse configuration specified in https://arxiv.org/abs/1904.10509 that
    generalizes to arbitrary dimensions
    """
    def __init__(self, shape, n_head, causal, block, num_local_blocks):
        self.n_head = n_head
        self.shape = shape
        self.causal = causal
        self.block = block
        self.num_local_blocks = num_local_blocks

        assert self.num_local_blocks >= 1, 'Must have at least 1 local block'
        assert self.seq_len % self.block == 0, 'seq len must be divisible by block size'

        self._block_shape = self._compute_block_shape()
        self._block_shape_cum = self._block_shape_cum_sizes()

    @property
    def seq_len(self):
        return np.prod(self.shape)

    @property
    def num_blocks(self):
        return self.seq_len // self.block

    def set_local_layout(self, layout):
        num_blocks = self.num_blocks
        for row in range(0, num_blocks):
            end = min(row + self.num_local_blocks, num_blocks)
            for col in range(
                    max(0, row - self.num_local_blocks),
                    (row + 1 if self.causal else end)):
                layout[:, row, col] = 1
        return layout

    def set_global_layout(self, layout):
        num_blocks = self.num_blocks
        n_dim = len(self._block_shape)
        for row in range(num_blocks):
            assert self._to_flattened_idx(self._to_unflattened_idx(row)) == row
            cur_idx = self._to_unflattened_idx(row)
            # no strided attention over last dim
            for d in range(n_dim - 1):
                end = self._block_shape[d]
                for i in range(0, (cur_idx[d] + 1 if self.causal else end)):
                    new_idx = list(cur_idx)
                    new_idx[d] = i
                    new_idx = tuple(new_idx)

                    col = self._to_flattened_idx(new_idx)
                    layout[:, row, col] = 1

        return layout

    def make_layout(self):
        layout = torch.zeros((self.n_head, self.num_blocks, self.num_blocks), dtype=torch.int64)
        layout = self.set_local_layout(layout)
        layout = self.set_global_layout(layout)
        return layout

    def make_sparse_attn_mask(self):
        block_layout = self.make_layout()
        assert block_layout.shape[1] == block_layout.shape[2] == self.num_blocks

        num_dense_blocks = block_layout.sum().item()
        attn_mask = torch.ones(num_dense_blocks, self.block, self.block)
        counter = 0
        for h in range(self.n_head):
            for i in range(self.num_blocks):
                for j in range(self.num_blocks):
                    elem = block_layout[h, i, j].item()
                    if elem == 1:
                        assert i >= j
                        if i == j: # need to mask within block on diagonals
                            attn_mask[counter] = torch.tril(attn_mask[counter])
                        counter += 1
        assert counter == num_dense_blocks

        return attn_mask.unsqueeze(0)

    def get_non_block_layout_row(self, block_layout, row):
        block_row = row // self.block
        block_row = block_layout[:, [block_row]] # n_head x 1 x n_blocks
        block_row = block_row.repeat_interleave(self.block, dim=-1)
        block_row[:, :, row + 1:] = 0.
        return block_row

    ############# Helper functions ##########################

    def _compute_block_shape(self):
        n_dim = len(self.shape)
        cum_prod = 1
        for i in range(n_dim - 1, -1, -1):
            cum_prod *= self.shape[i]
            if cum_prod > self.block:
                break
        assert cum_prod % self.block == 0
        new_shape = (*self.shape[:i], cum_prod // self.block)

        assert np.prod(new_shape) == np.prod(self.shape) // self.block

        return new_shape

    def _block_shape_cum_sizes(self):
        bs = np.flip(np.array(self._block_shape))
        return tuple(np.flip(np.cumprod(bs)[:-1])) + (1,)

    def _to_flattened_idx(self, idx):
        assert len(idx) == len(self._block_shape), f"{len(idx)} != {len(self._block_shape)}"
        flat_idx = 0
        for i in range(len(self._block_shape)):
            flat_idx += idx[i] * self._block_shape_cum[i]
        return flat_idx

    def _to_unflattened_idx(self, flat_idx):
        assert flat_idx < np.prod(self._block_shape)
        idx = []
        for i in range(len(self._block_shape)):
            idx.append(flat_idx // self._block_shape_cum[i])
            flat_idx %= self._block_shape_cum[i]
        return tuple(idx)


class AxialBlock(nn.Module):
    def __init__(self, n_hiddens, n_head):
        super().__init__()
        kwargs = dict(shape=(0,) * 3, dim_q=n_hiddens,
                      dim_kv=n_hiddens, n_head=n_head,
                      n_layer=1, causal=False, attn_type='axial')
        self.attn_w = MultiHeadAttention(attn_kwargs=dict(axial_dim=-2),
                                         **kwargs)
        self.attn_h = MultiHeadAttention(attn_kwargs=dict(axial_dim=-3),
                                         **kwargs)
        self.attn_t = MultiHeadAttention(attn_kwargs=dict(axial_dim=-4),
                                         **kwargs)

    def forward(self, x):
        x = shift_dim(x, 1, -1)
        x = self.attn_w(x, x, x) + self.attn_h(x, x, x) + self.attn_t(x, x, x)
        x = shift_dim(x, -1, 1)
        return x

class AttentionResidualBlock(nn.Module):
    def __init__(self, n_hiddens):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(n_hiddens),
            nn.ReLU(),
            nn.Conv2d(n_hiddens, n_hiddens // 2, 3, 1, 1),
            nn.BatchNorm2d(n_hiddens // 2),
            nn.ReLU(),
            nn.Conv2d(n_hiddens // 2, n_hiddens, 3, 1, 1),
            nn.BatchNorm2d(n_hiddens),
            nn.ReLU(),
            AxialBlock(n_hiddens, 2)
        )

    def forward(self, x):
        return x + self.block(x)


class VisualEncoder(nn.Module):
    __constants__ = ['embedding_size', 'image_dim']

    def __init__(self, embedding_size, image_dim, activation_function='relu'):
        super().__init__()
        self.image_dim = image_dim
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        n_hiddens = 256
        n_res_layers = 4
        if image_dim == 128:
            self.conv0 = nn.Conv2d(3, 16, 4, stride=2)
            self.conv1 = nn.Conv2d(16, 32, 4, stride=2)
        elif image_dim == 64:
            self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        else:
            raise NotImplementedError
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.res_stack = nn.Sequential(
            *[AttentionResidualBlock(n_hiddens)
              for _ in range(n_res_layers)],
            nn.BatchNorm2d(n_hiddens),
            nn.ReLU()
        )
        self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)

    def forward(self, observation):
        if self.image_dim == 128:
            hidden = self.act_fn(self.conv0(observation))
            hidden = self.act_fn(self.conv1(hidden))
        elif self.image_dim == 64:
            hidden = self.act_fn(self.conv1(observation))
        else:
            raise NotImplementedError
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = self.act_fn(self.res_stack(hidden))
        hidden = hidden.view(-1, 1024)
        hidden = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
        return hidden


def Encoder(symbolic, observation_size, embedding_size, activation_function='relu'):
    if symbolic:
        return SymbolicEncoder(observation_size, embedding_size, activation_function)
    else:
        return VisualEncoder(embedding_size, observation_size[1], activation_function)
