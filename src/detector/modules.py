import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from jaxtyping import Float, Int
from einops import rearrange

from torch.nn.modules.normalization import RMSNorm


class RoPELayer(nn.Module):
    """Rotary Positional Embedding for injecting positional information."""
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, seq_len):
        """Get RoPE frequencies"""
        t = torch.arange(seq_len, device=self.inv_freq.device).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        return freqs  # Shape: (seq_len, dim // 2)


class BinaryAttentionBias(nn.Module):
    """Binary Variate Attention for time series data.
    Used for Any-Variate Attention with flattening channel strategy aka PatchTST, TimeRCD"""
    def __init__(self, num_heads: Int):
        super().__init__()
        self.num_heads = num_heads
        self.emd = nn.Embedding(2, num_heads)

    def forward(self,
                query_id: Int[torch.Tensor, "batch_size q_len"],
                kv_id: Int[torch.Tensor, "batch_size kv_len"],
                ) -> Float[torch.Tensor, "batch_size num_heads q_len kv_len"]:
        ind = torch.eq(query_id.unsqueeze(-1), kv_id.unsqueeze(-2))
        ind = ind.unsqueeze(1)  # (batch_size, 1, q_len, kv_len)
        weight = rearrange(self.emd.weight, "two num_heads -> two num_heads 1 1")  # (2, num_heads, 1, 1)
        bias = ~ind * weight[:1] + ind * weight[1:]  # (batch_size, num_heads, q_len, kv_len)
        return bias


class MultiheadAttentionWithRoPE(nn.Module):
    """Multi-head Attention with Rotary Positional Encoding (RoPE), non-causal by default."""
    def __init__(self, embed_dim, num_heads, num_features):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.num_features = num_features
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections for Q, K, V, and output
        self.q_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        # Binary attention bias for time series
        if num_features > 1:
            self.binary_attention_bias = BinaryAttentionBias(num_heads)

    def apply_rope(self, x, rope_freqs):
        """Apply Rotary Positional Encoding to the input tensor."""
        B, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim, "Embedding dimension mismatch"
        assert rope_freqs.shape == (seq_len, embed_dim // 2), "rope_freqs shape mismatch"

        # Reshape for rotation: split embed_dim into pairs
        x_ = x.view(B, seq_len, embed_dim // 2, 2)
        cos = rope_freqs.cos().unsqueeze(0)  # (1, seq_len, embed_dim // 2, 1)
        sin = rope_freqs.sin().unsqueeze(0)  # (1, seq_len, embed_dim // 2, 1)

        # Apply rotation to each pair
        x_rot = torch.stack(
            [
                x_[..., 0] * cos - x_[..., 1] * sin,
                x_[..., 0] * sin + x_[..., 1] * cos,
            ],
            dim=-1
        )
        return x_rot.view(B, seq_len, embed_dim)

    def forward(self, query, key, value, rope_freqs, query_id=None, kv_id=None, attn_mask=None):
        """
        Forward pass for multi-head attention with RoPE.

        Args:
            query (Tensor): Shape (B, T, C)
            key (Tensor): Shape (B, T, C)
            value (Tensor): Shape (B, T, C)
            rope_freqs (Tensor): RoPE frequencies, shape (T, embed_dim // 2)
            query_id (Tensor, optional): Shape (B, q_len), feature IDs for query
            kv_id (Tensor, optional): Shape (B, kv_len), feature IDs for key/value
            attn_mask (Tensor, optional): Shape (B, T), True for valid positions, False for padding.

        Returns:
            Tensor: Attention output, shape (B, T, C)
        """
        B, T, C = query.shape
        assert key.shape == (B, T, C) and value.shape == (B, T, C), "query, key, value shapes must match"

        # Project inputs to Q, K, V
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # Apply RoPE to Q and K
        Q_rot = self.apply_rope(Q, rope_freqs)
        K_rot = self.apply_rope(K, rope_freqs)

        # Reshape for multi-head attention
        Q_rot = Q_rot.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        K_rot = K_rot.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)
        V = V.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # (B, nh, T, hs)

        # Prepare attention mask for padding
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        else:
            attn_mask = None

        if query_id is not None and kv_id is not None:
            # Add binary attention bias
            attn_bias = self.binary_attention_bias(query_id, kv_id)  # (B, num_heads, q_len, kv_len)
            scores = torch.matmul(Q_rot, K_rot.transpose(-2, -1)) / math.sqrt(
                self.head_dim)  # (B, num_heads, q_len, kv_len)
            scores += attn_bias
            if attn_mask is not None:
                scores = scores.masked_fill(~attn_mask, float('-inf'))
            attn_weights = F.softmax(scores, dim=-1)  # (B, num_heads, q_len, kv_len)
            y = torch.matmul(attn_weights, V)  # (B, num_heads, q_len, hs)

        else:
            # Compute scaled dot-product attention (non-causal) without binary bias
            # for param in self.binary_attention_bias.parameters():
            #     param.requires_grad = False
            y = F.scaled_dot_product_attention(
                Q_rot, K_rot, V,
                attn_mask=attn_mask,
                is_causal=False  # Non-causal attention for encoder
            )  # (B, nh, T, hs)

        # Reshape and project output
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.out_proj(y)
        return y


class LlamaMLP(nn.Module):
    def __init__(self, d_model, dim_feedforward=2048):
        super().__init__()
        self.hidden_size = d_model
        self.intermediate_size = dim_feedforward
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=True)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)

    def forward(self, x):
        up_proj = self.up_proj(x)
        gate_proj = F.gelu(self.gate_proj(x))
        down_proj = self.down_proj(gate_proj * up_proj)
        return down_proj


class TransformerEncoderLayerWithRoPE(nn.Module):
    """Transformer Encoder Layer with RoPE and RMSNorm."""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu", num_features=1):
        super().__init__()
        self.self_attn = MultiheadAttentionWithRoPE(d_model, nhead, num_features)
        self.dropout = nn.Dropout(dropout)
        self.input_norm = RMSNorm(d_model)
        self.output_norm = RMSNorm(d_model)
        self.mlp = LlamaMLP(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, src, rope_freqs, src_id=None, attn_mask=None):
        residual = src
        src = self.input_norm(src)
        src = self.self_attn(src, src, src, rope_freqs, src_id, src_id, attn_mask=attn_mask)
        src = src + residual
        residual = src
        src = self.output_norm(src)
        src = self.mlp(src)
        src = residual + self.dropout2(src)
        return src


class CustomTransformerEncoder(nn.Module):
    """Stack of Transformer Encoder Layers."""
    def __init__(self, d_model, nhead, dim_feedforward, dropout, activation, num_layers, num_features):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayerWithRoPE(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                activation=activation,
                num_features=num_features
            ) for _ in range(num_layers)
        ])

    def forward(self, src, rope_freqs, src_id=None, attn_mask=None):
        output = src
        for layer in self.layers:
            output = layer(output, rope_freqs, src_id, attn_mask=attn_mask)
        return output


class TimeSeriesEncoderBase(nn.Module):
    """
    Time Series Encoder Base class.

    Args:
        d_model (int): Model dimension
        d_proj (int): Projection dimension
        patch_size (int): Size of each patch
        num_layers (int): Number of encoder layers
        num_heads (int): Number of attention heads
        d_ff_dropout (float): Dropout rate
        max_total_tokens (int): Maximum sequence length
        use_rope (bool): Use RoPE if True
        num_features (int): Number of features in the time series
        activation (str): "relu" or "gelu"

    Inputs:
        time_series (Tensor): Shape (batch_size, seq_len, num_features)
        mask (Tensor): Shape (batch_size, seq_len)

    Outputs:
        local_embeddings (Tensor): Shape (batch_size, seq_len, num_features, d_proj)
    """

    def __init__(self, d_model=2048, d_proj=512, patch_size=32, num_layers=6, num_heads=8,
                 d_ff_dropout=0.1, max_total_tokens=8192, use_rope=True, num_features=1,
                 activation="relu"):
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.d_proj = d_proj
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff_dropout = d_ff_dropout
        self.max_total_tokens = max_total_tokens
        self.use_rope = use_rope
        self.num_features = num_features
        self.activation = activation

        # Patch embedding layer
        self.embedding_layer = nn.Linear(patch_size, d_model)

        if use_rope:
            # Initialize RoPE and custom encoder
            self.rope_embedder = RoPELayer(d_model)
            self.transformer_encoder = CustomTransformerEncoder(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=d_ff_dropout,
                activation=activation,
                num_layers=num_layers,
                num_features=num_features
            )
            self._init_parameters()
        else:
            # Standard encoder without RoPE
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_model * 4,
                dropout=d_ff_dropout,
                batch_first=True,
                activation=activation
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Output projection layers
        self.projection_layer = nn.Linear(d_model, patch_size * d_proj)
        
    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'weight' in name and 'linear' in name:
                if self.activation == "relu":
                    nn.init.kaiming_uniform_(param, nonlinearity='relu')
                elif self.activation == "gelu":
                    nn.init.kaiming_uniform_(param, nonlinearity='gelu')
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)

    def forward(self, time_series, mask):
        """Forward pass to generate local embeddings."""
        raise NotImplementedError


class TimeSeriesEncoderFlatten(TimeSeriesEncoderBase):
    """
    Time Series Encoder with PatchTST-like patching.
    Using channel-flattening strategy aka PatchTST, TimeRCD.

    Args:
        d_model (int): Model dimension
        d_proj (int): Projection dimension
        patch_size (int): Size of each patch
        num_layers (int): Number of encoder layers
        num_heads (int): Number of attention heads
        d_ff_dropout (float): Dropout rate
        max_total_tokens (int): Maximum sequence length
        use_rope (bool): Use RoPE if True
        num_features (int): Number of features in the time series
        activation (str): "relu" or "gelu"

    Inputs:
        time_series (Tensor): Shape (B, seq_len, num_features)
        mask (Tensor): Shape (B, seq_len)

    Outputs:
        local_embeddings (Tensor): Shape (B, seq_len, num_features, d_proj)
    """
    def forward(self, time_series, mask):
        """Forward pass to generate local embeddings."""
        if time_series.dim() == 2:
            time_series = time_series.unsqueeze(-1)

        device = time_series.device
        B, seq_len, num_features = time_series.size()
        assert num_features == self.num_features, f"Number of features mismatch with data: {num_features} vs param: {self.num_features}"
        assert mask.size() == (B, seq_len), "Mask shape mismatch"

        # Pad sequence to be divisible by patch_size
        padded_length = math.ceil(seq_len / self.patch_size) * self.patch_size
        if padded_length > seq_len:
            pad_amount = padded_length - seq_len
            time_series = F.pad(time_series, (0, 0, 0, pad_amount), value=0)
            mask = F.pad(mask, (0, pad_amount), value=0)

        '''
        =============================== Global IDEA ==============================
        (B, seq_len, num_features) -> (B, num_features * num_patches, patch_size)
        This flattening approach allows to process channels together in every patch
        '''

        # Variate-Window Tokenization

        ## Patching by time dimension -- Window Tokenization
        num_patches = padded_length // self.patch_size
        # (B, seq_len, num_features) -> (B, num_patches, patch_size, num_features)
        patches = time_series.view(B, num_patches, self.patch_size, num_features)

        ## Variate Tokenization -- flatten num_features & num_patches
        patches = patches.permute(0, 3, 1, 2).contiguous()  # (B, num_features, num_patches, patch_size)
        patches = patches.view(B, num_features * num_patches, self.patch_size)  # (B, num_features * num_patches, patch_size)

        ## Embed patches (last dim)
        # (B, num_features * num_patches, patch_size) -> (B, num_features * num_patches, d_model)
        embedded_patches = self.embedding_layer(patches)

        ## Create patch-level mask
        # (B, seq_len) -> (B, num_patches, patch_size)
        mask = mask.view(B, num_patches, self.patch_size)
        # mask full patch as an element
        patch_mask = mask.sum(dim=-1) > 0  # (B, num_patches)

        ## Convert to (B, num_features * num_patches)
        full_mask = patch_mask.unsqueeze(1).expand(-1, num_features, -1)  # (B, num_features, num_patches)
        full_mask = full_mask.reshape(B, num_features * num_patches)  # (B, num_features * num_patches)

        ## Generate RoPE frequencies
        if self.use_rope:
            total_length = num_patches * num_features
            rope_freqs = self.rope_embedder(total_length).to(device)
        else:
            rope_freqs = None

        # Encode sequence
        if num_features > 1:
            # Using Any-Variate Attention
            ## Feature ID for channel identification (aka PatchTST)
            feature_id = torch.arange(num_features, device=device).repeat_interleave(num_patches)
            feature_id = feature_id.unsqueeze(0).expand(B, -1)  # (B, num_features * num_patches)
        else:
            feature_id = None

        output = self.transformer_encoder(
            embedded_patches,
            rope_freqs=rope_freqs,
            src_id=feature_id,
            attn_mask=full_mask
        )  # (B, num_features * num_patches, d_model)

        # Extract and project local embeddings
        patch_proj = self.projection_layer(output)  # (B, num_features * num_patches, patch_size * d_proj)

        # Reverse reshape
        local_embeddings = patch_proj.view(B, num_features, num_patches, self.patch_size, self.d_proj) # unfold
        local_embeddings = local_embeddings.permute(0, 2, 3, 1, 4)  # (B, num_patches, patch_size, num_features, d_proj)
        local_embeddings = local_embeddings.view(B, -1, num_features, self.d_proj)[:, :seq_len, :, :]  # (B, seq_len, num_features, d_proj)

        return local_embeddings


class TimeSeriesEncoderIndependent(TimeSeriesEncoderBase):
    """
    Time Series Encoder using channel-independent strategy aka MOMENT

    Args:
        d_model (int): Model dimension
        d_proj (int): Projection dimension
        patch_size (int): Size of each patch
        num_layers (int): Number of encoder layers
        num_heads (int): Number of attention heads
        d_ff_dropout (float): Dropout rate
        max_total_tokens (int): Maximum sequence length
        use_rope (bool): Use RoPE if True
        num_features (int): Number of features in the time series
        activation (str): "relu" or "gelu"

    Inputs:
        time_series (Tensor): Shape (batch_size, seq_len, num_features)
        mask (Tensor): Shape (batch_size, seq_len)

    Outputs:
        local_embeddings (Tensor): Shape (batch_size, seq_len, num_features, d_proj)
    """
    def forward(self, time_series, mask):
        """Forward pass to generate local embeddings."""
        if time_series.dim() == 2:
            time_series = time_series.unsqueeze(-1)

        device = time_series.device
        B, seq_len, num_features = time_series.size()
        assert num_features == self.num_features, f"Number of features mismatch with data: {num_features} vs param: {self.num_features}"
        assert mask.size() == (B, seq_len), "Mask shape mismatch"

        # Pad sequence to be divisible by patch_size
        padded_length = math.ceil(seq_len / self.patch_size) * self.patch_size
        if padded_length > seq_len:
            pad_amount = padded_length - seq_len
            time_series = F.pad(time_series, (0, 0, 0, pad_amount), value=0)
            mask = F.pad(mask, (0, pad_amount), value=0)

        '''
        =============================== Global IDEA ==============================
        (B, seq_len, num_features) -> (B * num_features, num_patches, patch_size)
        This allows to process every variate independent and vectorized
        '''

        # Reshape to process **every channel as a batch**
        time_series_ci = time_series.permute(0, 2, 1)  # (B, num_features, seq_len)
        time_series_ci = time_series_ci.reshape(B * num_features, -1)  # (B * num_features, seq_len)

        mask_ci = mask.unsqueeze(1).expand(-1, num_features, -1)  # (B, num_features, seq_len)
        mask_ci = mask_ci.reshape(B * num_features, seq_len)  # (B * num_features, seq_len)

        # Variate-Window Independent Tokenization

        ## Patching by time dimension
        num_patches = padded_length // self.patch_size
        patches = time_series_ci.view(B * num_features, num_patches, self.patch_size)  # (B * num_features, num_patches, patch_size)

        ## Patch-level mask
        mask_patches = mask_ci.view(B * num_features, num_patches, self.patch_size)  # (B * num_features, num_patches, patch_size)
        # mask full patch as an element
        patch_mask = mask_patches.sum(dim=-1) > 0  # (B * num_features, num_patches)

        ## Embed patches (last dim)
        # (B * num_features, num_patches, patch_size) -> (B * num_features, num_patches, d_model)
        embedded_patches = self.embedding_layer(patches)

        ## Generate RoPE frequencies
        if self.use_rope:
            rope_freqs = self.rope_embedder(num_patches).to(device)
        else:
            rope_freqs = None

        # Encode sequence
        output = self.transformer_encoder(
            embedded_patches,
            rope_freqs=rope_freqs,
            attn_mask=patch_mask
        ) # (B * num_features, num_patches, d_model)

        # Extract and project local embeddings
        patch_proj = self.projection_layer(output)  # (B * num_features, num_patches, patch_size * d_proj)

        # Reverse reshape
        local_embeddings = patch_proj.view(B, num_features, num_patches, self.patch_size, self.d_proj) # unfold
        local_embeddings = local_embeddings.permute(0, 2, 3, 1, 4)  # (B, num_patches, patch_size, num_features, d_proj)
        local_embeddings = local_embeddings.view(B, -1, num_features, self.d_proj)[:, :seq_len, :, :]  # (B, seq_len, num_features, d_proj)

        return local_embeddings


class TimeSeriesEncoderMixing(TimeSeriesEncoderBase):
    """
    Time Series Encoder using channel-mixing strategy

    Args:
        d_model (int): Model dimension
        d_proj (int): Projection dimension
        patch_size (int): Size of each patch
        num_layers (int): Number of encoder layers
        num_heads (int): Number of attention heads
        d_ff_dropout (float): Dropout rate
        max_total_tokens (int): Maximum sequence length
        use_rope (bool): Use RoPE if True
        num_features (int): Number of features in the time series
        activation (str): "relu" or "gelu"

    Inputs:
        time_series (Tensor): Shape (batch_size, seq_len, num_features)
        mask (Tensor): Shape (batch_size, seq_len)

    Outputs:
        local_embeddings (Tensor): Shape (batch_size, seq_len, num_features, d_proj)
    """
    def __init__(self, d_model=2048, d_proj=512, patch_size=32, num_layers=6, num_heads=8,
                 d_ff_dropout=0.1, max_total_tokens=8192, use_rope=True, num_features=1,
                 activation="relu"):
        super().__init__(d_model, d_proj, patch_size, num_layers, num_heads,
                 d_ff_dropout, max_total_tokens, use_rope, num_features,
                 activation)

        # Patch embedding layer
        '''
        Note: embed every patch with channels -> mixing mechanism
        Intput: patch_size * num_features shape, not only patch_size!
        '''
        self.embedding_layer = nn.Linear(patch_size * num_features, d_model)
        self.channel_embedding = nn.Embedding(num_features, d_proj)

    def forward(self, time_series, mask):
        """Forward pass to generate local embeddings."""
        if time_series.dim() == 2:
            time_series = time_series.unsqueeze(-1)

        device = time_series.device
        B, seq_len, num_features = time_series.size()
        assert num_features == self.num_features, f"Number of features mismatch with data: {num_features} vs param: {self.num_features}"
        assert mask.size() == (B, seq_len), "Mask shape mismatch"

        # Pad sequence to be divisible by patch_size
        padded_length = math.ceil(seq_len / self.patch_size) * self.patch_size
        if padded_length > seq_len:
            pad_amount = padded_length - seq_len
            time_series = F.pad(time_series, (0, 0, 0, pad_amount), value=0)
            mask = F.pad(mask, (0, pad_amount), value=0)

        '''
        =============================== Global IDEA ==============================
        (B, seq_len, num_features) -> (B, num_patches, patch_size * num_features)
        Then all channels will be mixed into one embedding
        This mixing approach allows to process channels together in every patch
        '''

        # Variate-Window Mixing Tokenization

        ## Patching by time dimension -- Window Tokenization
        num_patches = padded_length // self.patch_size
        patches = time_series.view(B, num_patches, self.patch_size, num_features)  # (B, num_patches, patch_size, num_features)

        ## Variate Tokenization -- patch_size & num_features into token
        patches = patches.view(B, num_patches, self.patch_size * num_features)  # (B, num_patches, patch_size * num_features)

        ## Embed patches (last dim)
        # (B, num_patches, patch_size * num_features) -> (B, num_patches, d_model)
        embedded_patches = self.embedding_layer(patches)

        ## Patch-level mask
        # (B, seq_len) -> (B, num_patches, patch_size)
        mask_patches = mask.view(B, num_patches, self.patch_size)
        # mask full patch as an element
        patch_mask = mask_patches.sum(dim=-1) > 0  # (B, num_patches)

        if self.use_rope:
            # Every token -- patch with all channels
            total_length = num_patches
            rope_freqs = self.rope_embedder(total_length).to(device)
        else:
            rope_freqs = None

        # Encode sequence
        output = self.transformer_encoder(
            embedded_patches,
            rope_freqs=rope_freqs,
            attn_mask=patch_mask
        )  # (B, num_patches, d_model)

        # Extract and project local embeddings
        patch_proj = self.projection_layer(output)  # (B, num_patches, patch_size * d_proj)

        # Reverse reshape
        local_embeddings = patch_proj.view(B, num_patches, self.patch_size, self.d_proj)  # unfold
        # Repeat embedding for every channel (equal for every channel)
        local_embeddings = local_embeddings.unsqueeze(3)  # (B, num_patches, patch_size, 1, d_proj)
        local_embeddings = local_embeddings.expand(-1, -1, -1, num_features, -1)
        local_embeddings = local_embeddings.view(B, -1, num_features, self.d_proj)[:, :seq_len, :, :]  # (B, seq_len, num_features, d_proj)

        # Additive channel embedding (как positional encoding)
        channel_embeds = self.channel_embedding(
            torch.arange(num_features, device=local_embeddings.device)
        )  # (num_features, d_proj)
        channel_embeds = channel_embeds.unsqueeze(0).unsqueeze(0)  # (1, 1, num_features, d_proj)
        
        return local_embeddings + channel_embeds  # (B, seq_len, num_features, d_proj)
