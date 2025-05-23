import torch
import torch.nn as nn
from torch.nn import functional as F

from .config import TextConfig
from .layers import QuantizedLinear, layer_norm, mlp
from .rope import apply_rotary_emb, precompute_freqs_cis


def text_encoder(input_ids: torch.Tensor, w: nn.Module):
    """
    Encode input token IDs into embeddings using learned word token embeddings.
    
    This is the first step in processing text tokens, converting discrete token IDs
    into dense vector representations that the model can work with.
    
    Args:
        input_ids: Tensor of shape (batch_size, seq_len) containing token IDs
        w: Module containing the word token embeddings (wte parameter)
        
    Returns:
        Tensor of shape (batch_size, seq_len, d_model) containing token embeddings
    """
    return F.embedding(input_ids, w.wte)


def attn(
    x: torch.Tensor,
    w: nn.Module,
    freqs_cis: torch.Tensor,
    kv_cache: nn.Module,
    attn_mask: torch.Tensor,
    n_heads: int,
    n_kv_heads: int,
    position_ids: torch.Tensor,
):
    """
    Multi-head attention with KV caching and grouped query attention (GQA) support.
    
    This implements the core attention mechanism used in transformer models, with several
    optimizations:
    - KV caching for efficient autoregressive generation
    - Grouped Query Attention (GQA) where key/value heads can be fewer than query heads
    - Rotary positional embeddings (RoPE) for position encoding
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, d_model)
        w: Attention weights module containing qkv projection and output projection
        freqs_cis: Precomputed frequencies for rotary positional embeddings
        kv_cache: Key-value cache for efficient generation (None during training)
        attn_mask: Attention mask to prevent attending to certain positions
        n_heads: Number of attention heads for queries
        n_kv_heads: Number of attention heads for keys/values (can be < n_heads for GQA)
        position_ids: Position indices for each token in the sequence
        
    Returns:
        Tensor of shape (batch_size, seq_len, d_model) after attention and projection
    """
    bsz, q_len, d_model = x.shape
    head_dim = d_model // n_heads

    qkv_out = w.qkv(x)  # shape: (bsz, q_len, (n_heads + 2*n_kv_heads)*head_dim)
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim
    q, k, v = qkv_out.split([q_dim, kv_dim, kv_dim], dim=-1)
    del qkv_out

    q = q.view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    k = k.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)
    v = v.view(bsz, q_len, n_kv_heads, head_dim).transpose(1, 2)

    q = apply_rotary_emb(q, freqs_cis, position_ids, n_heads)
    k = apply_rotary_emb(k, freqs_cis, position_ids, n_kv_heads)

    if kv_cache is not None:
        k, v = kv_cache.update(position_ids, k, v)

    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, enable_gqa=n_heads != n_kv_heads
    )
    out = out.transpose(1, 2).reshape(bsz, q_len, d_model)
    out = w.proj(out)
    return out


def _attn(
    x: torch.Tensor,
    w: torch.Tensor,
    freqs_cis: torch.Tensor,
    attn_mask: torch.Tensor,
    n_heads: int,
    n_kv_heads: int,
):
    """
    Simplified attention function for training/inference without KV caching.
    
    This is a streamlined version of the attention mechanism that doesn't use
    KV caching, making it suitable for training where all tokens are processed
    at once rather than autoregressively.
    
    Args:
        x: Input tensor of shape (batch_size, seq_len, d_model)
        w: Attention weights containing qkv and proj layers
        freqs_cis: Precomputed rotary positional embedding frequencies
        attn_mask: Attention mask for preventing attention to certain positions
        n_heads: Number of query attention heads
        n_kv_heads: Number of key/value attention heads (for GQA)
        
    Returns:
        Tensor of shape (batch_size, seq_len, d_model) after attention
    """
    bsz, q_len, d_model = x.shape
    head_dim = d_model // n_heads
    pos = 0

    qkv_out = w.qkv(x)  # shape: (bsz, q_len, (n_heads + 2*n_kv_heads)*head_dim)
    q_dim = n_heads * head_dim
    kv_dim = n_kv_heads * head_dim

    q = qkv_out[..., :q_dim].view(bsz, q_len, n_heads, head_dim).transpose(1, 2)
    k = (
        qkv_out[..., q_dim : q_dim + kv_dim]
        .view(bsz, q_len, n_kv_heads, head_dim)
        .transpose(1, 2)
    )
    v = (
        qkv_out[..., q_dim + kv_dim :]
        .view(bsz, q_len, n_kv_heads, head_dim)
        .transpose(1, 2)
    )

    position_ids = torch.arange(pos, pos + q_len, dtype=torch.long)
    q = apply_rotary_emb(q, freqs_cis, position_ids, n_heads)
    k = apply_rotary_emb(k, freqs_cis, position_ids, n_kv_heads)
    out = F.scaled_dot_product_attention(
        q, k, v, attn_mask=attn_mask, enable_gqa=n_heads != n_kv_heads
    )
    out = out.transpose(1, 2).reshape(bsz, q_len, d_model)
    out = w.proj(out)
    return out


def _produce_hidden(
    inputs_embeds: torch.Tensor,
    w: nn.Module,
    config: TextConfig,
    attention_mask: torch.Tensor = None,
):
    """
    Process input embeddings through transformer blocks to produce hidden states.
    
    This function runs the input embeddings through all transformer blocks, applying
    layer normalization, attention, and MLP layers at each step. It supports custom
    attention masks for controlling which tokens can attend to each other.
    
    The function implements the standard transformer architecture with pre-layer
    normalization and residual connections. The attention mask allows for flexible
    attention patterns beyond the default causal masking.
    
    Args:
        inputs_embeds: Input token embeddings of shape (batch_size, seq_len, d_model)
        w: Transformer weights module containing blocks and frequency embeddings
        config: Text model configuration containing architectural parameters
        attention_mask: Optional mask of shape (batch_size, seq_len) where 1 = attend,
                       0 = ignore. If None, uses default causal masking with special
                       handling for sequences longer than 730 tokens.
                       
    Returns:
        Tensor of shape (batch_size, seq_len, d_model) containing final hidden states
        after processing through all transformer blocks
    """
    hidden_BTC = inputs_embeds

    bsz, q_len, d_model = inputs_embeds.shape

    # Create causal mask with attention_mask if provided
    if attention_mask is not None:
        # Convert attention_mask to boolean where True = attend, False = ignore
        attn_mask = attention_mask.to(dtype=torch.bool)
        # Expand mask to match attention scores shape
        attn_mask = attn_mask[:, None, None, :].expand(bsz, 1, q_len, q_len)
    else:
        # Original causal mask logic
        attn_mask = torch.zeros(q_len, q_len)
        attn_mask[:730, :730] = 1
        for i in range(730, q_len):
            attn_mask[i, : i + 1] = 1
        attn_mask = attn_mask.to(dtype=torch.bool)

    for i, block in enumerate(w.blocks):
        l_in = layer_norm(hidden_BTC, block.ln)
        l_attn = _attn(
            x=l_in,
            w=block.attn,
            freqs_cis=w.freqs_cis,
            attn_mask=attn_mask,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
        )
        l_mlp = mlp(l_in, block.mlp)
        hidden_BTC = hidden_BTC + l_attn + l_mlp

    return hidden_BTC


def text_decoder(
    x: torch.Tensor,
    w: nn.Module,
    attn_mask: torch.Tensor,
    position_ids: torch.Tensor,
    config: TextConfig,
):
    """
    Decode hidden states through transformer blocks with KV caching for generation.
    
    This function is optimized for autoregressive text generation, where tokens are
    generated one at a time. It uses KV caching to avoid recomputing attention keys
    and values for previously generated tokens, significantly speeding up inference.
    
    Args:
        x: Input hidden states of shape (batch_size, seq_len, d_model)
        w: Transformer weights module containing blocks with KV caches
        attn_mask: Attention mask to control which positions can be attended to
        position_ids: Position indices for each token in the sequence
        config: Text model configuration with architectural parameters
        
    Returns:
        Tensor of shape (batch_size, seq_len, d_model) containing processed hidden states
    """
    for i, block in enumerate(w.blocks):
        l_in = layer_norm(x, block.ln)
        l_attn = attn(
            l_in,
            block.attn,
            freqs_cis=w.freqs_cis,
            kv_cache=block.kv_cache,
            attn_mask=attn_mask,
            n_heads=config.n_heads,
            n_kv_heads=config.n_kv_heads,
            position_ids=position_ids,
        )
        l_mlp = mlp(l_in, block.mlp)
        x = x + l_attn + l_mlp

    return x


def lm_head(hidden_BTC: torch.Tensor, w: nn.Module):
    """
    Apply language modeling head to generate logits for the last token.
    
    This function extracts the final hidden state (last token) from the sequence,
    applies layer normalization, and projects it through the language modeling head
    to produce vocabulary logits. This is used during autoregressive generation
    where only the next token prediction is needed.
    
    Args:
        hidden_BTC: Hidden states of shape (batch_size, seq_len, d_model)
        w: Module containing post_ln (layer norm) and lm_head (linear projection)
        
    Returns:
        Tensor of shape (batch_size, vocab_size) containing logits for the last token
    """
    hidden_BC = hidden_BTC[:, -1, :]
    hidden_BC = layer_norm(hidden_BC, w.post_ln)
    logits = w.lm_head(hidden_BC)
    return logits


def _lm_head(hidden_BTC: torch.Tensor, w: nn.Module):
    """
    Apply language modeling head to generate logits for all tokens in sequence.
    
    Unlike lm_head() which only processes the last token, this function applies
    layer normalization and the language modeling head to all tokens in the sequence.
    This is typically used during training where loss is computed across all positions,
    or when you need logits for the entire sequence.
    
    Args:
        hidden_BTC: Hidden states of shape (batch_size, seq_len, d_model)
        w: Module containing post_ln (layer norm) and lm_head (linear projection)
        
    Returns:
        Tensor of shape (batch_size, seq_len, vocab_size) containing logits for all tokens
    """
    hidden_BTC = layer_norm(hidden_BTC, w.post_ln)
    logits = w.lm_head(hidden_BTC)
    return logits


def build_text_model(config: TextConfig, dtype: torch.dtype) -> nn.Module:
    """
    Build a complete text transformer model from configuration.
    
    This function constructs the full transformer architecture including:
    - Word token embeddings (wte)
    - Multiple transformer blocks with attention and MLP layers
    - Post-layer normalization and language modeling head
    - Rotary positional embedding frequencies
    
    The model supports several advanced features:
    - Grouped Query Attention (GQA) where n_kv_heads can be less than n_heads
    - Optional quantization for memory efficiency
    - Rotary positional embeddings for better position encoding
    
    Args:
        config: TextConfig object containing model hyperparameters like:
            - dim: Model dimension (d_model)
            - n_layers: Number of transformer blocks
            - n_heads: Number of attention heads for queries
            - n_kv_heads: Number of attention heads for keys/values (for GQA)
            - ff_dim: Feed-forward network hidden dimension
            - vocab_size: Size of the vocabulary
            - max_context: Maximum sequence length for positional embeddings
            - group_size: Quantization group size (None for no quantization)
        dtype: PyTorch data type for model parameters (e.g., torch.float16, torch.bfloat16)
        
    Returns:
        nn.Module: Complete transformer model ready for training or inference
    """
    qkv_dim = int(config.dim * (1 + 2 * config.n_kv_heads / config.n_heads))
    linear_cls = QuantizedLinear if config.group_size is not None else nn.Linear

    text = nn.ModuleDict(
        {
            "blocks": nn.ModuleList(
                [
                    nn.ModuleDict(
                        {
                            "ln": nn.LayerNorm(config.dim, dtype=dtype),
                            "attn": nn.ModuleDict(
                                {
                                    "qkv": linear_cls(config.dim, qkv_dim, dtype=dtype),
                                    "proj": linear_cls(
                                        config.dim, config.dim, dtype=dtype
                                    ),
                                }
                            ),
                            "mlp": nn.ModuleDict(
                                {
                                    "fc1": linear_cls(
                                        config.dim, config.ff_dim, dtype=dtype
                                    ),
                                    "fc2": linear_cls(
                                        config.ff_dim, config.dim, dtype=dtype
                                    ),
                                }
                            ),
                        }
                    )
                    for _ in range(config.n_layers)
                ]
            ),
            "post_ln": nn.LayerNorm(config.dim, dtype=dtype),
            "lm_head": linear_cls(config.dim, config.vocab_size, dtype=dtype),
        }
    )
    text.wte = nn.Parameter(torch.empty(config.vocab_size, config.dim, dtype=dtype))
    text.register_buffer(
        "freqs_cis",
        precompute_freqs_cis(config.dim // (2 * config.n_heads), config.max_context),
        persistent=False,
    )

    return text
