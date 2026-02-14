import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, 
                                 K: torch.Tensor, 
                                 V: torch.Tensor,
                                 mask: torch.Tensor = None) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    
    Args:
        Q: (batch_size, seq_len_q, d_k)
        K: (batch_size, seq_len_k, d_k)
        V: (batch_size, seq_len_k, d_v)
        mask: Optional mask tensor
        
    Returns:
        Output tensor after applying attention
    """

    # Step 1: Compute attention scores
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Step 2: Apply mask (if provided)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 3: Softmax over last dimension
    attention_weights = F.softmax(scores, dim=-1)

    # Step 4: Multiply with V
    output = torch.matmul(attention_weights, V)

    return output