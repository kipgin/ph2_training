import gc
import torch
import torch.nn as nn

from .embeddings import CLIPTextEmbeddings
from .attention import CLIPEncoder

def _build_causal_attention_mask(bsz, seq_len, dtype, device):
    mask = torch.empty(seq_len, seq_len, dtype=dtype, device=device)
    mask.fill_(float("-inf"))
    mask.triu_(1)
    return mask.unsqueeze(0).unsqueeze(0)


class CLIPTextModel(nn.Module):
    """Flat CLIPTextModel whose state_dict keys match transformers >= 5.6.

    Since transformers 5.6 removed the intermediate `text_model` attribute,
    the state_dict keys are now flat (e.g. `embeddings.token_embedding.weight`)
    rather than `text_model.embeddings.token_embedding.weight`.  We therefore
    place `embeddings`, `encoder`, and `final_layer_norm` directly on this
    class to produce the same key layout.
    """
    def __init__(self, config=None, vocab_size=49408, hidden_size=768, max_position_embeddings=77, num_layers=12, num_heads=12, intermediate_size=3072, num_hidden_layers=None, num_attention_heads=None):
        super().__init__()
        if num_hidden_layers is not None:
            num_layers = num_hidden_layers
        if num_attention_heads is not None:
            num_heads = num_attention_heads

        if config is not None:
            vocab_size = getattr(config, "vocab_size", vocab_size)
            hidden_size = getattr(config, "hidden_size", hidden_size)
            max_position_embeddings = getattr(config, "max_position_embeddings", max_position_embeddings)
            num_layers = getattr(config, "num_hidden_layers", num_layers)
            num_heads = getattr(config, "num_attention_heads", num_heads)
            intermediate_size = getattr(config, "intermediate_size", intermediate_size)

        # Flat structure to match transformers >= 5.6 state_dict key layout
        self.embeddings = CLIPTextEmbeddings(vocab_size, hidden_size, max_position_embeddings)
        self.encoder = CLIPEncoder(num_layers, hidden_size, num_heads, intermediate_size)
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)

    def forward(self, input_ids):
        bsz, seq_len = input_ids.shape
        attention_mask = _build_causal_attention_mask(
            bsz, seq_len,
            self.embeddings.token_embedding.weight.dtype,
            input_ids.device
        )
        hidden_states = self.embeddings(input_ids)
        hidden_states = self.encoder(hidden_states, attention_mask=attention_mask)
        hidden_states = self.final_layer_norm(hidden_states)
        return (hidden_states,)

    @classmethod
    def from_pretrained(cls, model_id, subfolder="text_encoder", torch_dtype=torch.float32, **kwargs):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        from transformers import CLIPTextModel as HFCLIPTextModel
        model = cls(**kwargs)
        hf_model = HFCLIPTextModel.from_pretrained(model_id, subfolder=subfolder, torch_dtype=torch_dtype)
        model = model.to(dtype=torch_dtype)
        result = model.load_state_dict(hf_model.state_dict(), strict=False)
        missing_keys = [k for k in result.missing_keys if k != "embeddings.position_ids"]
        if missing_keys:
            print(f"[CLIP] Missing keys ({len(missing_keys)}): {missing_keys[:5]} ...")
        if result.unexpected_keys:
            print(f"[CLIP] Unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys[:5]} ...")
        if not missing_keys and not result.unexpected_keys:
            print("[CLIP] All keys loaded successfully ✅")
        
        del hf_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return model
