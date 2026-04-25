# Per-checkpoint config diffs vs baseline

Only fields that differ from the faithful GPT-2 baseline are shown.


## 01-modern-block

| field | baseline | this checkpoint |
|-------|----------|-----------------|
| `positional_encoding` | `None` | `rope` |
| `norm_type` | `None` | `rmsnorm` |
| `mlp_type` | `None` | `swiglu` |
| `qk_norm` | `None` | `True` |

## 02-muon

| field | baseline | this checkpoint |
|-------|----------|-----------------|
| `positional_encoding` | `None` | `rope` |
| `norm_type` | `None` | `rmsnorm` |
| `mlp_type` | `None` | `swiglu` |
| `qk_norm` | `None` | `True` |

## 03-modded-tricks

| field | baseline | this checkpoint |
|-------|----------|-----------------|
| `positional_encoding` | `None` | `rope` |
| `norm_type` | `None` | `rmsnorm` |
| `mlp_type` | `None` | `relu2` |
| `qk_norm` | `None` | `True` |
| `zero_init_proj` | `None` | `True` |
| `u_net_skips` | `None` | `True` |
| `logit_softcap` | `None` | `30.0` |

## 05-speed-pack

| field | baseline | this checkpoint |
|-------|----------|-----------------|
| `positional_encoding` | `None` | `rope` |
| `norm_type` | `None` | `rmsnorm` |
| `mlp_type` | `None` | `relu2` |
| `qk_norm` | `None` | `True` |
| `zero_init_proj` | `None` | `True` |
| `u_net_skips` | `None` | `True` |
| `n_kv_head` | `None` | `4` |
| `use_liger_fused_ce` | `None` | `False` |

## 06-muon-mup

| field | baseline | this checkpoint |
|-------|----------|-----------------|
| `positional_encoding` | `None` | `rope` |
| `norm_type` | `None` | `rmsnorm` |
| `mlp_type` | `None` | `relu2` |
| `qk_norm` | `None` | `True` |
| `zero_init_proj` | `None` | `True` |
| `u_net_skips` | `None` | `True` |
| `logit_softcap` | `None` | `30.0` |

## 10-mla

| field | baseline | this checkpoint |
|-------|----------|-----------------|
| `positional_encoding` | `None` | `rope` |
| `norm_type` | `None` | `rmsnorm` |
| `mlp_type` | `None` | `relu2` |
| `qk_norm` | `None` | `True` |
| `zero_init_proj` | `None` | `True` |
| `u_net_skips` | `None` | `True` |
| `logit_softcap` | `None` | `30.0` |
| `attention_type` | `None` | `mla` |

## 11-loopllm

| field | baseline | this checkpoint |
|-------|----------|-----------------|
| `positional_encoding` | `None` | `rope` |
| `norm_type` | `None` | `rmsnorm` |
| `mlp_type` | `None` | `relu2` |
| `qk_norm` | `None` | `True` |
| `zero_init_proj` | `None` | `True` |
| `u_net_skips` | `None` | `False` |
| `logit_softcap` | `None` | `30.0` |
| `weight_tied` | `None` | `True` |
