import torch

def fedavg(updates):
    
    if not updates:
        raise ValueError("fedavg needs at least one update")

    for i, (_, n) in enumerate(updates):
        if n <= 0:
            raise ValueError(f"update {i} has a non-positive sample count: {n}")

    total = sum(n_samples for _, n_samples in updates)

    reference_sd = updates[0][0]
    keys = set(reference_sd.keys())
    for i, (sd, _) in enumerate(updates[1:], start=1):
        other = set(sd.keys())
        if other != keys:
            raise ValueError(
                f"update {i} doesn't match update 0: "
                f"missing {sorted(keys - other)}, extra {sorted(other - keys)}. "
                "Are all nodes running the same model code?"
            )
        for key in keys:
            if sd[key].shape != reference_sd[key].shape:
                raise ValueError(
                    f"update {i}: '{key}' has shape {tuple(sd[key].shape)}, "
                    f"expected {tuple(reference_sd[key].shape)}. "
                    "Are all nodes running the same model code?"
                )

    averaged = {}
    for key in reference_sd:
        ref_tensor = reference_sd[key]
        combined = sum(sd[key].float() * (n / total) for sd, n in updates)
        if not ref_tensor.is_floating_point():
            combined = combined.round()   # integer tensors: round, don't truncate
        averaged[key] = combined.to(ref_tensor.dtype)
    return averaged