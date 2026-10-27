"""Validate loaded inference weights; never silently accept random decoder weights."""

from contextlib import contextmanager

ALIASES = (
    (".attention.attention.query.", ".attention.q_proj."),
    (".attention.attention.key.", ".attention.k_proj."),
    (".attention.attention.value.", ".attention.v_proj."),
    (".attention.output.dense.", ".attention.o_proj."),
)


def align_conditioner_keys(state_dict, target_keys):
    """Rename only when the target explicitly contains that exact counterpart."""
    out = {}
    renamed = []
    for key, tensor in state_dict.items():
        new = key
        if key not in target_keys:
            for old, replacement in ALIASES:
                if old in key and key.replace(old, replacement) in target_keys:
                    new = key.replace(old, replacement)
                    break
                if replacement in key and key.replace(replacement, old) in target_keys:
                    new = key.replace(replacement, old)
                    break
        if new in out:
            raise ValueError(f"Weight-key collision: {new}")
        out[new] = tensor
        if new != key:
            renamed.append([key, new])
    return out, renamed


def validate_keys(module_name, missing, unexpected):
    # The published 2mv inference checkpoint omits the mesh encoder, which is
    # not executed by image-to-shape decoding. Do NOT exempt decoder/post_kl.
    allowed = [
        k
        for k in missing
        if module_name == "ShapeVAE" and k.startswith(("encoder.", "pre_kl."))
    ]
    fatal = [k for k in missing if k not in allowed]
    if fatal or unexpected:
        raise RuntimeError(
            f"{module_name}: missing inference weights={fatal}; "
            f"unexpected weights={list(unexpected)}"
        )
    return allowed


@contextmanager
def checked_loading(torch, reports):
    """Single-process load scope. Restores the loader even when construction fails."""
    original = torch.nn.Module.load_state_dict

    def checked(module, state_dict, strict=True, **kwargs):
        renamed = []
        if type(module).__name__ == "SingleImageEncoder":
            state_dict, renamed = align_conditioner_keys(
                state_dict, set(module.state_dict())
            )
        # Always collect discrepancies, then apply our stricter inference rule.
        result = original(module, state_dict, strict=False, **kwargs)
        allowed = validate_keys(
            type(module).__name__, result.missing_keys, result.unexpected_keys
        )
        reports.append(
            dict(
                module=type(module).__name__,
                tensors=len(state_dict),
                renamed=renamed,
                omitted_training_keys=allowed,
                missing_inference_keys=[],
                unexpected_keys=[],
            )
        )
        return result

    torch.nn.Module.load_state_dict = checked
    try:
        yield
    finally:
        torch.nn.Module.load_state_dict = original
