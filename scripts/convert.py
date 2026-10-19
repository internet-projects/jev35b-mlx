"""Convert autotrust/JEV-35B to MLX with mlx-vlm (the vision tower is kept, in bf16).

  python scripts/convert.py <hf folder or repo> <out folder> <recipe>

recipes
  8      everything 8-bit (affine, group 64)                         -> JEV-35B-mlx-8bit   (36 GB)
  4e     routed experts 4-bit, every other layer 8-bit               -> JEV-35B-mlx-4bit-mixed (21 GB)
  3e     routed experts 3-bit, every other layer 8-bit               -> JEV-35B-mlx-3bit-mixed (17 GB)
  N      everything N-bit, lm_head 8-bit
  bf16   no quantization (reference)
Always: router gates (mlp.gate, shared_expert_gate) and the vision tower stay bf16.
The decision files (adapter_vllm/decision_head.json, calibration.json, judge_config.json) are copied along.
"""
import os, shutil, sys

from mlx_vlm.convert import convert
from mlx_vlm.utils import skip_multimodal_module


def main():
    src, out, recipe = sys.argv[1], sys.argv[2], sys.argv[3]
    if not os.path.isdir(src):
        from huggingface_hub import snapshot_download
        src = snapshot_download(src, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "adapter_vllm/*"])
    if recipe == "bf16":
        convert(src, out, quantize=False, dtype="bfloat16")
    else:
        experts_only = recipe.endswith("e")
        bits = int(recipe.rstrip("e"))

        def predicate(path, module):
            if skip_multimodal_module(path) or not hasattr(module, "to_quantized"):
                return False                                   # vision tower: bf16
            if hasattr(module, "weight") and module.weight.shape[-1] % 64 != 0:
                return False
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return False                                   # router gates: bf16
            if experts_only:
                return {"group_size": 64, "bits": bits if "switch_mlp" in path else 8}
            if path.endswith("lm_head"):                       # the decision head lives in lm_head rows
                return {"group_size": 64, "bits": 8}
            return {"group_size": 64, "bits": bits}

        convert(src, out, quantize=True, q_bits=bits, q_group_size=64, quant_predicate=predicate)
    os.makedirs(os.path.join(out, "adapter_vllm"), exist_ok=True)
    for f in ("calibration.json", "judge_config.json", "adapter_vllm/decision_head.json"):
        shutil.copy(os.path.join(src, f), os.path.join(out, f))
    print("converted", out)


if __name__ == "__main__":
    main()
