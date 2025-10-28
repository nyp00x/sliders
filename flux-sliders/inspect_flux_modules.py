#!/usr/bin/env python3
"""
Inspect FLUX Transformer module names to understand its architecture
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import torch
from diffusers import FluxTransformer2DModel

def inspect_flux_modules(model_path):
    print(f"Loading FLUX model from: {model_path}")
    print("=" * 80)

    # Load transformer
    transformer = FluxTransformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16
    )

    print(f"\nTotal modules in transformer: {len(list(transformer.named_modules()))}")
    print("\n" + "=" * 80)

    # Find all attention-related modules
    print("\nATTENTION-RELATED MODULES:")
    print("-" * 80)
    attn_modules = []
    for name, module in transformer.named_modules():
        if any(keyword in name.lower() for keyword in ['attn', 'attention']):
            attn_modules.append((name, module.__class__.__name__))

    if attn_modules:
        for name, class_name in attn_modules[:50]:  # Show first 50
            print(f"{name:80s} -> {class_name}")
        if len(attn_modules) > 50:
            print(f"\n... and {len(attn_modules) - 50} more attention modules")
        print(f"\nTotal attention modules: {len(attn_modules)}")
    else:
        print("No modules with 'attn' or 'attention' in name found!")

    # Check for Linear layers
    print("\n" + "=" * 80)
    print("\nLINEAR LAYERS (first 30):")
    print("-" * 80)
    linear_modules = []
    for name, module in transformer.named_modules():
        if 'Linear' in module.__class__.__name__:
            linear_modules.append((name, module.__class__.__name__))

    for name, class_name in linear_modules[:30]:
        print(f"{name:80s} -> {class_name}")
    if len(linear_modules) > 30:
        print(f"\n... and {len(linear_modules) - 30} more Linear modules")
    print(f"\nTotal Linear modules: {len(linear_modules)}")

    # Check module types
    print("\n" + "=" * 80)
    print("\nMODULE TYPE SUMMARY:")
    print("-" * 80)
    module_types = {}
    for name, module in transformer.named_modules():
        class_name = module.__class__.__name__
        if class_name not in module_types:
            module_types[class_name] = 0
        module_types[class_name] += 1

    for class_name, count in sorted(module_types.items(), key=lambda x: x[1], reverse=True)[:20]:
        print(f"{class_name:40s}: {count:5d}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to FLUX model")
    args = parser.parse_args()

    inspect_flux_modules(args.model_path)
