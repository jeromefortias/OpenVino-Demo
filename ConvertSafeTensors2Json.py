# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 21:57:22 2026

@author: jerom
"""

import os
import json
from safetensors import safe_open

safetensors_path = "./safetensors_xlm_roberta/model.safetensors"
output_weights_json = "model_raw_weights.json"

if not os.path.exists(safetensors_path):
    raise FileNotFoundError(f"Safetensors file not found at '{safetensors_path}'.")

print("[+] Reading .safetensors weights into memory...")
weights_dict = {}

with safe_open(safetensors_path, framework="pt", device="cpu") as f:
    for key in f.keys():
        print(f"    - Serializing tensor: {key}")
        # Convert torch Tensor -> numpy array -> Python list -> JSON serializable
        weights_dict[key] = f.get_tensor(key).detach().cpu().numpy().tolist()

print(f"[+] Dumping raw float lists to '{output_weights_json}' (this may take a few minutes)...")
with open(output_weights_json, "w", encoding="utf-8") as f:
    json.dump(weights_dict, f)

print(f"[+] Done! Raw weights exported to '{output_weights_json}'.")