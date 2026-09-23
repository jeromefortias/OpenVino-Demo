import os
import torch
from safetensors import safe_open

safetensors_path = "./safetensors_xlm_roberta/model.safetensors"
output_weights_txt = "model_raw_weights.txt"

if not os.path.exists(safetensors_path):
    raise FileNotFoundError(f"Safetensors file not found at '{safetensors_path}'.")

print("[+] Writing raw tensor weights to text file (this will create a large file)...")

with open(output_weights_txt, "w", encoding="utf-8") as out:
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for idx, key in enumerate(f.keys(), 1):
            print(f"    [{idx}/{len(f.keys())}] Exporting tensor: {key}")
            tensor = f.get_tensor(key)
            
            out.write(f"=== TENSOR: {key} | SHAPE: {list(tensor.shape)} | DTYPE: {tensor.dtype} ===\n")
            
            # Flatten tensor to write all numbers clearly
            flattened_weights = tensor.flatten().tolist()
            
            # Write numbers in chunks to avoid memory bottlenecks
            chunk_size = 1000
            for i in range(0, len(flattened_weights), chunk_size):
                chunk = flattened_weights[i:i + chunk_size]
                out.write(" ".join(map(str, chunk)) + "\n")
                
            out.write("\n\n")

print(f"[+] Done! All raw numerical weights written to '{output_weights_txt}'.")