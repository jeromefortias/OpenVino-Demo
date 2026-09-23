import os
import time
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from optimum.intel.openvino import OVModelForTokenClassification
from optimum.intel.openvino.quantization import OVQuantizer

# ---------------------------------------------------------
# STEP 0: CONFIGURATION & INPUT INITIALIZATION
# ---------------------------------------------------------
print("=" * 80)
print("STEP 0: INITIALIZING BENCHMARK CONFIGURATION & LOADING INPUT DATA")
print("=" * 80)

model_name = "FacebookAI/xlm-roberta-large-finetuned-conll03-english"
local_safetensors_dir = "./safetensors_xlm_roberta"
local_openvino_dir = "./openvino_xlm_roberta"          # Unquantized FP16 Base OpenVINO Model
local_openvino_int8_dir = "./openvino_xlm_roberta_int8"  # INT8 Quantized Model
local_openvino_int4_dir = "./openvino_xlm_roberta_int4"  # INT4 Compressed Model

input_file = "lines.txt"
csv_file = "benchmark_summary_xlm_roberta_4way.csv"

# Output Graphics
dashboard_file = "average_performance_comparison_xlm_roberta_4way.png"
latency_box_file = "latency_distribution_boxplot_xlm_roberta.png"
scaling_scatter_file = "token_scaling_scatter_xlm_roberta.png"

if not os.path.exists(input_file):
    raise FileNotFoundError(f"'{input_file}' not found. Please create the text file with sentences.")

with open(input_file, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"[+] Loaded {len(lines)} sentences from '{input_file}'.")
print(f"[+] Target Hugging Face Model: '{model_name}'\n")

# ---------------------------------------------------------
# STEP 0.5: DOWNLOAD AND STORE MODEL LOCALLY
# ---------------------------------------------------------
print("=" * 80)
print("STEP 0.5: DOWNLOADING AND STORING HUGGING FACE MODEL LOCALLY")
print("=" * 80)

if not os.path.exists(local_safetensors_dir):
    print(f"[+] Local folder '{local_safetensors_dir}' does not exist. Creating directory...")
    os.makedirs(local_safetensors_dir, exist_ok=True)
    
    print(f"[+] Downloading model weights ({model_name}) from Hugging Face Hub...")
    downloaded_model = AutoModelForTokenClassification.from_pretrained(
        model_name, 
        use_safetensors=True
    )
    print(f"[+] Downloading tokenizer for '{model_name}'...")
    downloaded_tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"[+] Saving model weights and tokenizer into local folder: '{local_safetensors_dir}'...")
    downloaded_model.save_pretrained(local_safetensors_dir)
    downloaded_tokenizer.save_pretrained(local_safetensors_dir)
    print(f"[+] Model successfully downloaded and saved to disk at '{local_safetensors_dir}'.\n")
else:
    print(f"[+] Model already stored locally in '{local_safetensors_dir}'. Skipping download.\n")

# Load Tokenizer from local directory
tokenizer = AutoTokenizer.from_pretrained(local_safetensors_dir)
print("[+] Local tokenizer successfully loaded.\n")

# ---------------------------------------------------------
# STEP 1: MODEL EXPORT & QUANTIZATION PREPARATION
# ---------------------------------------------------------
print("=" * 80)
print("STEP 1: PREPARING OPENVINO MODELS (BASE FP16, INT8, INT4)")
print("=" * 80)

# A. Base Unquantized OpenVINO Model (FP16 Export)
if not os.path.exists(local_openvino_dir):
    print(f"[+] [Base OpenVINO] Exporting local model '{local_safetensors_dir}' to OpenVINO IR FP16...")
    base_ov_model = OVModelForTokenClassification.from_pretrained(local_safetensors_dir, export=True)
    base_ov_model.save_pretrained(local_openvino_dir)
    tokenizer.save_pretrained(local_openvino_dir)
    print(f"[+] [Base OpenVINO] Model saved locally at '{local_openvino_dir}'.")
else:
    print(f"[+] [Base OpenVINO] Found existing model at '{local_openvino_dir}'. Skipping export.")

# B. OpenVINO INT8 Model (NNCF)
if not os.path.exists(local_openvino_int8_dir):
    print("[+] [OpenVINO INT8] Quantizing base model to 8-bit integers using NNCF...")
    os.makedirs(local_openvino_int8_dir, exist_ok=True)
    base_ov_model = OVModelForTokenClassification.from_pretrained(local_openvino_dir)
    quantizer = OVQuantizer.from_pretrained(base_ov_model)
    quantizer.quantize(save_directory=local_openvino_int8_dir)
    tokenizer.save_pretrained(local_openvino_int8_dir)
    print(f"[+] [OpenVINO INT8] Quantized model saved at '{local_openvino_int8_dir}'.")
else:
    print(f"[+] [OpenVINO INT8] Found existing INT8 model at '{local_openvino_int8_dir}'. Skipping quantization.")

# C. OpenVINO INT4 Model (Weight Compression)
if not os.path.exists(local_openvino_int4_dir):
    print("[+] [OpenVINO INT4] Compressing model weights to 4-bit integers...")
    os.makedirs(local_openvino_int4_dir, exist_ok=True)
    ov_int4_model = OVModelForTokenClassification.from_pretrained(
        local_openvino_dir,
        quantization_config={"bits": 4, "sym": False, "group_size": 128}
    )
    ov_int4_model.save_pretrained(local_openvino_int4_dir)
    tokenizer.save_pretrained(local_openvino_int4_dir)
    print(f"[+] [OpenVINO INT4] Compressed model saved at '{local_openvino_int4_dir}'.")
else:
    print(f"[+] [OpenVINO INT4] Found existing INT4 model at '{local_openvino_int4_dir}'. Skipping compression.")

print("\nAll 4 engine variants are compiled and ready for benchmark.\n")

# ---------------------------------------------------------
# STEP 2: BENCHMARK 1 - PYTORCH GPU (NVIDIA CUDA)
# ---------------------------------------------------------
print("=" * 80)
print("STEP 2: RUNNING BENCHMARK [1/4] -> PyTorch Native on NVIDIA RTX 4090 (CUDA)")
print("=" * 80)

if not torch.cuda.is_available():
    raise SystemError("CUDA is not available. Please verify PyTorch CUDA installation and NVIDIA drivers.")

gpu_device = "cuda"
gpu_name = torch.cuda.get_device_name(0)
print(f"[+] Target hardware device selected: '{gpu_device.upper()}' ({gpu_name})")
print(f"[+] Loading weights into PyTorch model pipeline from '{local_safetensors_dir}'...")

gpu_model = AutoModelForTokenClassification.from_pretrained(
    local_safetensors_dir,
    use_safetensors=True
).to(gpu_device)

# Using device=0 explicitly routes Hugging Face pipeline to CUDA device 0
gpu_pipeline = pipeline("ner", model=gpu_model, tokenizer=tokenizer, device=0, aggregation_strategy="simple")

print("[+] Running warm-up inference pass (initializing CUDA context and allocating VRAM)...")
_ = gpu_pipeline("Warm-up run to initialize CUDA context.")
torch.cuda.synchronize()

print(f"[+] Starting timed benchmark execution for {len(lines)} sentences...")
gpu_results = []
for idx, line in enumerate(lines, 1):
    token_count = len(tokenizer.encode(line))
    
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    _ = gpu_pipeline(line)
    torch.cuda.synchronize()
    t_end = time.perf_counter()
    
    raw_ms = (t_end - t_start) * 1000
    ms_per_token = raw_ms / token_count if token_count > 0 else 0
    
    gpu_results.append({
        "line_index": idx,
        "engine": f"PyTorch GPU (CUDA - {gpu_name})",
        "token_count": token_count,
        "execution_time_ms": raw_ms,
        "ms_per_token": ms_per_token
    })
    
    if idx % 25 == 0 or idx == len(lines):
        print(f"    - Processed [{idx:3d}/{len(lines)}] sentences | Last line latency: {raw_ms:6.2f} ms ({ms_per_token:4.2f} ms/token)")

print("[+] Cleaning up GPU VRAM memory before starting CPU benchmarks...")
del gpu_model, gpu_pipeline
torch.cuda.empty_cache()
print("[+] PyTorch GPU Benchmark completed successfully.\n")

# ---------------------------------------------------------
# STEP 3: BENCHMARK 2 - OPENVINO AMD CPU (UNQUANTIZED FP16)
# ---------------------------------------------------------
print("=" * 80)
print("STEP 3: RUNNING BENCHMARK [2/4] -> OpenVINO AMD CPU (Base Unquantized / FP16)")
print("=" * 80)

print(f"[+] Loading unquantized OpenVINO execution graph from '{local_openvino_dir}'...")
ov_base_model = OVModelForTokenClassification.from_pretrained(local_openvino_dir, device="CPU")
ov_base_pipeline = pipeline("ner", model=ov_base_model, tokenizer=tokenizer, aggregation_strategy="simple")

print("[+] Running warm-up inference pass (initializing AMD CPU execution threads)...")
_ = ov_base_pipeline("Warm-up run for CPU initialization.")

print(f"[+] Starting timed benchmark execution for {len(lines)} sentences...")
ov_base_results = []
for idx, line in enumerate(lines, 1):
    token_count = len(tokenizer.encode(line))
    
    t_start = time.perf_counter()
    _ = ov_base_pipeline(line)
    t_end = time.perf_counter()
    
    raw_ms = (t_end - t_start) * 1000
    ms_per_token = raw_ms / token_count if token_count > 0 else 0
    
    ov_base_results.append({
        "line_index": idx,
        "engine": "OpenVINO CPU (Base FP16)",
        "token_count": token_count,
        "execution_time_ms": raw_ms,
        "ms_per_token": ms_per_token
    })
    
    if idx % 25 == 0 or idx == len(lines):
        print(f"    - Processed [{idx:3d}/{len(lines)}] sentences | Last line latency: {raw_ms:6.2f} ms ({ms_per_token:4.2f} ms/token)")

print("[+] OpenVINO Base CPU Benchmark completed successfully.\n")

# ---------------------------------------------------------
# STEP 4: BENCHMARK 3 - OPENVINO INT8 (AMD CPU)
# ---------------------------------------------------------
print("=" * 80)
print("STEP 4: RUNNING BENCHMARK [3/4] -> OpenVINO INT8 (AMD CPU Quantized)")
print("=" * 80)

print(f"[+] Loading INT8 OpenVINO execution graph from '{local_openvino_int8_dir}'...")
ov_int8_model = OVModelForTokenClassification.from_pretrained(local_openvino_int8_dir, device="CPU")
ov_int8_pipeline = pipeline("ner", model=ov_int8_model, tokenizer=tokenizer, aggregation_strategy="simple")

print("[+] Running warm-up inference pass (initializing vector INT8 CPU kernels)...")
_ = ov_int8_pipeline("Warm-up run for INT8 initialization.")

print(f"[+] Starting timed benchmark execution for {len(lines)} sentences...")
ov_int8_results = []
for idx, line in enumerate(lines, 1):
    token_count = len(tokenizer.encode(line))
    
    t_start = time.perf_counter()
    _ = ov_int8_pipeline(line)
    t_end = time.perf_counter()
    
    raw_ms = (t_end - t_start) * 1000
    ms_per_token = raw_ms / token_count if token_count > 0 else 0
    
    ov_int8_results.append({
        "line_index": idx,
        "engine": "OpenVINO INT8 (CPU)",
        "token_count": token_count,
        "execution_time_ms": raw_ms,
        "ms_per_token": ms_per_token
    })
    
    if idx % 25 == 0 or idx == len(lines):
        print(f"    - Processed [{idx:3d}/{len(lines)}] sentences | Last line latency: {raw_ms:6.2f} ms ({ms_per_token:4.2f} ms/token)")

print("[+] OpenVINO INT8 CPU Benchmark completed successfully.\n")

# ---------------------------------------------------------
# STEP 5: BENCHMARK 4 - OPENVINO INT4 (AMD CPU)
# ---------------------------------------------------------
print("=" * 80)
print("STEP 5: RUNNING BENCHMARK [4/4] -> OpenVINO INT4 (AMD CPU Compressed)")
print("=" * 80)

print(f"[+] Loading INT4 OpenVINO execution graph from '{local_openvino_int4_dir}'...")
ov_int4_model = OVModelForTokenClassification.from_pretrained(local_openvino_int4_dir, device="CPU")
ov_int4_pipeline = pipeline("ner", model=ov_int4_model, tokenizer=tokenizer, aggregation_strategy="simple")

print("[+] Running warm-up inference pass (initializing INT4 dequantization graphs)...")
_ = ov_int4_pipeline("Warm-up run for INT4 initialization.")

print(f"[+] Starting timed benchmark execution for {len(lines)} sentences...")
ov_int4_results = []
for idx, line in enumerate(lines, 1):
    token_count = len(tokenizer.encode(line))
    
    t_start = time.perf_counter()
    _ = ov_int4_pipeline(line)
    t_end = time.perf_counter()
    
    raw_ms = (t_end - t_start) * 1000
    ms_per_token = raw_ms / token_count if token_count > 0 else 0
    
    ov_int4_results.append({
        "line_index": idx,
        "engine": "OpenVINO INT4 (CPU)",
        "token_count": token_count,
        "execution_time_ms": raw_ms,
        "ms_per_token": ms_per_token
    })
    
    if idx % 25 == 0 or idx == len(lines):
        print(f"    - Processed [{idx:3d}/{len(lines)}] sentences | Last line latency: {raw_ms:6.2f} ms ({ms_per_token:4.2f} ms/token)")

print("[+] OpenVINO INT4 CPU Benchmark completed successfully.\n")

# ---------------------------------------------------------
# STEP 6: DATA CONSOLIDATION & GRAPHICS GENERATION
# ---------------------------------------------------------
print("=" * 80)
print("STEP 6: CONSOLIDATING METRICS & GENERATING HIGH-RESOLUTION GRAPHICS")
print("=" * 80)

df_all = pd.concat([
    pd.DataFrame(gpu_results),
    pd.DataFrame(ov_base_results),
    pd.DataFrame(ov_int8_results),
    pd.DataFrame(ov_int4_results)
], ignore_index=True)

df_all.to_csv(csv_file, index=False)
print(f"[+] Raw per-sentence benchmark metrics exported to '{csv_file}'.")

# Summary Averages
df_avg = df_all.groupby("engine", as_index=False)[["execution_time_ms", "ms_per_token"]].mean()

print("\n--- SUMMARY OF AVERAGE PERFORMANCE RESULTS ---")
for _, row in df_avg.iterrows():
    print(f" Engine: {row['engine']:<32} | Avg Latency: {row['execution_time_ms']:6.2f} ms | Cost: {row['ms_per_token']:5.3f} ms/token")
print("-" * 80)

sns.set_theme(style="darkgrid")
palette = ["#007ACC", "#8E44AD", "#27AE60", "#E67E22"]

# --- GRAPHIC 1: Summary Bar Charts ---
print("\n[+] Generating Graphic 1: Average Latency and Token Cost Comparison...")
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

ax1 = sns.barplot(data=df_avg, x="engine", y="ms_per_token", ax=axes[0], palette=palette)
axes[0].set_title("1. Average Cost per Token (ms / token)", fontsize=12, fontweight="bold")
axes[0].set_xlabel("Execution Engine", fontsize=10)
axes[0].set_ylabel("ms / token", fontsize=10)
axes[0].tick_params(axis='x', rotation=15)

for p in ax1.patches:
    h = p.get_height()
    if h > 0:
        ax1.annotate(f"{h:.3f} ms/token",
                     (p.get_x() + p.get_width() / 2., h),
                     ha='center', va='bottom', fontsize=10, fontweight='bold',
                     xytext=(0, 5), textcoords='offset points')

ax2 = sns.barplot(data=df_avg, x="engine", y="execution_time_ms", ax=axes[1], palette=palette)
axes[1].set_title("2. Average Sentence Latency (ms)", fontsize=12, fontweight="bold")
axes[1].set_xlabel("Execution Engine", fontsize=10)
axes[1].set_ylabel("Execution Time (ms)", fontsize=10)
axes[1].tick_params(axis='x', rotation=15)

for p in ax2.patches:
    h = p.get_height()
    if h > 0:
        ax2.annotate(f"{h:.2f} ms",
                     (p.get_x() + p.get_width() / 2., h),
                     ha='center', va='bottom', fontsize=10, fontweight='bold',
                     xytext=(0, 5), textcoords='offset points')

plt.tight_layout()
plt.savefig(dashboard_file, dpi=300)
plt.close()
print(f"    - Saved: '{dashboard_file}'")

# --- GRAPHIC 2: Latency Distribution Boxplot ---
print("[+] Generating Graphic 2: Latency Distribution Boxplot...")
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_all, x="engine", y="execution_time_ms", palette=palette, width=0.5)
plt.title("3. Latency Distribution per Engine Across All Sentences", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Execution Engine", fontsize=11)
plt.ylabel("Execution Time (ms)", fontsize=11)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(latency_box_file, dpi=300)
plt.close()
print(f"    - Saved: '{latency_box_file}'")

# --- GRAPHIC 3: Performance Scaling Scatter Plot ---
print("[+] Generating Graphic 3: Performance Scaling Scatter Plot with Regression...")
plt.figure(figsize=(11, 6))
ax_scatter = sns.scatterplot(
    data=df_all,
    x="token_count",
    y="execution_time_ms",
    hue="engine",
    style="engine",
    s=70,
    palette=palette
)

for idx, engine in enumerate(df_all["engine"].unique()):
    df_engine = df_all[df_all["engine"] == engine]
    sns.regplot(
        data=df_engine,
        x="token_count",
        y="execution_time_ms",
        scatter=False,
        color=palette[idx % len(palette)],
        ax=ax_scatter
    )

plt.title("4. Performance Scaling Behavior: Token Count vs Execution Latency", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Sentence Length (Token Count)", fontsize=11)
plt.ylabel("Execution Latency (ms)", fontsize=11)
plt.legend(title="Execution Engine", frameon=True)
plt.tight_layout()
plt.savefig(scaling_scatter_file, dpi=300)
plt.close()
print(f"    - Saved: '{scaling_scatter_file}'")

print("\n" + "=" * 80)
print("BENCHMARK EXECUTION COMPLETE & ALL GRAPHICS SAVED TO DISK!")
print("=" * 80)