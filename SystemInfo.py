import platform
import os
import json
import subprocess
import psutil

def collect_system_info():
    info = {
        "os": {},
        "cpu": {},
        "ram": {},
        "gpu": []
    }

    # ---------------------------------------------------------
    # 1. OS INFORMATION
    # ---------------------------------------------------------
    info["os"] = {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "architecture": platform.machine(),
        "pointer": platform.architecture()[0],
        "hostname": platform.node()
    }

    # ---------------------------------------------------------
    # 2. CPU INFORMATION
    # ---------------------------------------------------------
    cpu_model = platform.processor()
    if platform.system() == "Windows":
        try:
            cmd = "wmic cpu get name"
            wmic_output = subprocess.check_output(cmd, shell=True).decode().split("\n")
            if len(wmic_output) > 1 and wmic_output[1].strip():
                cpu_model = wmic_output[1].strip()
        except Exception:
            pass

    freq = psutil.cpu_freq()
    info["cpu"] = {
        "model": cpu_model,
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "max_frequency_mhz": round(freq.max, 2) if freq else None,
        "current_frequency_mhz": round(freq.current, 2) if freq else None
    }

    # ---------------------------------------------------------
    # 3. RAM INFORMATION
    # ---------------------------------------------------------
    ram = psutil.virtual_memory()
    info["ram"] = {
        "total_gb": round(ram.total / (1024 ** 3), 2),
        "available_gb": round(ram.available / (1024 ** 3), 2),
        "used_gb": round(ram.used / (1024 ** 3), 2),
        "usage_percent": ram.percent
    }

    # ---------------------------------------------------------
    # 4. GPU INFORMATION
    # ---------------------------------------------------------
    gpu_detected = False

    # Attempt 1: PyTorch Detection
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                prop = torch.cuda.get_device_properties(i)
                info["gpu"].append({
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "vram_total_gb": round(prop.total_memory / (1024 ** 3), 2),
                    "detection_source": "PyTorch (CUDA)"
                })
            gpu_detected = True
    except ImportError:
        pass

    # Attempt 2: nvidia-smi Fallback
    if not gpu_detected:
        try:
            cmd = "nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv,noheader,nounits"
            output = subprocess.check_output(cmd, shell=True).decode().strip()
            for line in output.splitlines():
                gpu_idx, name, vram_mb, driver = [x.strip() for x in line.split(",")]
                info["gpu"].append({
                    "index": int(gpu_idx),
                    "name": name,
                    "vram_total_gb": round(float(vram_mb) / 1024, 2),
                    "driver_version": driver,
                    "detection_source": "nvidia-smi"
                })
            gpu_detected = True
        except Exception:
            pass

    # Attempt 3: Windows WMI Fallback (AMD / Intel integrated GPUs)
    if not gpu_detected and platform.system() == "Windows":
        try:
            cmd = "wmic path win32_videocard get name,adapterram"
            output = subprocess.check_output(cmd, shell=True).decode().strip()
            lines = [line.strip() for line in output.splitlines() if line.strip()]
            if len(lines) > 1:
                for idx, line in enumerate(lines[1:]):
                    parts = line.split()
                    gpu_name = " ".join(parts[:-1]) if len(parts) > 1 else line
                    info["gpu"].append({
                        "index": idx,
                        "name": gpu_name,
                        "vram_total_gb": None,
                        "detection_source": "Windows WMI"
                    })
        except Exception:
            pass

    return info

if __name__ == "__main__":
    json_filename = "SystemInfo.json"
    print("[+] Collecting hardware and operating system specifications...")
    
    sys_info = collect_system_info()
    
    with open(json_filename, "w", encoding="utf-8") as f:
        json.dump(sys_info, f, indent=4)
        
    print(f"[+] Successfully exported system info to '{os.path.abspath(json_filename)}'.")