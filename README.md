# Multi-GPU NVIDIA Testing Suite & GPU Diagnostic Tool

## Overview
This repository provides two Python scripts for working with NVIDIA GPUs:

- **gpu_test.py**: A comprehensive suite for testing, benchmarking, and stress-testing single and multi-GPU setups (any CUDA-capable NVIDIA GPU).
- **gpu_diagnostic.py**: A diagnostic and setup tool to identify and help fix common CUDA/GPU issues, including environment, driver, and PyTorch checks.

---

## Features
### gpu_test.py
- Checks for required Python packages (`torch`, `numpy`, `psutil`)
- Reports detailed GPU information (model, memory, compute capability)
- Tests individual GPU memory and compute performance
- Benchmarks multi-GPU data parallelism and communication
- Simulates NCCL collective operations
- Monitors system and GPU resource usage
- Optionally stress-tests all GPUs in parallel

### gpu_diagnostic.py
- Checks NVIDIA driver installation and version
- Verifies CUDA toolkit and runtime libraries
- Checks environment variables relevant to CUDA
- Diagnoses PyTorch CUDA support and installation
- Checks GPU device permissions and kernel modules
- Provides system information (OS, kernel, secure boot)
- Generates a shell script to fix common issues automatically

---

## Requirements
- Python 3.x
- NVIDIA GPU with CUDA support
- NVIDIA drivers and CUDA toolkit installed
- Python packages: `torch`, `numpy`, `psutil`

---

## Installation
1. **Clone this repository:**
   ```bash
   git clone <repo-url>
   cd <repo-directory>
   ```
2. **Install required Python packages:**
   ```bash
   pip install torch numpy psutil
   ```
   - For PyTorch with CUDA, see: https://pytorch.org/get-started/locally/

---

## Usage
### 1. GPU Diagnostic Tool
Run the diagnostic tool to check your system and generate a fix script if issues are found:
```bash
python3 gpu_diagnostic.py
```
- If issues are detected, a `fix_gpu_setup.sh` script will be generated with recommended actions.

### 2. Multi-GPU Testing Suite
Run the test suite to benchmark and stress-test your GPUs:
```bash
python3 gpu_test.py
```
- Follow the prompts for optional stress testing.

---

## Example Output
**gpu_test.py**
```
Multi-GPU NVIDIA Testing Suite
Started at: 2024-06-07 12:00:00
==================================================
=== GPU Information ===
CUDA Available: True
CUDA Version: 12.1
PyTorch Version: 2.1.0
GPU Count: 2
...
All 2 GPUs passed individual tests!
...
Testing completed at: 2024-06-07 12:10:00
All tests finished!
```

**gpu_diagnostic.py**
```
GPU Diagnostic and Setup Tool
==================================================
=== NVIDIA Driver Check ===
✅ nvidia-smi is working
Driver Version: 535.54.03
...
❌ Issues found:
  - PyTorch CUDA support
💡 RECOMMENDED ACTIONS:
1. Run the generated fix script
2. Reboot your system
...
```

---

## Troubleshooting
- If you encounter missing packages, install them with:
  ```bash
  pip install torch numpy psutil
  ```
- For CUDA or driver issues, run `gpu_diagnostic.py` and follow the generated recommendations.
- For PyTorch CUDA support, ensure you have installed the correct CUDA-enabled version of PyTorch.

---

## License
[MIT License](LICENSE)
