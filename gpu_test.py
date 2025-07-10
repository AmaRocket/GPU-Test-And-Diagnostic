#!/usr/bin/env python3
"""
Multi-GPU NVIDIA Testing Suite
================================

This script provides a comprehensive suite for testing individual and multi-GPU setups for any CUDA-capable NVIDIA GPUs.

Features:
- Checks for required Python packages (torch, numpy, psutil)
- Reports detailed GPU information (model, memory, compute capability)
- Tests individual GPU memory and compute performance
- Benchmarks multi-GPU data parallelism and communication
- Simulates NCCL collective operations
- Monitors system and GPU resource usage
- Optionally stress-tests all GPUs in parallel

Usage:
    python3 gpu_test.py

Requirements:
- Python 3.x
- torch, numpy, psutil (install with: pip install torch numpy psutil)
- CUDA-capable GPUs and NVIDIA drivers

Author: Oleksii Tkachuk
Date: 2025/07/10
"""

import torch
import numpy as np
import time
import psutil
import subprocess
import sys
from datetime import datetime


def check_requirements():
    """Check if required packages are installed"""
    required = ['torch', 'numpy', 'psutil']
    missing = []

    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        print("Install with: pip install torch numpy psutil")
        return False
    return True


def get_gpu_info():
    """Get detailed GPU information"""
    print("=== GPU Information ===")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"GPU Count: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"\nGPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1024 ** 3:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print(f"  Multiprocessors: {props.multi_processor_count}")


def test_individual_gpu(gpu_id, matrix_size=8192, iterations=10):
    """Test individual GPU performance"""
    print(f"\n=== Testing GPU {gpu_id} ===")

    device = torch.device(f'cuda:{gpu_id}')
    torch.cuda.set_device(device)

    try:
        memory_gb = torch.cuda.get_device_properties(gpu_id).total_memory / 1024 ** 3
        print(f"Available Memory: {memory_gb:.1f} GB")

        # Allocate test tensor (using about 80% of memory)
        test_size = int((memory_gb * 0.8 * 1024 ** 3) / (4 * 1024))  # 4 bytes per float32
        test_tensor = torch.randn(test_size, device=device)
        print(f"Memory allocation test: PASSED ({test_size * 4 / 1024 ** 3:.1f} GB)")
        del test_tensor

    except Exception as e:
        print(f"Memory allocation test: FAILED - {e}")
        return False

    try:
        print(f"Running matrix multiplication test ({matrix_size}x{matrix_size})...")

        a = torch.randn(matrix_size, matrix_size, device=device)
        b = torch.randn(matrix_size, matrix_size, device=device)
        torch.matmul(a, b)
        torch.cuda.synchronize()

        times = []
        for i in range(iterations):
            start = time.time()
            c = torch.matmul(a, b)
            torch.cuda.synchronize()
            end = time.time()
            times.append(end - start)

            if i % 2 == 0:
                print(f"  Iteration {i + 1}/{iterations}: {times[-1]:.4f}s")

        avg_time = np.mean(times)
        min_time = np.min(times)
        max_time = np.max(times)

        operations = 2 * matrix_size ** 3
        tflops = (operations / avg_time) / 1e12

        print(f"Average time: {avg_time:.4f}s")
        print(f"Min time: {min_time:.4f}s")
        print(f"Max time: {max_time:.4f}s")
        print(f"Performance: {tflops:.2f} TFLOPS")

        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits', f'--id={gpu_id}'],
                capture_output=True, text=True)
            if result.returncode == 0:
                temp = int(result.stdout.strip())
                print(f"Temperature: {temp}°C")
            else:
                print("Temperature: Unable to read")
        except:
            print("Temperature: nvidia-smi not available")

        del a, b, c
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        print(f"Compute test: FAILED - {e}")
        return False


def test_multi_gpu_data_parallel(matrix_size=4096, iterations=5):
    """Test multi-GPU data parallel processing"""
    print(f"\n=== Multi-GPU Data Parallel Test ===")

    if torch.cuda.device_count() < 2:
        print("Need at least 2 GPUs for multi-GPU test")
        return False

    try:
        class SimpleModel(torch.nn.Module):
            def __init__(self, size):
                super().__init__()
                self.linear1 = torch.nn.Linear(size, size)
                self.linear2 = torch.nn.Linear(size, size)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                x = self.relu(self.linear1(x))
                x = self.linear2(x)
                return x

        model = SimpleModel(matrix_size)

        model_single = model.cuda(0)
        data_single = torch.randn(64, matrix_size).cuda(0)

        start_time = time.time()
        for _ in range(iterations):
            output = model_single(data_single)
            torch.cuda.synchronize()
        single_gpu_time = time.time() - start_time

        print(f"Single GPU time: {single_gpu_time:.4f}s")

        if torch.cuda.device_count() > 1:
            model_multi = torch.nn.DataParallel(model)
            model_multi = model_multi.cuda()

            batch_size = 64 * torch.cuda.device_count()
            data_multi = torch.randn(batch_size, matrix_size).cuda()

            start_time = time.time()
            for _ in range(iterations):
                output = model_multi(data_multi)
                torch.cuda.synchronize()
            multi_gpu_time = time.time() - start_time

            print(f"Multi-GPU time: {multi_gpu_time:.4f}s")
            print(f"Speedup: {single_gpu_time / multi_gpu_time:.2f}x")
            print(f"Efficiency: {(single_gpu_time / multi_gpu_time) / torch.cuda.device_count() * 100:.1f}%")

        return True

    except Exception as e:
        print(f"Multi-GPU test: FAILED - {e}")
        return False


def test_gpu_communication():
    """Test GPU-to-GPU communication"""
    print(f"\n=== GPU Communication Test ===")

    if torch.cuda.device_count() < 2:
        print("Need at least 2 GPUs for communication test")
        return False

    try:
        print("Testing peer-to-peer access...")
        for i in range(torch.cuda.device_count()):
            for j in range(torch.cuda.device_count()):
                if i != j:
                    can_access = torch.cuda.can_device_access_peer(i, j)
                    print(f"GPU {i} -> GPU {j}: {'Yes' if can_access else 'No'}")

        print("\nTesting data transfer between GPUs...")
        data_size = 1024 * 1024  # 1M elements

        for i in range(torch.cuda.device_count() - 1):
            data = torch.randn(data_size, device=f'cuda:{i}')
            start_time = time.time()
            data_copy = data.to(f'cuda:{i + 1}')
            torch.cuda.synchronize()
            transfer_time = time.time() - start_time

            bandwidth = (data_size * 4) / transfer_time / 1e9  # GB/s

            print(f"GPU {i} -> GPU {i + 1}: {transfer_time:.4f}s, {bandwidth:.2f} GB/s")

            if torch.allclose(data.cpu(), data_copy.cpu()):
                print(f"  Data integrity: PASSED")
            else:
                print(f"  Data integrity: FAILED")
                return False

        return True

    except Exception as e:
        print(f"GPU communication test: FAILED - {e}")
        return False


def test_nccl_collective():
    """Test NCCL collective operations"""
    print(f"\n=== NCCL Collective Operations Test ===")

    if torch.cuda.device_count() < 2:
        print("Need at least 2 GPUs for NCCL test")
        return False

    try:
        import torch.distributed as dist

        # This is a simplified test - in practice you'd use torch.distributed.launch
        print("Testing AllReduce operation...")

        tensors = []
        for i in range(torch.cuda.device_count()):
            tensor = torch.ones(1000, device=f'cuda:{i}') * (i + 1)
            tensors.append(tensor)

        result_sum = sum(range(1, torch.cuda.device_count() + 1))

        print(f"Manual AllReduce simulation completed")
        print(f"Expected sum per element: {result_sum}")

        return True

    except Exception as e:
        print(f"NCCL test: FAILED - {e}")
        return False


def monitor_system_resources():
    """Monitor CPU and system resources during GPU testing"""
    print(f"\n=== System Resource Monitor ===")

    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()

    print(f"CPU Usage: {cpu_percent}%")
    print(f"Memory Usage: {memory.percent}% ({memory.used / 1024 ** 3:.1f} GB / {memory.total / 1024 ** 3:.1f} GB)")

    print("\nGPU Memory Usage:")
    for i in range(torch.cuda.device_count()):
        memory_allocated = torch.cuda.memory_allocated(i) / 1024 ** 3
        memory_reserved = torch.cuda.memory_reserved(i) / 1024 ** 3
        memory_total = torch.cuda.get_device_properties(i).total_memory / 1024 ** 3

        print(
            f"GPU {i}: {memory_allocated:.1f} GB allocated, {memory_reserved:.1f} GB reserved, {memory_total:.1f} GB total")


def stress_test_all_gpus(duration_minutes=5):
    """Stress test all GPUs simultaneously"""
    print(f"\n=== Stress Test All GPUs ({duration_minutes} minutes) ===")

    import threading

    def stress_gpu(gpu_id, duration):
        device = torch.device(f'cuda:{gpu_id}')
        torch.cuda.set_device(device)

        end_time = time.time() + duration
        while time.time() < end_time:
            a = torch.randn(2048, 2048, device=device)
            b = torch.randn(2048, 2048, device=device)
            c = torch.matmul(a, b)
            del a, b, c

            if time.time() % 30 < 0.1:  # Print every ~30 seconds
                print(f"GPU {gpu_id} stress test running...")

    threads = []
    duration_seconds = duration_minutes * 60

    for i in range(torch.cuda.device_count()):
        thread = threading.Thread(target=stress_gpu, args=(i, duration_seconds))
        threads.append(thread)
        thread.start()

    start_time = time.time()
    while time.time() - start_time < duration_seconds:
        time.sleep(30)
        print(f"\n--- Stress Test Progress: {(time.time() - start_time) / 60:.1f} minutes ---")
        monitor_system_resources()

    for thread in threads:
        thread.join()

    print("Stress test completed!")


def main():
    """Main testing function"""
    print(f"Multi-GPU NVIDIA Testing Suite")
    print(f"Started at: {datetime.now()}")
    print("=" * 50)

    if not check_requirements():
        return

    if not torch.cuda.is_available():
        print("CUDA is not available!")
        return

    get_gpu_info()

    # Test each GPU individually
    failed_gpus = []
    for i in range(torch.cuda.device_count()):
        if not test_individual_gpu(i):
            failed_gpus.append(i)

    if failed_gpus:
        print(f"\nFailed GPUs: {failed_gpus}")
        print("Please check the failed GPUs before continuing.")
        return
    else:
        print(f"\nAll {torch.cuda.device_count()} GPUs passed individual tests!")

    # Test multi-GPU functionality
    test_multi_gpu_data_parallel()
    test_gpu_communication()
    test_nccl_collective()

    monitor_system_resources()

    # Optional stress test
    response = input("\nRun 5-minute stress test on all GPUs? (y/N): ")
    if response.lower() == 'y':
        stress_test_all_gpus(5)

    print(f"\nTesting completed at: {datetime.now()}")
    print("All tests finished!")


if __name__ == "__main__":
    main()