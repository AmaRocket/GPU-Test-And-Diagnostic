#!/usr/bin/env python3
"""
GPU Diagnostic and Setup Tool
============================

This script diagnoses and helps fix common CUDA/GPU issues on Linux systems (with some checks also working on macOS and Windows).

Features:
- Checks NVIDIA driver installation and version
- Verifies CUDA toolkit and runtime libraries
- Checks environment variables relevant to CUDA
- Diagnoses PyTorch CUDA support and installation
- Checks GPU device permissions and kernel modules
- Provides system information (OS, kernel, secure boot)
- Generates a shell script to fix common issues automatically

Usage:
    python3 gpu_diagnostic.py

If issues are found, a fix script (fix_gpu_setup.sh) will be generated with recommended actions.

Requirements:
- Python 3.x
- Some checks require root privileges (for running fix script)
- Linux system (most checks; some may work on macOS/Windows with adaptation)

Author: Oleksii Tkachuk
Date: 2025/07/10
"""

import subprocess
import sys
import os
import re
from pathlib import Path


def run_command(cmd, capture_output=True):
    """Run a command and return result"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=capture_output, text=True)
        return result.returncode, result.stdout, result.stderr
    except Exception as e:
        return -1, "", str(e)


def check_nvidia_driver():
    """Check NVIDIA driver installation"""
    print("=== NVIDIA Driver Check ===")

    # Check nvidia-smi
    ret, stdout, stderr = run_command("nvidia-smi")
    if ret != 0:
        print("❌ nvidia-smi not found or not working")
        print(f"Error: {stderr}")
        return False

    print("✅ nvidia-smi is working")
    print(stdout)

    # Extract driver version
    driver_match = re.search(r'Driver Version: (\d+\.\d+\.\d+)', stdout)
    if driver_match:
        driver_version = driver_match.group(1)
        print(f"Driver Version: {driver_version}")

        # Check if driver is recent enough
        major_version = int(driver_version.split('.')[0])
        if major_version >= 470:  # Minimum for CUDA 11.0+
            print("✅ Driver version is compatible with modern CUDA")
        else:
            print("⚠️ Driver version might be too old")

    return True


def check_cuda_installation():
    """Check CUDA toolkit installation"""
    print("\n=== CUDA Installation Check ===")

    # Check nvcc
    ret, stdout, stderr = run_command("nvcc --version")
    if ret != 0:
        print("❌ nvcc not found - CUDA toolkit not installed or not in PATH")
        print("Install CUDA toolkit:")
        print(
            "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb")
        print("sudo dpkg -i cuda-keyring_1.0-1_all.deb")
        print("sudo apt-get update")
        print("sudo apt-get -y install cuda")
        return False

    print("✅ CUDA toolkit found")
    print(stdout)

    # Check CUDA libraries
    cuda_paths = [
        "/usr/local/cuda/lib64",
        "/usr/lib/x86_64-linux-gnu",
        "/usr/local/cuda-12.0/lib64",
        "/usr/local/cuda-11.8/lib64"
    ]

    cuda_lib_found = False
    for path in cuda_paths:
        if os.path.exists(f"{path}/libcudart.so"):
            print(f"✅ CUDA runtime library found at {path}")
            cuda_lib_found = True
            break

    if not cuda_lib_found:
        print("❌ CUDA runtime library not found")
        return False

    return True


def check_environment_variables():
    """Check important environment variables"""
    print("\n=== Environment Variables Check ===")

    # Check CUDA_HOME
    cuda_home = os.environ.get('CUDA_HOME')
    if cuda_home:
        print(f"✅ CUDA_HOME: {cuda_home}")
    else:
        print("⚠️ CUDA_HOME not set")
        # Try to find CUDA installation
        possible_paths = ["/usr/local/cuda", "/usr/local/cuda-12.0", "/usr/local/cuda-11.8"]
        for path in possible_paths:
            if os.path.exists(path):
                print(f"💡 Suggestion: export CUDA_HOME={path}")
                break

    # Check PATH
    path = os.environ.get('PATH', '')
    if '/usr/local/cuda/bin' in path or 'cuda' in path:
        print("✅ CUDA binaries in PATH")
    else:
        print("⚠️ CUDA binaries not in PATH")
        print("💡 Add to ~/.bashrc: export PATH=/usr/local/cuda/bin:$PATH")

    # Check LD_LIBRARY_PATH
    ld_path = os.environ.get('LD_LIBRARY_PATH', '')
    if 'cuda' in ld_path:
        print("✅ CUDA libraries in LD_LIBRARY_PATH")
    else:
        print("⚠️ CUDA libraries not in LD_LIBRARY_PATH")
        print("💡 Add to ~/.bashrc: export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH")

    # Check CUDA_VISIBLE_DEVICES
    cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_devices:
        print(f"ℹ️ CUDA_VISIBLE_DEVICES: {cuda_devices}")
    else:
        print("ℹ️ CUDA_VISIBLE_DEVICES not set (will use all GPUs)")


def check_pytorch_installation():
    """Check PyTorch CUDA support"""
    print("\n=== PyTorch CUDA Support Check ===")

    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")

        # Check if CUDA is available
        if torch.cuda.is_available():
            print("✅ PyTorch CUDA is available")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU count: {torch.cuda.device_count()}")

            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}")
        else:
            print("❌ PyTorch CUDA is NOT available")
            print("This is likely the main issue!")

            # Check if PyTorch was installed with CUDA support
            print(f"PyTorch CUDA version: {torch.version.cuda}")
            if torch.version.cuda is None:
                print("❌ PyTorch was installed without CUDA support")
                print("💡 Reinstall PyTorch with CUDA:")
                print("pip3 uninstall torch torchvision torchaudio")
                print("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
                return False
            else:
                print("✅ PyTorch has CUDA support, but can't access GPUs")
                return False

    except ImportError:
        print("❌ PyTorch not installed")
        print("Install PyTorch with CUDA support:")
        print("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return False

    return True


def check_gpu_permissions():
    """Check GPU device permissions"""
    print("\n=== GPU Device Permissions Check ===")

    # Check /dev/nvidia* devices
    nvidia_devices = list(Path('/dev').glob('nvidia*'))
    if not nvidia_devices:
        print("❌ No /dev/nvidia* devices found")
        print("Try: sudo nvidia-modprobe")
        return False

    print(f"✅ Found {len(nvidia_devices)} NVIDIA devices")
    for device in nvidia_devices:
        stat = device.stat()
        permissions = oct(stat.st_mode)[-3:]
        print(f"  {device}: permissions {permissions}")

        # Check if current user can access
        if os.access(device, os.R_OK | os.W_OK):
            print(f"  ✅ {device} is accessible")
        else:
            print(f"  ❌ {device} is not accessible")
            print(f"  💡 Try: sudo chmod 666 {device}")
            print(f"  💡 Or add user to nvidia group: sudo usermod -a -G nvidia $USER")


def check_nvidia_modprobe():
    """Check if nvidia kernel modules are loaded"""
    print("\n=== NVIDIA Kernel Modules Check ===")

    ret, stdout, stderr = run_command("lsmod | grep nvidia")
    if ret != 0 or not stdout:
        print("❌ NVIDIA kernel modules not loaded")
        print("Try:")
        print("sudo modprobe nvidia")
        print("sudo nvidia-modprobe")
        return False

    print("✅ NVIDIA kernel modules loaded:")
    print(stdout)
    return True


def check_system_info():
    """Check system information"""
    print("\n=== System Information ===")

    # OS info
    ret, stdout, stderr = run_command("lsb_release -a")
    if ret == 0:
        print("OS Information:")
        print(stdout)

    # Kernel version
    ret, stdout, stderr = run_command("uname -r")
    if ret == 0:
        print(f"Kernel: {stdout.strip()}")

    # Check for secure boot
    if os.path.exists("/sys/firmware/efi"):
        ret, stdout, stderr = run_command("mokutil --sb-state")
        if ret == 0:
            print(f"Secure Boot: {stdout.strip()}")
            if "enabled" in stdout.lower():
                print("⚠️ Secure Boot is enabled - this can cause NVIDIA driver issues")
                print("💡 Consider disabling Secure Boot in BIOS")


def generate_fix_script():
    """Generate a script to fix common issues"""
    print("\n=== Generating Fix Script ===")

    script_content = '''#!/bin/bash
# GPU Setup Fix Script
# Run this script to fix common CUDA/GPU issues

echo "=== GPU Setup Fix Script ==="

# Update system
echo "Updating system packages..."
sudo apt update

# Install build essentials
echo "Installing build essentials..."
sudo apt install -y build-essential dkms

# Load NVIDIA modules
echo "Loading NVIDIA kernel modules..."
sudo modprobe nvidia
sudo nvidia-modprobe

# Fix device permissions
echo "Fixing device permissions..."
sudo chmod 666 /dev/nvidia*
sudo chmod 666 /dev/nvidiactl

# Add user to nvidia group
echo "Adding user to nvidia group..."
sudo groupadd -f nvidia
sudo usermod -a -G nvidia $USER

# Set environment variables
echo "Setting up environment variables..."
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc

# Reload environment
source ~/.bashrc

# Test nvidia-smi
echo "Testing nvidia-smi..."
nvidia-smi

echo "=== Fix script completed ==="
echo "You may need to:"
echo "1. Reboot the system"
echo "2. Log out and log back in"
echo "3. Reinstall PyTorch with CUDA support"
'''

    with open('fix_gpu_setup.sh', 'w') as f:
        f.write(script_content)

    print("✅ Fix script generated: fix_gpu_setup.sh")
    print("Run with: chmod +x fix_gpu_setup.sh && ./fix_gpu_setup.sh")


def main():
    """Main diagnostic function"""
    print("GPU Diagnostic and Setup Tool")
    print("=" * 50)

    issues_found = []

    # Run all checks
    if not check_nvidia_driver():
        issues_found.append("NVIDIA driver")

    if not check_cuda_installation():
        issues_found.append("CUDA installation")

    check_environment_variables()

    if not check_pytorch_installation():
        issues_found.append("PyTorch CUDA support")

    if not check_gpu_permissions():
        issues_found.append("GPU permissions")

    if not check_nvidia_modprobe():
        issues_found.append("NVIDIA kernel modules")

    check_system_info()

    # Summary
    print("\n" + "=" * 50)
    print("DIAGNOSTIC SUMMARY")
    print("=" * 50)

    if issues_found:
        print("❌ Issues found:")
        for issue in issues_found:
            print(f"  - {issue}")

        print("\n💡 RECOMMENDED ACTIONS:")
        print("1. Run the generated fix script")
        print("2. Reboot your system")
        print("3. Reinstall PyTorch with CUDA support")
        print("4. Test again")

        generate_fix_script()
    else:
        print("✅ No major issues found!")
        print("Your GPU setup appears to be working correctly.")


if __name__ == "__main__":
    main()