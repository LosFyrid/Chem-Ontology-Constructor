#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Requirements and setup check for consistency analysis.
"""

import sys
import subprocess
import importlib

# Required packages
REQUIRED_PACKAGES = [
    ('numpy', 'numpy'),
    ('pandas', 'pandas'),
    ('scipy', 'scipy'),
    ('matplotlib', 'matplotlib'),
    ('seaborn', 'seaborn'),
]

def check_and_install_packages():
    """Check if required packages are installed, install if missing."""
    missing_packages = []
    
    print("Checking required packages...")
    
    for package_name, import_name in REQUIRED_PACKAGES:
        try:
            importlib.import_module(import_name)
            print(f"  + {package_name}")
        except ImportError:
            print(f"  - {package_name} (missing)")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\nInstalling missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
                print(f"  + Installed {package}")
            except subprocess.CalledProcessError as e:
                print(f"  - Failed to install {package}: {e}")
                return False
    
    print("\nAll required packages are available!")
    return True

if __name__ == "__main__":
    if check_and_install_packages():
        print("Setup complete. You can now run the consistency analysis.")
    else:
        print("Setup failed. Please install missing packages manually.")
        sys.exit(1)