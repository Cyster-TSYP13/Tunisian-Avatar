#!/usr/bin/env python3
"""
Check if all required dependencies are installed
(With TensorFlow workaround)
"""

import sys
import os

# Disable TensorFlow to avoid Keras 3 issue
os.environ['TRANSFORMERS_NO_TF'] = '1'
os.environ['SENTENCE_TRANSFORMERS_NO_TF'] = '1'

required_packages = {
    'sentence_transformers': 'sentence-transformers',
    'langchain': 'langchain',
    'langdetect': 'langdetect',
    'chromadb': 'chromadb',
    'requests': 'requests',
    'bs4': 'beautifulsoup4',
    'torch': 'torch',
    'tiktoken': 'tiktoken',
    'pypdf': 'pypdf'
}

missing = []
installed = []

for package, pip_name in required_packages.items():
    try:
        __import__(package)
        installed.append(pip_name)
        print(f"✅ {pip_name}")
    except ImportError:
        missing.append(pip_name)
        print(f"❌ {pip_name} - NOT INSTALLED")

print("\n" + "="*60)
if missing:
    print(f"\n⚠️  Missing {len(missing)} package(s). Install with:")
    print(f"\npip install {' '.join(missing)}")
else:
    print("\n✅ All required packages are installed!")
print("="*60)

sys.exit(0 if not missing else 1)
