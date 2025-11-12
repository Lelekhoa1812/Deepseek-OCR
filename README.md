---
title: OCR-VLs
emoji: 📝
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
python_version: "3.11.3"
pinned: true
short_description: 'DeepSeek Paddle olm dots Gemini | OCR to MD'
# Build optimizations for ZeroGPU deployment
# These settings help reduce build time and image size
hf_ignore:
  - "local/**"
  - "*.pdf"
  - "*.png"
  - "*.jpg"
  - "*.jpeg"
  - "__pycache__/**"
  - "*.pyc"
  - ".git/**"
  - "test/**"
  - "tests/**"
# Note: ZeroGPU Spaces are currently locked to Python 3.10.13
# This cannot be overridden even if runtime.txt specifies a different version
# olmOCR requires Python >= 3.11, so it will be disabled on ZeroGPU Spaces
---
