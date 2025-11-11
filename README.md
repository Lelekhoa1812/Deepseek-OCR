---
title: OCR-VLs
emoji: 📝
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
python_version: "3.11.0"
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
---
