---
title: SIH-ISRO Thermal Enhancement
emoji: 🛰️
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
license: mit
---

# SIH-ISRO Thermal Image Enhancement Service

This is a machine learning service for enhancing thermal/infrared images using Super-Resolution CNN (SRCNN).

## Features
- AI-powered thermal image enhancement
- Super-resolution processing 
- Real-time inference via API
- Compatible with various thermal image formats

## API Usage

### Endpoint
```
POST https://your-username-sih-isro-ml-service.hf.space/api/predict
```

### Request Format
```json
{
  "data": [
    "base64_encoded_image_string",
    {} 
  ]
}
```

### Response Format
```json
{
  "data": [
    "enhanced_image_array",
    {
      "enhancement_factor": "2x",
      "processing_time": "0.5s", 
      "model": "SRCNN",
      "status": "success"
    }
  ]
}
```

## Model Details
- Architecture: Super-Resolution CNN (SRCNN)
- Input: Thermal/Infrared images
- Output: Enhanced high-resolution images
- Framework: PyTorch
