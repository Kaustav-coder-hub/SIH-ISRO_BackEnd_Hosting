# SIH-ISRO Backend

Flask-based API server for the Advanced Thermal Satellite Image Super-Resolution System. This backend provides deep learning inference, image processing, and data management capabilities for thermal satellite imagery analysis.

## 🌟 Features

### Core Capabilities
- **Thermal Super-Resolution**: SRCNN and Professional Thermal SR models
- **Deep Learning Inference**: PyTorch-based neural networks
- **Image Processing**: OpenCV and Scikit-Image algorithms
- **RESTful API**: Flask endpoints for frontend integration
- **Data Persistence**: SQLite database for analysis storage
- **Comprehensive Metrics**: PSNR, SSIM, MS-SSIM evaluation

### Advanced Algorithms
- **SRCNN (Super-Resolution CNN)**: Dong et al. (ECCV 2014) implementation
- **Professional Thermal SR**: Multi-scale residual blocks for thermal data
- **Guided Filtering**: He et al. (ECCV 2010) edge-preserving method
- **Physics-Based Correction**: Thermal statistics preservation
- **Multi-Scale Analysis**: SIFT-based alignment and evaluation

## 🏗️ Project Structure

```
backend/
├── app.py                   # Flask application entry point
├── inference.py             # Deep learning inference engine
├── database.py              # SQLite database operations
├── routes.py                # API endpoint definitions
├── visualization.py         # Image processing utilities
├── test_inference.py        # Unit tests for inference
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Getting Started

### Prerequisites

- **Python 3.8+** (3.9+ recommended)
- **pip** package manager
- **CUDA-capable GPU** (optional, for faster processing)
- **Git** for version control

### Installation

1. **Navigate to backend directory**:
   ```bash
   cd backend
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   
   # Activate virtual environment
   # On Linux/macOS:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**:
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import cv2; print(f'OpenCV: {cv2.__version__}')"
   ```

### Running the Server

1. **Start Flask development server**:
   ```bash
   python app.py
   ```

2. **Server will start on**:
   - Development: `http://localhost:5000`
   - Health check: `http://localhost:5000/api/status`

3. **For production deployment**:
   ```bash
   # Using Gunicorn
   pip install gunicorn
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

## 🌐 Deployment Options

### 1. Traditional Server Deployment

#### Using Gunicorn + Nginx
```bash
# Install Gunicorn
pip install gunicorn

# Create systemd service file
sudo nano /etc/systemd/system/sih-isro-backend.service
```

Service file content:
```ini
[Unit]
Description=SIH-ISRO Backend
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/path/to/backend
Environment="PATH=/path/to/backend/venv/bin"
ExecStart=/path/to/backend/venv/bin/gunicorn -w 4 -b 127.0.0.1:5000 app:app
Restart=always

[Install]
WantedBy=multi-user.target
```

Nginx configuration:
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 2. Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Run the application
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:
```bash
# Build Docker image
docker build -t sih-isro-backend .

# Run container
docker run -p 5000:5000 sih-isro-backend
```

### 3. Cloud Platform Deployment

#### AWS EC2
```bash
# Launch EC2 instance with Ubuntu
# Install Python and dependencies
sudo apt update
sudo apt install python3-pip python3-venv nginx

# Clone repository and setup
git clone <your-repo>
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Setup systemd service and nginx
```

#### Google Cloud Platform
```bash
# Create VM instance
gcloud compute instances create sih-isro-backend \
    --image-family=ubuntu-2004-lts \
    --image-project=ubuntu-os-cloud \
    --machine-type=n1-standard-2

# SSH and setup similar to EC2
```

#### Heroku
Create `Procfile`:
```
web: gunicorn app:app
```

Deploy:
```bash
# Install Heroku CLI
# Login and create app
heroku login
heroku create sih-isro-backend

# Deploy
git push heroku main
```

### 4. Container Orchestration

#### Docker Compose
Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  backend:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
    volumes:
      - ./data:/app/data
    restart: always
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - backend
    restart: always
```

## ⚙️ Configuration

### Environment Variables

Create `.env` file:
```env
FLASK_ENV=production
FLASK_DEBUG=False
DATABASE_URL=sqlite:///thermal_analysis.db
CORS_ORIGINS=http://localhost:5173,https://your-frontend-domain.com
PYTORCH_DEVICE=cuda  # or cpu
MAX_IMAGE_SIZE=2048
MODEL_CACHE_DIR=./models
```

### Production Settings

Update `app.py` for production:
```python
import os
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)

# Production configuration
if os.environ.get('FLASK_ENV') == 'production':
    app.config['DEBUG'] = False
    app.config['TESTING'] = False
    CORS(app, origins=os.environ.get('CORS_ORIGINS', '').split(','))
else:
    app.config['DEBUG'] = True
    CORS(app)  # Allow all origins in development
```

## 🔧 Technology Stack

### Core Framework
- **Flask 2.3.3**: Lightweight web framework
- **Flask-CORS**: Cross-origin resource sharing
- **Gunicorn**: WSGI HTTP server for production

### Deep Learning & Computer Vision
- **PyTorch 2.1.0**: Deep learning framework
- **Torchvision 0.16.0**: Computer vision utilities
- **OpenCV 4.8.1**: Image processing library
- **Scikit-Image 0.21.0**: Advanced image algorithms
- **NumPy 1.24.3**: Numerical computing
- **SciPy 1.11.3**: Scientific computing

### Visualization & Analysis
- **Matplotlib 3.7.2**: Plotting and visualization
- **Requests 2.31.0**: HTTP client library

### Database
- **SQLite**: Lightweight embedded database
- **Custom database.py**: Analysis storage utilities

## 📊 API Endpoints

### Core Analysis
```
POST /analyze
Content-Type: application/json

{
  "lat": 28.6139,
  "lon": 77.2090,
  "satellite_image": "data:image/jpeg;base64,/9j/4AAQ...",
  "infrared_image": "data:image/jpeg;base64,/9j/4AAQ...",
  "location_name": "New Delhi"
}

Response:
{
  "infrared": "data:image/png;base64,iVBORw0KGgo...",
  "infrared_original": "data:image/png;base64,iVBORw0KGgo...",
  "rgb": "data:image/png;base64,iVBORw0KGgo...",
  "visualization": "data:image/png;base64,iVBORw0KGgo...",
  "metrics": {
    "quantitative_metrics": {
      "psnr": 32.45,
      "ssim": 0.8923,
      "rmse_kelvin": 2.34
    },
    "qualitative_metrics": {
      "edge_sharpness": 0.756,
      "edge_enhancement_factor": 1.23
    }
  },
  "coordinates": {"lat": 28.6139, "lon": 77.2090},
  "status": "success"
}
```

### Data Management
```
GET /api/saved-graphs
Response: [
  {
    "id": 1,
    "location_name": "Delhi",
    "timestamp": "2025-09-21T10:30:00Z",
    "metrics": {...}
  }
]

POST /api/save-graph
Content-Type: application/json
{
  "location_name": "Delhi Analysis",
  "metrics": {...},
  "visualization_data": "base64..."
}

DELETE /api/graphs/{id}
Response: {"message": "Graph deleted successfully"}
```

### Health Check
```
GET /api/status
Response: {
  "status": "ok",
  "version": "1.0",
  "models_loaded": true,
  "gpu_available": true
}
```

## 🧪 Testing

### Run Unit Tests
```bash
# Run all tests
python -m pytest test_inference.py -v

# Run specific test
python -m pytest test_inference.py::test_srcnn_model -v

# Run with coverage
pip install pytest-cov
python -m pytest --cov=inference test_inference.py
```

### Manual Testing
```bash
# Test model loading
python -c "from inference import analyze_location; print('Models loaded successfully')"

# Test API endpoint
curl -X POST http://localhost:5000/api/status
```

## 🔍 Deep Learning Models

### SRCNN Architecture
```python
class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
```

### Professional Thermal SR
- **Multi-scale residual blocks**: Enhanced detail preservation
- **Thermal-specific optimizations**: Physics-based corrections
- **Edge-preserving filters**: Guided filtering integration

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   ```python
   # In inference.py, set device to CPU
   device = 'cpu'  # Force CPU usage
   ```

2. **OpenCV import errors**:
   ```bash
   # Install system dependencies
   sudo apt-get install libglib2.0-0 libsm6 libxext6 libxrender-dev
   ```

3. **PyTorch installation issues**:
   ```bash
   # Install specific PyTorch version
   pip install torch==2.1.0+cpu torchvision==0.16.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
   ```

4. **Memory issues with large images**:
   ```python
   # Resize images before processing
   MAX_SIZE = 1024
   if img.shape[0] > MAX_SIZE:
       img = cv2.resize(img, (MAX_SIZE, MAX_SIZE))
   ```

### Performance Optimization

- **Use GPU**: Enable CUDA for faster inference
- **Image preprocessing**: Resize large images
- **Model caching**: Load models once at startup
- **Batch processing**: Process multiple images together

### Logging

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 📈 Performance Metrics

### Benchmarks
- **SRCNN inference**: ~2-5 seconds per image (GPU)
- **Complete analysis**: ~10-15 seconds per location
- **Memory usage**: ~2-4GB RAM typical
- **GPU memory**: ~1-2GB VRAM for processing

### Scalability
- **Concurrent requests**: 4-8 with Gunicorn
- **Image size limits**: 2048x2048 pixels recommended
- **Database**: SQLite for up to 1M records

## 📞 Support

For backend-specific issues:
- Check Flask logs for errors
- Verify PyTorch and OpenCV installation
- Monitor system resources (CPU, GPU, memory)
- Test individual model components

## 🤝 Contributing

1. Follow PEP 8 Python style guidelines
2. Add unit tests for new features
3. Document API changes
4. Test with different image sizes and formats
5. Optimize for performance and memory usage

---

**Backend for SIH-ISRO Thermal Analysis System**
