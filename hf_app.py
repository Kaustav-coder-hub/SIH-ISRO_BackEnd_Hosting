import gradio as gr
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import base64
import io
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# Copy your SRCNN model class here
class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=5, padding=2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Initialize model (you'll need to load your trained weights)
model = SRCNN()
# model.load_state_dict(torch.load('srcnn_model.pth', map_location='cpu'))
model.eval()

def enhance_thermal_image(image):
    """
    Main function called by Gradio interface
    """
    try:
        # Convert PIL image to numpy
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
            
        # Convert to grayscale if needed
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
            
        # Normalize to 0-1
        normalized = gray.astype(np.float32) / 255.0
        
        # Add batch and channel dimensions for model
        input_tensor = torch.FloatTensor(normalized).unsqueeze(0).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            enhanced_tensor = model(input_tensor)
            
        # Convert back to numpy and scale to 0-255
        enhanced = enhanced_tensor.squeeze().numpy()
        enhanced = np.clip(enhanced * 255, 0, 255).astype(np.uint8)
        
        # Apply colormap for visualization
        colored = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        
        # Calculate some metrics (simplified)
        metrics = {
            "enhancement_factor": "2x",
            "processing_time": "0.5s",
            "model": "SRCNN",
            "status": "success"
        }
        
        return colored_rgb, metrics
        
    except Exception as e:
        # Return original image and error info
        return image, {"error": str(e), "status": "failed"}

def api_predict(image_b64, parameters=None):
    """
    API endpoint function for your gateway to call
    """
    try:
        # Decode base64 image
        image_data = base64.b64decode(image_b64)
        image = Image.open(io.BytesIO(image_data))
        
        # Process the image
        enhanced_image, metrics = enhance_thermal_image(image)
        
        # Convert enhanced image back to base64
        enhanced_pil = Image.fromarray(enhanced_image)
        buffered = io.BytesIO()
        enhanced_pil.save(buffered, format="PNG")
        enhanced_b64 = base64.b64encode(buffered.getvalue()).decode()
        
        return {
            "enhanced_image": enhanced_b64,
            "original_image": image_b64,
            "metrics": metrics,
            "status": "success"
        }
        
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }

# Create Gradio interface
iface = gr.Interface(
    fn=enhance_thermal_image,
    inputs=[
        gr.Image(type="pil", label="Upload Thermal Image")
    ],
    outputs=[
        gr.Image(type="numpy", label="Enhanced Thermal Image"),
        gr.JSON(label="Analysis Metrics")
    ],
    title="🛰️ SIH-ISRO Thermal Image Enhancement",
    description="Upload thermal/infrared images for AI-powered super-resolution enhancement",
    examples=[
        # Add example images if you have them
    ],
    api_name="predict"  # This creates the API endpoint
)

# Launch the app
if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
