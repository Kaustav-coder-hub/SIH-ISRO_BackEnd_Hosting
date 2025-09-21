from flask import Flask, jsonify, request
from flask_cors import CORS
import os
from inference import analyze_location
from database import save_graph, get_saved_graphs, get_graph_by_id, delete_graph

app = Flask(__name__)

# Production CORS configuration
if os.environ.get('FLASK_ENV') == 'production':
    # Add your frontend URLs here when deployed
    CORS(app, origins=[
        "http://localhost:5173",  # Local development
        "https://your-frontend-domain.vercel.app",  # Add your actual frontend URL
        "https://your-frontend-domain.netlify.app"   # Add your actual frontend URL
    ])
else:
    CORS(app)  # Allow all origins in development

@app.route('/')
def hello():
    return jsonify({'message': 'API is running'})

@app.route('/api/status')
def status():
    return jsonify({'status': 'ok', 'version': '1.0'})

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        print("📡 Received analyze request")
        data = request.get_json()
        print(f"📊 Request data keys: {list(data.keys()) if data else 'No data'}")
        
        if not data:
            return jsonify({'error': 'No data provided', 'status': 'error'}), 400
            
        lat = data.get('lat')
        lon = data.get('lon')
        satellite_img = data.get('satellite_image')  # Base64 image from frontend
        infrared_img = data.get('infrared_image')    # Optional infrared image
        location_name = data.get('location_name', 'Location')
        
        print(f"📍 Coordinates: lat={lat}, lon={lon}")
        print(f"📍 Location name: {location_name}")
        print(f"🖼️ Images provided: satellite={satellite_img is not None}, infrared={infrared_img is not None}")
        
        # Validate required parameters
        if lat is None or lon is None:
            return jsonify({'error': 'Latitude and longitude required', 'status': 'error'}), 400
        
        # Validate coordinate ranges
        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            return jsonify({'error': 'Invalid latitude or longitude format', 'status': 'error'}), 400
            
        if not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
            return jsonify({'error': 'Latitude or longitude out of valid range', 'status': 'error'}), 400
        
        # Validate image data if provided
        if satellite_img and not isinstance(satellite_img, str):
            return jsonify({'error': 'Invalid satellite image format', 'status': 'error'}), 400
            
        if infrared_img and not isinstance(infrared_img, str):
            return jsonify({'error': 'Invalid infrared image format', 'status': 'error'}), 400
        
        # Use the inference module to analyze the location with captured images
        print("🔬 Starting analysis...")
        result = analyze_location(lat, lon, satellite_img, infrared_img)
        
        if 'error' in result:
            print(f"❌ Analysis error: {result['error']}")
            return jsonify({
                'error': result['error'],
                'status': 'analysis_error',
                'message': f"Analysis failed: {result['error']}"
            }), 500
        else:
            print("✅ Analysis completed successfully")
            # Ensure all values in result are JSON serializable
            return jsonify(result)
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Server error: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': error_msg, 
            'status': 'server_error',
            'message': f'Server error occurred: {error_msg}'
        }), 500

@app.route('/save-graph', methods=['POST'])
def save_analysis_graph():
    try:
        print("📡 Received save-graph request")
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided', 'success': False}), 400
            
        # Extract required fields
        lat = data.get('lat')
        lon = data.get('lon')
        rgb = data.get('rgb')
        infrared = data.get('infrared')
        
        # Extract optional fields
        location_name = data.get('location_name', 'Location')
        visualization = data.get('visualization')
        metrics = data.get('metrics')
        
        # Validate required fields
        if None in [lat, lon, rgb, infrared]:
            missing_fields = [field for field in ['lat', 'lon', 'rgb', 'infrared'] 
                             if data.get(field) is None]
            return jsonify({
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'success': False
            }), 400
            
        # Save to database
        result = save_graph(
            lat=lat,
            lon=lon,
            location_name=location_name,
            rgb=rgb,
            infrared=infrared,
            visualization=visualization,
            metrics=metrics
        )
        
        if result['success']:
            print(f"✅ Graph saved successfully with ID: {result.get('id')}")
            return jsonify(result)
        else:
            print(f"❌ Failed to save graph: {result.get('error')}")
            return jsonify(result), 500
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Server error in save-graph: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': error_msg,
            'success': False,
            'message': f'Server error occurred: {error_msg}'
        }), 500

@app.route('/saved-graphs', methods=['GET'])
def get_all_saved_graphs():
    try:
        print("📡 Received request for saved graphs")
        
        # Get all saved graphs from the database
        graphs = get_saved_graphs()
        
        print(f"✅ Retrieved {len(graphs)} saved graphs")
        return jsonify(graphs)
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Server error in getting saved graphs: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': error_msg,
            'success': False,
            'message': f'Server error occurred: {error_msg}'
        }), 500

@app.route('/saved-graphs/<int:graph_id>', methods=['GET'])
def get_saved_graph(graph_id):
    try:
        print(f"📡 Received request for graph {graph_id}")
        
        # Get the specific graph from the database
        graph = get_graph_by_id(graph_id)
        
        if graph:
            print(f"✅ Retrieved graph {graph_id}")
            return jsonify(graph)
        else:
            print(f"❌ Graph {graph_id} not found")
            return jsonify({
                'error': f'Graph {graph_id} not found',
                'success': False
            }), 404
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Server error in getting graph {graph_id}: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': error_msg,
            'success': False,
            'message': f'Server error occurred: {error_msg}'
        }), 500

@app.route('/saved-graphs/<int:graph_id>', methods=['DELETE'])
def delete_saved_graph(graph_id):
    try:
        print(f"📡 Received request to delete graph {graph_id}")
        
        # Delete the graph from the database
        result = delete_graph(graph_id)
        
        if result['success']:
            print(f"✅ Deleted graph {graph_id}")
            return jsonify(result)
        else:
            print(f"❌ Failed to delete graph {graph_id}: {result.get('message')}")
            if 'Graph not found' in result.get('message', ''):
                return jsonify(result), 404
            else:
                return jsonify(result), 500
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Server error in deleting graph {graph_id}: {error_msg}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': error_msg,
            'success': False,
            'message': f'Server error occurred: {error_msg}'
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    app.run(host='0.0.0.0', port=port, debug=debug)
