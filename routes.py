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