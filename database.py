import sqlite3
import json
import time
from pathlib import Path

DB_PATH = Path(__file__).parent / 'db' / 'isro_data.db'

def init_db():
    """Initialize the database and create necessary tables if they don't exist."""
    try:
        # Ensure the directory exists
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        # Create saved_graphs table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS saved_graphs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            lat REAL NOT NULL,
            lon REAL NOT NULL,
            location_name TEXT,
            rgb TEXT,
            infrared TEXT,
            visualization TEXT,
            metrics TEXT,
            timestamp INTEGER
        )
        ''')
        
        conn.commit()
        print(f"✅ Database initialized at {DB_PATH}")
        return True
    except Exception as e:
        print(f"❌ Database initialization error: {str(e)}")
        return False
    finally:
        if 'conn' in locals():
            conn.close()

def save_graph(lat, lon, location_name, rgb, infrared, visualization=None, metrics=None):
    """Save a graph analysis to the database.
    
    Args:
        lat: Latitude of the location
        lon: Longitude of the location
        location_name: Name of the location
        rgb: Base64 encoded RGB image
        infrared: Base64 encoded infrared image
        visualization: Base64 encoded visualization image (optional)
        metrics: Dictionary of metrics data (optional)
    
    Returns:
        Dictionary with success status and message
    """
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        # Convert metrics to JSON string if provided
        metrics_json = json.dumps(metrics) if metrics else None
        
        # Insert the graph data
        cursor.execute('''
        INSERT INTO saved_graphs (lat, lon, location_name, rgb, infrared, visualization, metrics, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            lat, 
            lon, 
            location_name, 
            rgb, 
            infrared, 
            visualization, 
            metrics_json, 
            int(time.time())
        ))
        
        graph_id = cursor.lastrowid
        conn.commit()
        
        return {
            'success': True,
            'id': graph_id,
            'message': 'Graph saved successfully'
        }
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        print(f"❌ Error saving graph: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'message': 'Failed to save graph'
        }
    finally:
        if 'conn' in locals():
            conn.close()

def get_saved_graphs():
    """Retrieve all saved graphs from the database.
    
    Returns:
        List of dictionaries containing graph data
    """
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row  # Return rows as dictionaries
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT * FROM saved_graphs 
        ORDER BY timestamp DESC
        ''')
        
        rows = cursor.fetchall()
        results = []
        
        for row in rows:
            row_dict = dict(row)
            
            # Parse metrics JSON if it exists
            if row_dict['metrics']:
                try:
                    row_dict['metrics'] = json.loads(row_dict['metrics'])
                except:
                    row_dict['metrics'] = {}
            
            results.append(row_dict)
        
        return results
    except Exception as e:
        print(f"❌ Error retrieving saved graphs: {str(e)}")
        return []
    finally:
        if 'conn' in locals():
            conn.close()

def get_graph_by_id(graph_id):
    """Retrieve a specific saved graph by ID.
    
    Args:
        graph_id: The ID of the graph to retrieve
    
    Returns:
        Dictionary containing graph data or None if not found
    """
    try:
        conn = sqlite3.connect(str(DB_PATH))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM saved_graphs WHERE id = ?', (graph_id,))
        row = cursor.fetchone()
        
        if not row:
            return None
        
        result = dict(row)
        
        # Parse metrics JSON if it exists
        if result['metrics']:
            try:
                result['metrics'] = json.loads(result['metrics'])
            except:
                result['metrics'] = {}
        
        return result
    except Exception as e:
        print(f"❌ Error retrieving graph {graph_id}: {str(e)}")
        return None
    finally:
        if 'conn' in locals():
            conn.close()

def delete_graph(graph_id):
    """Delete a saved graph by ID.
    
    Args:
        graph_id: The ID of the graph to delete
    
    Returns:
        Dictionary with success status and message
    """
    try:
        conn = sqlite3.connect(str(DB_PATH))
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM saved_graphs WHERE id = ?', (graph_id,))
        
        if cursor.rowcount > 0:
            conn.commit()
            return {
                'success': True,
                'message': f'Graph {graph_id} deleted successfully'
            }
        else:
            return {
                'success': False,
                'message': f'Graph {graph_id} not found'
            }
    except Exception as e:
        if 'conn' in locals():
            conn.rollback()
        print(f"❌ Error deleting graph {graph_id}: {str(e)}")
        return {
            'success': False,
            'error': str(e),
            'message': f'Failed to delete graph {graph_id}'
        }
    finally:
        if 'conn' in locals():
            conn.close()

# Initialize the database when this module is imported
init_db()
