#!/usr/bin/env python3
"""
Simple test script for the inference module
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))

from inference import analyze_location

def test_analyze_location():
    """Test the analyze_location function"""
    print("🧪 Testing analyze_location function...")
    
    try:
        # Test with basic coordinates (Kolkata)
        result = analyze_location(22.5726, 88.3639)
        
        if 'error' in result:
            print(f"❌ Error: {result['error']}")
            return False
        else:
            print("✅ Basic test passed!")
            print(f"Result keys: {list(result.keys())}")
            return True
            
    except Exception as e:
        print(f"❌ Exception during test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_analyze_location()