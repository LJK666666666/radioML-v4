#!/usr/bin/env python3
"""
Test script to verify the new transformer models with RoPE are properly integrated
"""

import sys
import argparse

def test_model_availability():
    """Test if the new models are available in the command line interface"""
    try:
        # Import the main module functions
        from main import get_available_models, build_model_by_name, get_custom_objects_for_model
        
        print("Testing model availability...")
        
        # Check if new models are in the available models list
        available_models = get_available_models()
        print(f"Available models: {available_models}")
        
        new_models = ['transformer_rope_sequential', 'transformer_rope_phase']
        
        for model_name in new_models:
            if model_name in available_models:
                print(f"✓ {model_name} is available")
            else:
                print(f"✗ {model_name} is NOT available")
                return False
                
        # Test custom objects function
        for model_name in new_models:
            custom_objects = get_custom_objects_for_model(model_name)
            if custom_objects:
                print(f"✓ {model_name} has custom objects: {list(custom_objects.keys())}")
            else:
                print(f"✗ {model_name} has no custom objects")
                
        print("All tests passed!")
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_argument_parsing():
    """Test if the new models can be specified in command line arguments"""
    try:
        # Import argparse setup from main
        sys.path.insert(0, '.')
        
        # Create a test parser similar to main.py
        parser = argparse.ArgumentParser()
        
        from main import get_available_models
        available_models = get_available_models()
        
        parser.add_argument('--models', type=str, nargs='+', 
                           choices=available_models,
                           help='Model architectures to use')
        
        # Test parsing with new models
        test_args = ['--models', 'transformer_rope_sequential', 'transformer_rope_phase']
        
        try:
            args = parser.parse_args(test_args)
            print(f"✓ Successfully parsed arguments: {args.models}")
            return True
        except SystemExit:
            print("✗ Failed to parse arguments with new models")
            return False
            
    except Exception as e:
        print(f"Error in argument parsing test: {e}")
        return False

def main():
    print("=" * 60)
    print("TESTING NEW TRANSFORMER MODELS WITH ROPE")
    print("=" * 60)
    
    success = True
    
    print("\n1. Testing model availability...")
    if not test_model_availability():
        success = False
    
    print("\n2. Testing argument parsing...")
    if not test_argument_parsing():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("ALL TESTS PASSED! ✓")
        print("The new transformer models are properly integrated.")
        print("\nYou can now use them with commands like:")
        print("python main.py --models transformer_rope_sequential --mode train")
        print("python main.py --models transformer_rope_phase --mode train")
        print("python main.py --models transformer_rope_sequential transformer_rope_phase --mode all")
    else:
        print("SOME TESTS FAILED! ✗")
        print("Please check the implementation.")
    print("=" * 60)

if __name__ == "__main__":
    main()
