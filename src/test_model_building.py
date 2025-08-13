#!/usr/bin/env python3
"""
Test script to verify that the new transformer models can be built successfully
"""

import numpy as np

def test_model_building():
    """Test building the new transformer models"""
    try:
        from model.transformer_model import (
            build_transformer_rope_sequential_model, 
            build_transformer_rope_phase_model,
            build_transformer_model
        )
        
        # Define test parameters
        input_shape = (2, 128)  # RadioML format: (I/Q channels, sequence_length)
        num_classes = 11  # Number of modulation types in RadioML2016.10a
        
        print("Testing model building...")
        
        # Test original transformer
        print("Building original transformer model...")
        model_orig = build_transformer_model(input_shape, num_classes)
        print(f"✓ Original transformer built successfully. Parameters: {model_orig.count_params()}")
        
        # Test RoPE sequential model
        print("Building RoPE sequential transformer model...")
        model_rope_seq = build_transformer_rope_sequential_model(input_shape, num_classes)
        print(f"✓ RoPE sequential transformer built successfully. Parameters: {model_rope_seq.count_params()}")
        
        # Test RoPE phase model
        print("Building RoPE phase transformer model...")
        model_rope_phase = build_transformer_rope_phase_model(input_shape, num_classes)
        print(f"✓ RoPE phase transformer built successfully. Parameters: {model_rope_phase.count_params()}")
        
        # Test model summaries
        print("\n" + "="*50)
        print("MODEL SUMMARIES")
        print("="*50)
        
        print("\n1. Original Transformer:")
        model_orig.summary()
        
        print("\n2. RoPE Sequential Transformer:")
        model_rope_seq.summary()
        
        print("\n3. RoPE Phase Transformer:")
        model_rope_phase.summary()
        
        # Test with dummy data
        print("\n" + "="*50)
        print("TESTING WITH DUMMY DATA")
        print("="*50)
        
        # Create dummy input data
        batch_size = 4
        dummy_input = np.random.randn(batch_size, *input_shape)
        
        print(f"Dummy input shape: {dummy_input.shape}")
        
        # Test predictions
        pred_orig = model_orig.predict(dummy_input, verbose=0)
        pred_rope_seq = model_rope_seq.predict(dummy_input, verbose=0)
        pred_rope_phase = model_rope_phase.predict(dummy_input, verbose=0)
        
        print(f"✓ Original transformer prediction shape: {pred_orig.shape}")
        print(f"✓ RoPE sequential prediction shape: {pred_rope_seq.shape}")
        print(f"✓ RoPE phase prediction shape: {pred_rope_phase.shape}")
        
        # Verify output shapes are correct
        expected_shape = (batch_size, num_classes)
        assert pred_orig.shape == expected_shape, f"Wrong output shape: {pred_orig.shape}"
        assert pred_rope_seq.shape == expected_shape, f"Wrong output shape: {pred_rope_seq.shape}"
        assert pred_rope_phase.shape == expected_shape, f"Wrong output shape: {pred_rope_phase.shape}"
        
        print("✓ All output shapes are correct")
        
        # Verify outputs are valid probabilities
        assert np.allclose(np.sum(pred_orig, axis=1), 1.0), "Original model outputs are not valid probabilities"
        assert np.allclose(np.sum(pred_rope_seq, axis=1), 1.0), "RoPE sequential outputs are not valid probabilities"
        assert np.allclose(np.sum(pred_rope_phase, axis=1), 1.0), "RoPE phase outputs are not valid probabilities"
        
        print("✓ All outputs are valid probability distributions")
        
        return True
        
    except Exception as e:
        print(f"Error in model building test: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("TESTING TRANSFORMER MODEL BUILDING")
    print("=" * 60)
    
    if test_model_building():
        print("\n" + "=" * 60)
        print("ALL MODEL BUILDING TESTS PASSED! ✓")
        print("The new transformer models are working correctly.")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("MODEL BUILDING TESTS FAILED! ✗")
        print("Please check the implementation.")
        print("=" * 60)

if __name__ == "__main__":
    main()
