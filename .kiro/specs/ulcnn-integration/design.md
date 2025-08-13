# Design Document

## Overview

The ULCNN integration involves adapting five different neural network models (MCLDNN, SCNN, MCNet, PET, ULCNN) from the existing ULCNN project into the current src structure. The integration requires creating a compatible complexnn module, adapting the models to work with the existing data pipeline, and ensuring compatibility with current TensorFlow/Keras versions.

## Architecture

### High-Level Architecture

```
src/
├── model/
│   ├── mcldnn_model.py          # MCLDNN model definition
│   ├── scnn_model.py            # SCNN model definition  
│   ├── mcnet_model.py           # MCNet model definition
│   ├── pet_model.py             # PET model definition
│   ├── ulcnn_model.py           # ULCNN model definition
│   ├── complexnn/               # Complex neural network module
│   │   ├── __init__.py
│   │   ├── conv.py             # Complex convolution layers
│   │   ├── bn.py               # Complex batch normalization
│   │   ├── dense.py            # Complex dense layers
│   │   └── utils.py            # Utility functions
│   └── ...
├── models.py                    # Updated to include ULCNN models
└── main.py                      # Updated to support ULCNN models
```

### Integration Strategy

1. **Module Adaptation**: Create a simplified complexnn module compatible with current TensorFlow/Keras versions
2. **Model Refactoring**: Adapt each ULCNN model to follow the existing function signature pattern
3. **Data Pipeline Integration**: Ensure ULCNN models work with the existing data preprocessing pipeline
4. **Testing Framework**: Integrate models into the existing training/evaluation framework

## Components and Interfaces

### ComplexNN Module

The complexnn module will be adapted from the original ULCNN project with the following key components:

#### ComplexConv1D
- **Purpose**: 1D complex convolution layer
- **Interface**: `ComplexConv1D(filters, kernel_size, padding='same', activation=None, **kwargs)`
- **Compatibility**: Adapted to work with current Keras API

#### ComplexBatchNormalization  
- **Purpose**: Batch normalization for complex-valued data
- **Interface**: `ComplexBatchNormalization(**kwargs)`
- **Implementation**: Simplified version focusing on core functionality

#### ComplexDense
- **Purpose**: Dense layer for complex-valued data
- **Interface**: `ComplexDense(units, activation=None, **kwargs)`
- **Implementation**: Complex matrix multiplication with proper weight initialization

### ULCNN Model Builders

Each model will be implemented in its own file with a builder function following the existing pattern:

#### File Structure and Function Signatures
```python
# src/model/mcldnn_model.py
def build_mcldnn_model(input_shape, num_classes):
    """Build MCLDNN model"""
    
# src/model/scnn_model.py
def build_scnn_model(input_shape, num_classes):
    """Build SCNN model"""
    
# src/model/mcnet_model.py
def build_mcnet_model(input_shape, num_classes):
    """Build MCNet model"""
    
# src/model/pet_model.py
def build_pet_model(input_shape, num_classes):
    """Build PET model"""
    
# src/model/ulcnn_model.py
def build_ulcnn_model(input_shape, num_classes):
    """Build ULCNN model"""
```

#### Model Adaptations

**MCLDNN (Multi-Channel LDNN)**
- Multi-input architecture with I/Q channels and separate I, Q inputs
- Combines 2D and 1D convolutions with LSTM layers
- Adaptation: Simplify to single input, maintain core architecture

**SCNN (Separable CNN)**
- Uses separable convolutions for efficiency
- Simple architecture with batch normalization
- Adaptation: Direct port with minimal changes

**MCNet (Multi-scale CNN)**
- Complex multi-scale architecture with skip connections
- Uses custom blocks (pre_block, m_block, m_block_p)
- Adaptation: Implement custom blocks as separate functions

**PET (Phase Enhancement Transformer)**
- Uses trigonometric transformations on I/Q data
- Combines spatial and temporal feature extraction
- Adaptation: Implement custom trigonometric layers

**ULCNN (Ultra-Lightweight CNN)**
- Uses complex convolutions with channel shuffle
- Implements channel attention mechanism
- Adaptation: Port complex layers and attention mechanism

## Data Models

### Input Data Format
- **Shape**: (batch_size, 2, 128) - I/Q channels with 128 samples
- **Type**: float32
- **Range**: Normalized signal values

### Model-Specific Data Handling

#### MCLDNN Data Preparation
```python
# Original: Multiple inputs [x_train, x_train_I, x_train_Q]
# Adapted: Single input with internal splitting
def prepare_mcldnn_data(X):
    # Split internally within model
    return X
```

#### Standard Models (SCNN, MCNet, ULCNN)
```python
# Direct use of existing data format
def prepare_standard_data(X):
    return X  # (batch_size, 2, 128)
```

#### PET Data Preparation
```python
# Requires expanded dimensions for trigonometric operations
def prepare_pet_data(X):
    return np.expand_dims(X, axis=3)  # (batch_size, 2, 128, 1)
```

## Error Handling

### Import Error Handling
```python
try:
    from model.complexnn import ComplexConv1D, ComplexBatchNormalization, ComplexDense
except ImportError as e:
    print(f"Warning: Complex layers not available: {e}")
    # Fallback to regular layers or skip complex models
```

### Model Building Error Handling
```python
def build_ulcnn_model(input_shape, num_classes):
    try:
        # Model building code
        return model
    except Exception as e:
        print(f"Error building ULCNN model: {e}")
        raise
```

### Training Error Handling
- Graceful degradation if complex operations fail
- Clear error messages for debugging
- Fallback options where possible

## Testing Strategy

### Unit Testing
1. **Complex Layer Testing**
   - Test each complex layer individually
   - Verify mathematical correctness of complex operations
   - Test serialization/deserialization

2. **Model Building Testing**
   - Test each model builder function
   - Verify model compilation
   - Test with dummy data

### Integration Testing
1. **Data Pipeline Testing**
   - Test data flow through each model
   - Verify input/output shapes
   - Test with real RadioML data

2. **Training Testing**
   - Test single epoch training for each model
   - Verify gradient flow
   - Test model saving/loading

### Performance Testing
1. **Memory Usage**
   - Monitor memory consumption during training
   - Compare with baseline models

2. **Training Speed**
   - Measure training time per epoch
   - Compare computational efficiency

## Implementation Plan

### Phase 1: Core Infrastructure
1. Create complexnn module structure
2. Implement basic complex layers (Conv1D, BatchNorm, Dense)
3. Test complex layers individually

### Phase 2: Model Implementation
1. Implement SCNN (simplest model first)
2. Implement ULCNN (core complex model)
3. Implement MCNet (most complex architecture)
4. Implement MCLDNN and PET (multi-input models)

### Phase 3: Integration
1. Update models.py to include ULCNN models
2. Update main.py to support new models
3. Test integration with existing pipeline

### Phase 4: Validation
1. Run single epoch tests for all models
2. Verify model saving/loading
3. Performance benchmarking

## Compatibility Considerations

### TensorFlow/Keras Version Compatibility
- Target: TensorFlow 2.x with Keras 3.x
- Remove deprecated APIs (CuDNNLSTM, etc.)
- Update layer implementations for current API

### Data Format Compatibility
- Ensure models work with existing (2, 128) input format
- Handle any necessary data transformations internally

### Serialization Compatibility
- Use @register_keras_serializable for custom layers
- Ensure models can be saved and loaded properly

## Risk Mitigation

### Technical Risks
1. **Complex Layer Compatibility**: Implement fallback mechanisms
2. **Memory Issues**: Monitor and optimize memory usage
3. **Training Instability**: Implement gradient clipping and proper initialization

### Integration Risks
1. **API Changes**: Maintain backward compatibility where possible
2. **Data Pipeline Issues**: Thorough testing with real data
3. **Performance Degradation**: Benchmark against existing models