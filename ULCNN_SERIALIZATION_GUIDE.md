# ULCNN Model Serialization Guide

## Overview

This guide explains the ULCNN model serialization capabilities and how to handle model saving/loading in the RadioML classification framework.

## ✅ Current Status

**All ULCNN models now support proper serialization and loading:**

- ✅ **SCNN** - Simple CNN (104,395 parameters)
- ✅ **MCNet** - Multi-Channel Network (82,571 parameters)  
- ✅ **ULCNN** - Ultra-Lightweight Complex Neural Network (9,751 parameters)
- ✅ **PET** - Phase Enhancement Transformer (71,871 parameters)
- ✅ **MCLDNN** - Multi-Channel Lightweight Deep Neural Network (405,175 parameters)

## 🔧 Technical Implementation

### Custom Layers with Serialization Support

All custom layers now have proper `@register_keras_serializable` decorators:

- `ComplexConv1D` - Complex 1D convolution
- `ComplexBatchNormalization` - Complex batch normalization
- `ComplexDense` - Complex dense layer
- `ChannelShuffle` - Channel shuffle operation
- `DWConvMobile` - Depthwise separable mobile convolution
- `ChannelAttention` - Channel attention mechanism
- `TransposeLayer` - Tensor transposition (replaces Lambda)
- `ExtractChannelLayer` - Channel extraction (replaces Lambda)
- `TrigonometricLayer` - Trigonometric functions (replaces Lambda)

### Serialization Format

- **Recommended format**: `.keras` (native Keras 3 format)
- **Legacy support**: `.h5` (with some limitations for complex models)
- **Custom objects**: Required for models with complex layers

## 📖 Usage Examples

### Training and Saving Models

```python
# Train ULCNN models using main.py
python src/main.py --models ulcnn pet mcldnn --mode train --epochs 100

# Models are automatically saved in .keras format
# Location: output/models/ulcnn_model_gpr_augment.keras
```

### Loading Models for Evaluation

```python
# Evaluate ULCNN models using main.py
python src/main.py --models ulcnn pet mcldnn --mode evaluate

# Models are automatically loaded with proper custom_objects
```

### Manual Model Loading

```python
import tensorflow as tf
from keras.models import load_model
from src.main import get_custom_objects_for_model

# Load ULCNN model
custom_objects = get_custom_objects_for_model('ulcnn')
model = load_model('path/to/ulcnn_model.keras', custom_objects=custom_objects)

# Use the model
predictions = model.predict(X_test)
```

## 🚨 Important Notes

### Existing Models (Pre-Fix)

**Models saved before the serialization fixes contain Lambda layers and cannot be loaded.**

**Affected models:**
- Any ULCNN, PET, or MCLDNN models saved before this fix
- Models with Lambda layer errors when loading

**Solutions:**
1. **Retrain models** (recommended) - Use the updated codebase to train new models
2. **Use working models** - SCNN and MCNet models from before the fix still work

### Working Models (Post-Fix)

**All new models trained with the updated codebase work perfectly:**
- Full serialization support
- Consistent predictions after loading
- Compatible with evaluation pipeline

## 🧪 Testing

### Verify New Models Work

```bash
# Test that new models can be built, saved, and loaded
conda run -n ljk2 python test_ulcnn_integration.py
```

### Check Existing Models Status

```bash
# Check which existing models work with current codebase
conda run -n ljk2 python test_existing_models.py
```

### Comprehensive Serialization Test

```bash
# Run detailed serialization tests
conda run -n ljk2 python test_ulcnn_serialization.py
```

## 📋 Model Compatibility Matrix

| Model | Build | Save | Load | Evaluate | Status |
|-------|-------|------|------|----------|--------|
| SCNN | ✅ | ✅ | ✅ | ✅ | **Ready** |
| MCNet | ✅ | ✅ | ✅ | ✅ | **Ready** |
| ULCNN | ✅ | ✅ | ✅ | ✅ | **Ready** |
| PET | ✅ | ✅ | ✅ | ✅ | **Ready** |
| MCLDNN | ✅ | ✅ | ✅ | ✅ | **Ready** |

## 🎯 Recommendations

1. **For new training**: Use the current codebase - all models will work perfectly
2. **For existing models**: 
   - SCNN and MCNet models from before the fix still work
   - ULCNN, PET, and MCLDNN models need to be retrained
3. **For production**: Always use the `.keras` format with proper custom_objects

## 🔍 Troubleshooting

### Lambda Layer Errors

**Error**: `Exception encountered when calling Lambda.call()`

**Cause**: Model was saved with old Lambda-based architecture

**Solution**: Retrain the model with the updated codebase

### Custom Object Errors

**Error**: `Could not interpret initializer identifier: sqrt_init`

**Cause**: Missing custom_objects when loading

**Solution**: Use `get_custom_objects_for_model()` when loading

### Example Fix

```python
# Instead of this (will fail):
model = load_model('ulcnn_model.keras')

# Use this (will work):
from src.main import get_custom_objects_for_model
custom_objects = get_custom_objects_for_model('ulcnn')
model = load_model('ulcnn_model.keras', custom_objects=custom_objects)
```

## ✅ Verification

The serialization implementation has been verified through:

- ✅ Unit tests for all custom layers
- ✅ Integration tests with main.py pipeline
- ✅ Save/load consistency tests
- ✅ Prediction consistency verification
- ✅ Evaluation pipeline compatibility

**Result**: All ULCNN models now have robust serialization support compatible with the existing training and evaluation infrastructure.