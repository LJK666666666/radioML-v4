# Enhanced Training Logging System for RadioML Signal Classification

This enhanced logging system provides detailed epoch-by-epoch tracking of neural network training progress for the RadioML signal classification project.

## Features

### 1. Detailed Epoch-by-Epoch Logging
- **Training & Validation Metrics**: Accuracy and loss for each epoch
- **Timing Information**: Per-epoch training time in seconds
- **Learning Rate Tracking**: Current learning rate at each epoch
- **Timestamps**: ISO format timestamps for each epoch

### 2. Multiple Output Formats
- **CSV Files**: Machine-readable format for analysis and plotting
- **JSON Files**: Structured format with complete training metadata
- **Text Summaries**: Human-readable training summaries
- **Console Output**: Real-time progress display

### 3. Comprehensive Reporting
- **Individual Model Reports**: Detailed analysis for each trained model
- **Comparative Analysis**: Cross-model performance comparison
- **Overfitting Detection**: Automatic identification of potential overfitting
- **Best Model Recommendations**: Performance-based model selection

## File Structure

When training is completed, the following files are generated:

```
models/
├── logs/                                    # Detailed logging directory
│   ├── cnn1d_model_detailed_log.csv       # CSV format logs
│   ├── cnn1d_model_detailed_log.json      # JSON format logs
│   ├── cnn2d_model_detailed_log.csv
│   ├── cnn2d_model_detailed_log.json
│   └── ...                                 # Logs for each model
├── cnn1d_model_training_summary.txt        # Individual model summaries
├── cnn2d_model_training_summary.txt
├── comprehensive_training_report.txt       # Cross-model comparison
├── cnn1d_history.png                      # Training plots
└── ...                                     # Model files and plots
```

## CSV Log Format

The CSV logs contain the following columns:

| Column | Description |
|--------|-------------|
| `epoch` | Epoch number (1-indexed) |
| `train_loss` | Training set loss |
| `train_accuracy` | Training set accuracy |
| `val_loss` | Validation set loss |
| `val_accuracy` | Validation set accuracy |
| `epoch_time_seconds` | Time taken for this epoch |
| `learning_rate` | Current learning rate |
| `timestamp` | ISO format timestamp |

## JSON Log Format

The JSON logs provide a structured view of the complete training process:

```json
{
  "model_name": "cnn1d_model",
  "training_start_time": "2025-05-28T10:30:00.123456",
  "total_training_time": 1234.56,
  "epochs": [
    {
      "epoch": 1,
      "train_loss": 2.3456,
      "train_accuracy": 0.1234,
      "val_loss": 2.4567,
      "val_accuracy": 0.1345,
      "epoch_time_seconds": 12.34,
      "learning_rate": 0.001,
      "timestamp": "2025-05-28T10:30:12.345678"
    }
    // ... more epochs
  ]
}
```

## Usage

### Basic Usage (Default Behavior)
The enhanced logging is enabled by default in the `train_model()` function:

```python
history = train_model(
    model, 
    X_train, y_train, 
    X_val, y_val, 
    model_path
)
```

### Disabling Detailed Logging
To disable detailed logging and use only basic Keras logging:

```python
history = train_model(
    model, 
    X_train, y_train, 
    X_val, y_val, 
    model_path,
    detailed_logging=False
)
```

### Custom Logging Directory
The logging callback automatically creates logs in a `logs/` subdirectory of the model save path. For a model saved to `../models/cnn1d_model.keras`, logs will be saved to `../models/logs/`.

## Console Output

During training, you'll see enhanced console output like this:

```
================================================================================
Starting detailed logging for cnn1d_model
Log files will be saved to:
  CSV: ../models/logs/cnn1d_model_detailed_log.csv
  JSON: ../models/logs/cnn1d_model_detailed_log.json
================================================================================

Epoch 1 - Starting...

Epoch 1 Results:
  Time: 12.34s
  Train - Loss: 2.3456, Accuracy: 0.1234
  Val   - Loss: 2.4567, Accuracy: 0.1345
  Learning Rate: 0.001
  ------------------------------------------------------------

Epoch 2 - Starting...
...
```

## Analysis and Visualization

### Using CSV Data
The CSV files can be easily loaded into analysis tools:

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load training log
df = pd.read_csv('models/logs/cnn1d_model_detailed_log.csv')

# Plot training progress
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(df['epoch'], df['train_accuracy'], label='Train')
plt.plot(df['epoch'], df['val_accuracy'], label='Validation')
plt.title('Accuracy Progress')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(df['epoch'], df['train_loss'], label='Train')
plt.plot(df['epoch'], df['val_loss'], label='Validation')
plt.title('Loss Progress')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(df['epoch'], df['epoch_time_seconds'])
plt.title('Epoch Training Time')

plt.tight_layout()
plt.show()
```

### Performance Analysis
The comprehensive training report provides automatic analysis:

- **Best Model Identification**: Automatically identifies the best-performing model
- **Overfitting Detection**: Flags models with large train/validation gaps
- **Training Efficiency**: Analyzes convergence patterns
- **Recommendations**: Provides actionable insights for model improvement

## Demo Script

Run the demo script to see the logging system in action:

```bash
cd /home/test/2/2.5fixed_parameter_setting/radioML-v2
python demo_enhanced_logging.py
```

This will create a simple model and demonstrate all logging features with dummy data.

## Integration with Existing Code

The enhanced logging system is fully integrated with the existing RadioML training pipeline:

1. **Automatic Integration**: No changes needed to existing training calls
2. **Backward Compatible**: Existing code continues to work unchanged
3. **Configurable**: Can be enabled/disabled as needed
4. **Non-Intrusive**: Doesn't interfere with existing callbacks or training logic

## Files Modified/Added

### New Files:
- `src/model/detailed_logging_callback.py` - Main logging callback implementation
- `demo_enhanced_logging.py` - Demonstration script
- `ENHANCED_LOGGING_README.md` - This documentation

### Modified Files:
- `src/models.py` - Added import for detailed logging callback
- `src/train.py` - Enhanced train_model function and added reporting functions

All changes maintain backward compatibility with existing code.
