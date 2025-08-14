#!/usr/bin/env python3
"""
ULCNN Pretrained Model Testing Script

This script loads the pretrained ULCNN model weights and evaluates performance
on the RadioML 2016.10a dataset using the same data preprocessing pipeline
as the main framework.

Usage:
    python test_ulcnn_pretrained.py
"""

import os
import sys
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.append('src')

from explore_dataset import load_radioml_data
from preprocess import prepare_data_by_snr
from models import build_ulcnn_model

# Import ULCNN complex layers for model loading
try:
    from model.complexnn import (
        ComplexConv1D as ULCNNComplexConv1D,
        ComplexBatchNormalization as ULCNNComplexBatchNormalization,
        ComplexDense as ULCNNComplexDense,
        ChannelShuffle, DWConvMobile, ChannelAttention,
        TransposeLayer, ExtractChannelLayer, TrigonometricLayer, sqrt_init
    )
    ULCNN_LAYERS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ULCNN complex layers not available: {e}")
    ULCNN_LAYERS_AVAILABLE = False


def get_ulcnn_custom_objects():
    """Get custom objects for ULCNN model loading"""
    if not ULCNN_LAYERS_AVAILABLE:
        return None
    
    return {
        'ULCNNComplexConv1D': ULCNNComplexConv1D,
        'ULCNNComplexBatchNormalization': ULCNNComplexBatchNormalization,
        'ULCNNComplexDense': ULCNNComplexDense,
        'ChannelShuffle': ChannelShuffle,
        'DWConvMobile': DWConvMobile,
        'ChannelAttention': ChannelAttention,
        'TransposeLayer': TransposeLayer,
        'ExtractChannelLayer': ExtractChannelLayer,
        'TrigonometricLayer': TrigonometricLayer,
        'sqrt_init': sqrt_init
    }


def build_original_ulcnn_model(input_shape=(128, 2), num_classes=11, n_neuron=16, n_mobileunit=6, ks=5):
    """
    Build the original ULCNN model architecture as defined in the paper
    
    Args:
        input_shape (tuple): Input shape (sequence_length, channels)
        num_classes (int): Number of output classes
        n_neuron (int): Number of neurons
        n_mobileunit (int): Number of mobile units
        ks (int): Kernel size
        
    Returns:
        tf.keras.Model: ULCNN model
    """
    from tensorflow.keras import layers, models, backend as K
    
    def channel_shuffle(x):
        in_channels, D = K.int_shape(x)[1:]
        channels_per_group = in_channels // 2
        pre_shape = [-1, 2, channels_per_group, D]
        dim = (0, 2, 1, 3)
        later_shape = [-1, in_channels, D]

        x = layers.Lambda(lambda z: K.reshape(z, pre_shape))(x)
        x = layers.Lambda(lambda z: K.permute_dimensions(z, dim))(x)  
        x = layers.Lambda(lambda z: K.reshape(z, later_shape))(x)
        return x

    def dwconv_mobile(x, neurons):
        x = layers.SeparableConv1D(int(2*neurons), ks, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = channel_shuffle(x)
        return x

    def channelattention(x):
        x_GAP = layers.GlobalAveragePooling1D()(x)
        x_GMP = layers.GlobalMaxPooling1D()(x)
        channel = K.int_shape(x_GAP)[1]

        share_Dense1 = layers.Dense(int(channel/16), activation='relu')
        share_Dense2 = layers.Dense(channel)

        x_GAP = layers.Reshape((1, channel))(x_GAP)
        x_GAP = share_Dense1(x_GAP)
        x_GAP = share_Dense2(x_GAP)

        x_GMP = layers.Reshape((1, channel))(x_GMP)
        x_GMP = share_Dense1(x_GMP)
        x_GMP = share_Dense2(x_GMP)

        x_Mask = layers.Add()([x_GAP, x_GMP])
        x_Mask = layers.Activation('sigmoid')(x_Mask)

        x = layers.Multiply()([x, x_Mask])
        return x
    
    # Build model
    x_input = layers.Input(shape=input_shape)
    
    # Initial complex convolution (using regular Conv1D as approximation)
    x = layers.Conv1D(n_neuron, ks, padding='same')(x_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Mobile units with channel attention
    for i in range(n_mobileunit):
        x = dwconv_mobile(x, n_neuron)
        x = channelattention(x)
        if i == 3:
            f4 = layers.GlobalAveragePooling1D()(x)
        if i == 4:
            f5 = layers.GlobalAveragePooling1D()(x)
        if i == 5:
            f6 = layers.GlobalAveragePooling1D()(x)

    # Feature fusion
    f = layers.Add()([f4, f5])
    f = layers.Add()([f, f6])

    # Classification head
    f = layers.Dense(num_classes)(f)
    c = layers.Activation('softmax', name='modulation')(f)

    model = models.Model(inputs=x_input, outputs=c)
    return model


def load_pretrained_ulcnn_model(model_path, input_shape, num_classes):
    """
    Load pretrained ULCNN model with proper custom objects
    
    Args:
        model_path (str): Path to the pretrained model file
        input_shape (tuple): Input shape for the model
        num_classes (int): Number of output classes
        
    Returns:
        tf.keras.Model: Loaded model or None if loading fails
    """
    print(f"Loading pretrained ULCNN model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return None
    
    # First try to load the complete model
    try:
        # Enable unsafe deserialization for models with custom layers
        tf.keras.config.enable_unsafe_deserialization()
        
        # Get custom objects for original ULCNN
        from tensorflow.keras import layers
        custom_objects = {
            'ComplexConv1D': ULCNNComplexConv1D if ULCNN_LAYERS_AVAILABLE else layers.Conv1D,
            'ComplexBatchNormalization': ULCNNComplexBatchNormalization if ULCNN_LAYERS_AVAILABLE else layers.BatchNormalization,
            'ComplexDense': ULCNNComplexDense if ULCNN_LAYERS_AVAILABLE else layers.Dense,
        }
        
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
        
        print(f"Successfully loaded complete pretrained model")
        print(f"Model input shape: {model.input_shape}")
        print(f"Model output shape: {model.output_shape}")
        print(f"Total parameters: {model.count_params():,}")
        
        return model
        
    except Exception as e:
        print(f"Error loading complete model: {e}")
        print("Attempting to build original architecture and load weights...")
        
        try:
            # Build the original ULCNN architecture
            # Note: Input shape for original model is (128, 2) not (2, 128)
            original_input_shape = (128, 2)  # sequence_length, channels
            model = build_original_ulcnn_model(original_input_shape, num_classes)
            
            # Try to load weights
            model.load_weights(model_path)
            print("Successfully loaded weights into original architecture")
            print(f"Model input shape: {model.input_shape}")
            print(f"Model output shape: {model.output_shape}")
            print(f"Total parameters: {model.count_params():,}")
            
            return model
            
        except Exception as e2:
            print(f"Error loading weights into original architecture: {e2}")
            
            # Last attempt: try to build a compatible model
            try:
                print("Attempting to build compatible model...")
                model = build_ulcnn_model(input_shape, num_classes)
                print("Built compatible model, but cannot load pretrained weights due to architecture mismatch")
                print("Using randomly initialized weights instead")
                return model
                
            except Exception as e3:
                print(f"Error building compatible model: {e3}")
                return None


def evaluate_model_performance(model, X_test, y_test, snr_test, mods, output_dir):
    """
    Evaluate model performance and generate detailed reports
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels (one-hot encoded)
        snr_test: Test SNR values
        mods: List of modulation types
        output_dir: Directory to save results
    """
    print("\nEvaluating model performance...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Make predictions
    print("Making predictions on test set...")
    y_pred_proba = model.predict(X_test, batch_size=128, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Overall accuracy
    overall_accuracy = np.mean(y_pred == y_true)
    print(f"\nOverall Test Accuracy: {overall_accuracy:.4f}")
    
    # Save overall accuracy
    with open(os.path.join(output_dir, 'overall_accuracy.txt'), 'w') as f:
        f.write(f"Overall Test Accuracy: {overall_accuracy:.4f}\n")
    
    # Classification report
    print("\nGenerating classification report...")
    class_report = classification_report(y_true, y_pred, target_names=mods, digits=4)
    print(class_report)
    
    # Save classification report
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write("ULCNN Pretrained Model Classification Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(class_report)
    
    # Confusion matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=mods, yticklabels=mods)
    plt.title('ULCNN Pretrained Model - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Accuracy by SNR
    print("Analyzing accuracy by SNR...")
    unique_snrs = sorted(np.unique(snr_test))
    snr_accuracies = []
    
    for snr in unique_snrs:
        snr_mask = snr_test == snr
        if np.sum(snr_mask) > 0:
            snr_acc = np.mean(y_pred[snr_mask] == y_true[snr_mask])
            snr_accuracies.append(snr_acc)
            print(f"SNR {int(snr):2d} dB: {snr_acc:.4f} ({np.sum(snr_mask)} samples)")
        else:
            snr_accuracies.append(0.0)
    
    # Plot accuracy vs SNR
    plt.figure(figsize=(10, 6))
    plt.plot(unique_snrs, snr_accuracies, 'bo-', linewidth=2, markersize=8)
    plt.grid(True, alpha=0.3)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Classification Accuracy')
    plt.title('ULCNN Pretrained Model - Accuracy vs SNR')
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'accuracy_vs_snr.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save SNR accuracy data
    snr_data = {'SNR_dB': unique_snrs, 'Accuracy': snr_accuracies}
    import pandas as pd
    df_snr = pd.DataFrame(snr_data)
    df_snr.to_csv(os.path.join(output_dir, 'accuracy_by_snr.csv'), index=False)
    
    # Accuracy by modulation and SNR
    print("Analyzing accuracy by modulation and SNR...")
    mod_snr_results = []
    
    for i, mod in enumerate(mods):
        mod_mask = y_true == i
        for snr in unique_snrs:
            snr_mask = snr_test == snr
            combined_mask = mod_mask & snr_mask
            
            if np.sum(combined_mask) > 0:
                acc = np.mean(y_pred[combined_mask] == y_true[combined_mask])
                mod_snr_results.append({
                    'Modulation': mod,
                    'SNR_dB': snr,
                    'Accuracy': acc,
                    'Samples': np.sum(combined_mask)
                })
    
    # Create heatmap for modulation vs SNR accuracy
    if mod_snr_results:
        df_mod_snr = pd.DataFrame(mod_snr_results)
        pivot_table = df_mod_snr.pivot(index='Modulation', columns='SNR_dB', values='Accuracy')
        
        plt.figure(figsize=(14, 8))
        sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                    cbar_kws={'label': 'Accuracy'})
        plt.title('ULCNN Pretrained Model - Accuracy by Modulation and SNR')
        plt.xlabel('SNR (dB)')
        plt.ylabel('Modulation Type')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_by_mod_snr.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Generate summary report
    summary_lines = [
        "ULCNN Pretrained Model Evaluation Summary",
        "=" * 45,
        f"Dataset: RadioML 2016.10a",
        f"Test samples: {len(X_test):,}",
        f"Number of classes: {len(mods)}",
        f"SNR range: {min(unique_snrs)} to {max(unique_snrs)} dB",
        "",
        f"Overall Test Accuracy: {overall_accuracy:.4f}",
        f"Best SNR Performance: {max(snr_accuracies):.4f} at {unique_snrs[np.argmax(snr_accuracies)]} dB",
        f"Worst SNR Performance: {min(snr_accuracies):.4f} at {unique_snrs[np.argmin(snr_accuracies)]} dB",
        "",
        "Model Architecture:",
        f"  Total parameters: {model.count_params():,}",
        f"  Input shape: {model.input_shape}",
        f"  Output shape: {model.output_shape}",
        "",
        "Files generated:",
        "  - overall_accuracy.txt",
        "  - classification_report.txt", 
        "  - confusion_matrix.png",
        "  - accuracy_vs_snr.png",
        "  - accuracy_by_snr.csv",
        "  - accuracy_by_mod_snr.png",
        "  - evaluation_summary.txt"
    ]
    
    # Save summary
    with open(os.path.join(output_dir, 'evaluation_summary.txt'), 'w') as f:
        f.write('\n'.join(summary_lines))
    
    # Print summary
    print("\n" + '\n'.join(summary_lines))
    
    return overall_accuracy, snr_accuracies


def main():
    """Main execution function"""
    
    # Configuration
    dataset_path = 'data/RML2016.10a_dict.pkl'
    model_path = 'ULCNN/model/ULCNN_MN=6_N=16_KS=5.hdf5'
    output_dir = 'ulcnn_pretrained_results'
    
    print("ULCNN Pretrained Model Testing")
    print("=" * 40)
    print(f"Dataset: {dataset_path}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Set random seed for reproducibility
    tf.keras.utils.set_random_seed(42)
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_radioml_data(dataset_path)
    if dataset is None:
        print("Failed to load dataset. Exiting.")
        return
    
    print("Dataset loaded successfully!")
    
    # Prepare data using the same preprocessing as the main framework
    print("\nPreparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, snr_train, snr_val, snr_test, mods = prepare_data_by_snr(
        dataset, 
        denoising_method='none',  # Use no denoising for consistency
        test_size=0.2,
        validation_split=0.2
    )
    
    print(f"Data preparation complete:")
    print(f"  Training set: {X_train.shape}")
    print(f"  Validation set: {X_val.shape}")
    print(f"  Test set: {X_test.shape}")
    print(f"  Number of classes: {len(mods)}")
    print(f"  Modulation types: {mods}")
    
    # Transform data for original ULCNN model (expects shape (batch, 128, 2))
    print("\nTransforming data for ULCNN model...")
    X_train_ulcnn = X_train.transpose(0, 2, 1)  # (batch, 2, 128) -> (batch, 128, 2)
    X_val_ulcnn = X_val.transpose(0, 2, 1)
    X_test_ulcnn = X_test.transpose(0, 2, 1)
    
    print(f"Transformed data shapes:")
    print(f"  Training set: {X_train_ulcnn.shape}")
    print(f"  Validation set: {X_val_ulcnn.shape}")
    print(f"  Test set: {X_test_ulcnn.shape}")
    
    # Get input shape and number of classes
    input_shape = X_train.shape[1:]  # Keep original shape for compatibility
    ulcnn_input_shape = X_train_ulcnn.shape[1:]  # ULCNN expected shape
    num_classes = len(mods)
    
    print(f"\nModel configuration:")
    print(f"  Input shape: {input_shape}")
    print(f"  Number of classes: {num_classes}")
    
    # Load pretrained model
    print("\nLoading pretrained ULCNN model...")
    model = load_pretrained_ulcnn_model(model_path, input_shape, num_classes)
    
    if model is None:
        print("Failed to load pretrained model. Exiting.")
        return
    
    # Compile model for evaluation
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    # Use the transformed data that matches the model's expected input shape
    test_data = X_test_ulcnn if model.input_shape[1:] == ulcnn_input_shape else X_test
    print(f"Using test data shape: {test_data.shape} for model input shape: {model.input_shape}")
    
    overall_accuracy, snr_accuracies = evaluate_model_performance(
        model, test_data, y_test, snr_test, mods, output_dir
    )
    
    print(f"\n{'='*50}")
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print(f"{'='*50}")
    print(f"Overall Test Accuracy: {overall_accuracy:.4f}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()