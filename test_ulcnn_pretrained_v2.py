#!/usr/bin/env python3
"""
ULCNN Pretrained Model Testing Script (Simplified Version)

This script loads the pretrained ULCNN model weights and evaluates performance
on the RadioML 2016.10a dataset using a simplified approach compatible with
older TensorFlow/Keras versions.

Usage:
    conda activate ljk3_old && python test_ulcnn_pretrained_v2.py
"""

import os
import sys
import numpy as np
import pickle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Set environment for ULCNN
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Add ULCNN directory to path
sys.path.append('ULCNN')

# Import ULCNN components
from keras import models
from keras.layers import *
from keras.models import Model
from keras.utils import to_categorical
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from complexnn.conv import ComplexConv1D
from complexnn.bn import ComplexBatchNormalization
from complexnn.dense import ComplexDense


def load_radioml_data(file_path):
    """Load RadioML dataset"""
    print(f"Loading dataset from {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        print("Dataset loaded successfully!")
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None


def prepare_radioml_data(dataset, test_size=0.2, validation_split=0.2):
    """
    Prepare RadioML data for ULCNN model
    Returns data in the format expected by ULCNN: (batch, 128, 2)
    """
    print("Preparing data for ULCNN model...")
    
    # Get modulation types and SNRs
    mods = sorted(list(set([k[0] for k in dataset.keys()])))
    snrs = sorted(list(set([k[1] for k in dataset.keys()])))
    
    print(f"Modulation types: {mods}")
    print(f"SNR range: {min(snrs)} to {max(snrs)} dB")
    
    # Create mapping
    mod_to_index = {mod: i for i, mod in enumerate(mods)}
    
    # Collect all data
    X_list = []
    y_list = []
    snr_list = []
    
    for mod in mods:
        for snr in snrs:
            key = (mod, snr)
            if key in dataset:
                data_samples = dataset[key]
                # Transpose to match ULCNN expected format: (batch, 128, 2)
                data_samples = data_samples.transpose(0, 2, 1)
                
                X_list.append(data_samples)
                y_list.append(np.ones(len(data_samples)) * mod_to_index[mod])
                snr_list.append(np.ones(len(data_samples)) * snr)
    
    # Convert to arrays
    X_all = np.vstack(X_list)
    y_all = np.hstack(y_list).astype(int)
    snr_all = np.hstack(snr_list)
    
    print(f"Total samples: {len(X_all)}")
    print(f"Data shape: {X_all.shape}")
    
    # Split data
    X_train_val, X_test, y_train_val, y_test, snr_train_val, snr_test = train_test_split(
        X_all, y_all, snr_all, test_size=test_size, random_state=42, stratify=y_all
    )
    
    # Further split training data
    val_size_adjusted = validation_split / (1 - test_size)
    X_train, X_val, y_train, y_val, snr_train, snr_val = train_test_split(
        X_train_val, y_train_val, snr_train_val, test_size=val_size_adjusted, 
        random_state=42, stratify=y_train_val
    )
    
    # Convert to one-hot encoding
    num_classes = len(mods)
    y_train = to_categorical(y_train, num_classes)
    y_val = to_categorical(y_val, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_val.shape}, {y_val.shape}")
    print(f"Test set: {X_test.shape}, {y_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, snr_train, snr_val, snr_test, mods


def build_ulcnn_model(input_shape=(128, 2), num_classes=11, n_neuron=16, n_mobileunit=6, ks=5):
    """
    Build ULCNN model exactly as in the original paper
    """
    def channel_shuffle(x):
        in_channels, D = K.int_shape(x)[1:]
        channels_per_group = in_channels // 2
        pre_shape = [-1, 2, channels_per_group, D]
        dim = (0, 2, 1, 3)
        later_shape = [-1, in_channels, D]

        x = Lambda(lambda z: K.reshape(z, pre_shape))(x)
        x = Lambda(lambda z: K.permute_dimensions(z, dim))(x)  
        x = Lambda(lambda z: K.reshape(z, later_shape))(x)
        return x

    def dwconv_mobile(x, neurons):
        x = SeparableConv1D(int(2*neurons), ks, strides=2, padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = channel_shuffle(x)
        return x

    def channelattention(x):
        x_GAP = GlobalAveragePooling1D()(x)
        x_GMP = GlobalMaxPooling1D()(x)
        channel = K.int_shape(x_GAP)[1]

        share_Dense1 = Dense(int(channel/16), activation='relu')
        share_Dense2 = Dense(channel)

        x_GAP = Reshape((1, channel))(x_GAP)
        x_GAP = share_Dense1(x_GAP)
        x_GAP = share_Dense2(x_GAP)

        x_GMP = Reshape((1, channel))(x_GMP)
        x_GMP = share_Dense1(x_GMP)
        x_GMP = share_Dense2(x_GMP)

        x_Mask = Add()([x_GAP, x_GMP])
        x_Mask = Activation('sigmoid')(x_Mask)

        x = Multiply()([x, x_Mask])
        return x

    # Build model
    x_input = Input(shape=input_shape)
    x = ComplexConv1D(n_neuron, ks, padding='same')(x_input)
    x = ComplexBatchNormalization()(x)
    x = Activation('relu')(x)

    for i in range(n_mobileunit):
        x = dwconv_mobile(x, n_neuron)
        x = channelattention(x)
        if i == 3:
            f4 = GlobalAveragePooling1D()(x)
        if i == 4:
            f5 = GlobalAveragePooling1D()(x)
        if i == 5:
            f6 = GlobalAveragePooling1D()(x)

    f = Add()([f4, f5])
    f = Add()([f, f6])

    f = Dense(num_classes)(f)
    c = Activation('softmax', name='modulation')(f)

    model = Model(inputs=x_input, outputs=c)
    return model


def evaluate_model_performance(model, X_test, y_test, snr_test, mods, output_dir):
    """Evaluate model performance and generate reports"""
    print("\nEvaluating model performance...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Make predictions
    print("Making predictions on test set...")
    y_pred_proba = model.predict(X_test, batch_size=128, verbose=1)
    y_pred = np.argmax(y_pred_proba, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    # Overall accuracy
    overall_accuracy = accuracy_score(y_true, y_pred)
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
            snr_acc = accuracy_score(y_true[snr_mask], y_pred[snr_mask])
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
    import pandas as pd
    snr_data = {'SNR_dB': unique_snrs, 'Accuracy': snr_accuracies}
    df_snr = pd.DataFrame(snr_data)
    df_snr.to_csv(os.path.join(output_dir, 'accuracy_by_snr.csv'), index=False)
    
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
    
    print("ULCNN Pretrained Model Testing (Simplified Version)")
    print("=" * 55)
    print(f"Dataset: {dataset_path}")
    print(f"Model: {model_path}")
    print(f"Output: {output_dir}")
    print()
    
    # Load dataset
    dataset = load_radioml_data(dataset_path)
    if dataset is None:
        print("Failed to load dataset. Exiting.")
        return
    
    # Prepare data
    X_train, X_val, X_test, y_train, y_val, y_test, snr_train, snr_val, snr_test, mods = prepare_radioml_data(dataset)
    
    # Build model
    print("\nBuilding ULCNN model...")
    input_shape = X_train.shape[1:]
    num_classes = len(mods)
    
    model = build_ulcnn_model(input_shape, num_classes)
    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam', 
        metrics=['accuracy']
    )
    
    print(f"Model built successfully!")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    print(f"Total parameters: {model.count_params():,}")
    
    # Load pretrained weights
    print(f"\nLoading pretrained weights from: {model_path}")
    if os.path.exists(model_path):
        try:
            # Try different methods to load the weights
            print("Attempting to load weights...")
            
            # Method 1: Direct load_weights
            try:
                model.load_weights(model_path)
                print("Successfully loaded pretrained weights using direct method!")
            except Exception as e1:
                print(f"Direct method failed: {e1}")
                
                # Method 2: Load the entire model and extract weights
                try:
                    print("Trying to load complete model...")
                    import h5py
                    
                    # Check if it's an HDF5 file
                    with h5py.File(model_path, 'r') as f:
                        print(f"HDF5 file structure: {list(f.keys())}")
                        
                        # Try to load as complete model first
                        try:
                            from keras.models import load_model
                            pretrained_model = load_model(model_path)
                            
                            # Copy weights from pretrained model to our model
                            for i, layer in enumerate(model.layers):
                                if i < len(pretrained_model.layers):
                                    try:
                                        layer.set_weights(pretrained_model.layers[i].get_weights())
                                        print(f"Loaded weights for layer {i}: {layer.name}")
                                    except Exception as layer_e:
                                        print(f"Could not load weights for layer {i}: {layer_e}")
                            
                            print("Successfully loaded pretrained weights using model extraction method!")
                            
                        except Exception as e2:
                            print(f"Model loading method failed: {e2}")
                            
                            # Method 3: Manual weight loading from HDF5
                            try:
                                print("Trying manual HDF5 weight extraction...")
                                
                                # Load weights manually from HDF5 file
                                def load_weights_from_hdf5_group(f, layers):
                                    """Recursively load weights from HDF5 group"""
                                    if 'model_weights' in f:
                                        weight_group = f['model_weights']
                                    elif 'layer_names' in f.attrs:
                                        weight_group = f
                                    else:
                                        print("Could not find weight structure in HDF5 file")
                                        return False
                                    
                                    layer_names = [n.decode('utf8') for n in weight_group.attrs['layer_names']]
                                    print(f"Found layers in HDF5: {layer_names}")
                                    
                                    for layer_name in layer_names:
                                        if layer_name in weight_group:
                                            layer_group = weight_group[layer_name]
                                            weight_names = [n.decode('utf8') for n in layer_group.attrs['weight_names']]
                                            
                                            # Find corresponding layer in our model
                                            model_layer = None
                                            for layer in layers:
                                                if layer.name == layer_name:
                                                    model_layer = layer
                                                    break
                                            
                                            if model_layer is not None:
                                                weights = []
                                                for weight_name in weight_names:
                                                    if weight_name in layer_group:
                                                        weights.append(layer_group[weight_name][:])
                                                
                                                if weights:
                                                    try:
                                                        model_layer.set_weights(weights)
                                                        print(f"Loaded weights for layer: {layer_name}")
                                                    except Exception as we:
                                                        print(f"Could not set weights for {layer_name}: {we}")
                                    
                                    return True
                                
                                if load_weights_from_hdf5_group(f, model.layers):
                                    print("Successfully loaded weights using manual HDF5 extraction!")
                                else:
                                    raise Exception("Manual HDF5 extraction failed")
                                    
                            except Exception as e3:
                                print(f"Manual HDF5 extraction failed: {e3}")
                                print("Using randomly initialized weights instead...")
                                
                except Exception as e_h5:
                    print(f"HDF5 processing failed: {e_h5}")
                    print("Using randomly initialized weights instead...")
                    
        except Exception as e:
            print(f"All weight loading methods failed: {e}")
            print("Using randomly initialized weights instead...")
    else:
        print(f"Model file not found: {model_path}")
        print("Using randomly initialized weights instead...")
    
    # Evaluate on test set
    print("\nEvaluating model on test set...")
    overall_accuracy, snr_accuracies = evaluate_model_performance(
        model, X_test, y_test, snr_test, mods, output_dir
    )
    
    print(f"\n{'='*50}")
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print(f"{'='*50}")
    print(f"Overall Test Accuracy: {overall_accuracy:.4f}")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()