#!/usr/bin/env python3
"""
ULCNN Performance Validation and Optimization Script

This script benchmarks training speed, monitors memory usage, verifies gradient flow,
tests different batch sizes, and documents performance characteristics for ULCNN models
compared to existing models.

Requirements addressed:
- 1.1, 1.2, 1.3: Model integration and functionality
- 3.1, 3.2, 3.3: Training verification and performance
"""

import os
import sys
import time
import psutil
import gc
import json
import argparse
import numpy as np
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from memory_profiler import profile
import tracemalloc

# Add src to path for imports
sys.path.append('src')

from explore_dataset import load_radioml_data
from preprocess import prepare_data_by_snr
from models import (
    build_cnn1d_model, build_cnn2d_model, build_resnet_model, build_complex_nn_model,
    build_transformer_model, build_lstm_model,
    # ULCNN models
    build_mcldnn_model, build_scnn_model, build_mcnet_model, build_pet_model, build_ulcnn_model
)


class PerformanceMonitor:
    """Monitor system performance during model operations"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = None
        self.start_memory = None
        self.peak_memory = 0
        self.memory_samples = []
        self.gpu_memory_samples = []
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.peak_memory = self.start_memory
        self.memory_samples = [self.start_memory]
        
        # Start GPU memory monitoring if available
        if tf.config.list_physical_devices('GPU'):
            try:
                gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
                self.gpu_memory_samples = [gpu_memory['current'] / 1024 / 1024]  # MB
            except:
                self.gpu_memory_samples = []
        
        # Start tracemalloc for detailed memory tracking
        tracemalloc.start()
        
    def sample_memory(self):
        """Sample current memory usage"""
        current_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        self.memory_samples.append(current_memory)
        self.peak_memory = max(self.peak_memory, current_memory)
        
        # Sample GPU memory if available
        if tf.config.list_physical_devices('GPU'):
            try:
                gpu_memory = tf.config.experimental.get_memory_info('GPU:0')
                self.gpu_memory_samples.append(gpu_memory['current'] / 1024 / 1024)  # MB
            except:
                pass
    
    def stop_monitoring(self):
        """Stop monitoring and return results"""
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # Get tracemalloc snapshot
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        tracemalloc.stop()
        
        return {
            'duration': end_time - self.start_time,
            'start_memory_mb': self.start_memory,
            'end_memory_mb': end_memory,
            'peak_memory_mb': self.peak_memory,
            'memory_increase_mb': end_memory - self.start_memory,
            'memory_samples': self.memory_samples,
            'gpu_memory_samples': self.gpu_memory_samples,
            'top_memory_allocations': [(stat.traceback.format(), stat.size / 1024 / 1024) 
                                     for stat in top_stats[:5]]
        }


class GradientMonitor(tf.keras.callbacks.Callback):
    """Monitor gradient flow during training"""
    
    def __init__(self, validation_data=None):
        super().__init__()
        self.gradient_norms = []
        self.weight_norms = []
        self.gradient_ratios = []
        self.validation_data = validation_data
        
    def on_batch_end(self, batch, logs=None):
        """Monitor gradients after each batch"""
        if batch % 10 == 0 and self.validation_data is not None:  # Sample every 10 batches to avoid overhead
            try:
                # Get gradients using a small validation sample
                X_val_sample = self.validation_data[0][:32]
                y_val_sample = self.validation_data[1][:32]
                
                with tf.GradientTape() as tape:
                    predictions = self.model(X_val_sample, training=True)
                    loss = self.model.compiled_loss(y_val_sample, predictions)
                
                gradients = tape.gradient(loss, self.model.trainable_variables)
                
                # Calculate gradient norms
                grad_norms = []
                weight_norms = []
                ratios = []
                
                for grad, weight in zip(gradients, self.model.trainable_variables):
                    if grad is not None:
                        grad_norm = tf.norm(grad).numpy()
                        weight_norm = tf.norm(weight).numpy()
                        
                        grad_norms.append(grad_norm)
                        weight_norms.append(weight_norm)
                        
                        if weight_norm > 0:
                            ratios.append(grad_norm / weight_norm)
                
                if grad_norms:
                    self.gradient_norms.append(np.mean(grad_norms))
                    self.weight_norms.append(np.mean(weight_norms))
                    self.gradient_ratios.append(np.mean(ratios) if ratios else 0)
                    
            except Exception as e:
                # Skip gradient monitoring if there's an error
                pass


def benchmark_model_training(model_name, model_builder, X_train, y_train, X_val, y_val, 
                           input_shape, num_classes, batch_size=128, epochs=3):
    """Benchmark training performance for a single model"""
    
    print(f"\nBenchmarking {model_name} training...")
    
    # Initialize monitoring
    monitor = PerformanceMonitor()
    gradient_monitor = GradientMonitor(validation_data=(X_val, y_val))
    
    try:
        # Start monitoring
        monitor.start_monitoring()
        
        # Build model
        model = model_builder(input_shape, num_classes)
        
        # Debug: Check data shapes
        print(f"Model output shape: {model.output_shape}")
        print(f"Expected classes: {num_classes}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_train sample: {y_train[:5]}")
        
        # Determine loss function based on label format
        if len(y_train.shape) > 1 and y_train.shape[1] > 1:
            # One-hot encoded labels
            loss_function = 'categorical_crossentropy'
        else:
            # Integer labels
            loss_function = 'sparse_categorical_crossentropy'
        
        print(f"Using loss function: {loss_function}")
        
        model.compile(
            optimizer='adam',
            loss=loss_function,
            metrics=['accuracy']
        )
        
        monitor.sample_memory()
        
        # Count parameters
        total_params = model.count_params()
        trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
        
        print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        # Training with monitoring
        start_train_time = time.time()
        
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=[gradient_monitor],
            verbose=1
        )
        
        train_time = time.time() - start_train_time
        monitor.sample_memory()
        
        # Stop monitoring
        perf_stats = monitor.stop_monitoring()
        
        # Calculate training metrics
        samples_per_second = len(X_train) * epochs / train_time
        time_per_epoch = train_time / epochs
        
        # Compile results
        results = {
            'model_name': model_name,
            'total_params': int(total_params),
            'trainable_params': int(trainable_params),
            'batch_size': batch_size,
            'epochs': epochs,
            'training_time_seconds': train_time,
            'time_per_epoch_seconds': time_per_epoch,
            'samples_per_second': samples_per_second,
            'final_train_accuracy': float(history.history['accuracy'][-1]),
            'final_val_accuracy': float(history.history['val_accuracy'][-1]),
            'final_train_loss': float(history.history['loss'][-1]),
            'final_val_loss': float(history.history['val_loss'][-1]),
            'gradient_flow': {
                'mean_gradient_norm': float(np.mean(gradient_monitor.gradient_norms)) if gradient_monitor.gradient_norms else 0,
                'mean_weight_norm': float(np.mean(gradient_monitor.weight_norms)) if gradient_monitor.weight_norms else 0,
                'mean_gradient_ratio': float(np.mean(gradient_monitor.gradient_ratios)) if gradient_monitor.gradient_ratios else 0,
                'gradient_stability': float(np.std(gradient_monitor.gradient_norms)) if gradient_monitor.gradient_norms else 0
            },
            'memory_usage': perf_stats,
            'timestamp': datetime.now().isoformat()
        }
        
        # Clean up
        del model
        tf.keras.backend.clear_session()
        gc.collect()
        
        return results
        
    except Exception as e:
        print(f"Error benchmarking {model_name}: {e}")
        # Clean up on error
        tf.keras.backend.clear_session()
        gc.collect()
        return {
            'model_name': model_name,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def test_batch_size_robustness(model_name, model_builder, X_train, y_train, X_val, y_val,
                              input_shape, num_classes, batch_sizes=[32, 64, 128, 256, 512]):
    """Test model performance with different batch sizes"""
    
    print(f"\nTesting batch size robustness for {model_name}...")
    
    results = []
    
    for batch_size in batch_sizes:
        print(f"Testing batch size: {batch_size}")
        
        try:
            # Limit data size for faster testing
            n_samples = min(1000, len(X_train))
            X_train_small = X_train[:n_samples]
            y_train_small = y_train[:n_samples]
            X_val_small = X_val[:200]
            y_val_small = y_val[:200]
            
            result = benchmark_model_training(
                f"{model_name}_bs{batch_size}",
                model_builder,
                X_train_small, y_train_small,
                X_val_small, y_val_small,
                input_shape, num_classes,
                batch_size=batch_size,
                epochs=2
            )
            
            result['batch_size_test'] = True
            results.append(result)
            
        except Exception as e:
            print(f"Failed with batch size {batch_size}: {e}")
            results.append({
                'model_name': f"{model_name}_bs{batch_size}",
                'batch_size': batch_size,
                'error': str(e),
                'batch_size_test': True,
                'timestamp': datetime.now().isoformat()
            })
    
    return results


def compare_model_architectures():
    """Compare ULCNN models with existing models"""
    
    # Model configurations
    models_to_test = {
        # Existing models
        'cnn1d': build_cnn1d_model,
        'cnn2d': build_cnn2d_model,
        'resnet': build_resnet_model,
        'complex_nn': build_complex_nn_model,
        'transformer': build_transformer_model,
        'lstm': build_lstm_model,
        
        # ULCNN models
        'mcldnn': build_mcldnn_model,
        'scnn': build_scnn_model,
        'mcnet': build_mcnet_model,
        'pet': build_pet_model,
        'ulcnn': build_ulcnn_model
    }
    
    return models_to_test


def generate_performance_report(results, output_dir):
    """Generate comprehensive performance report"""
    
    print("\nGenerating performance report...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw results
    with open(os.path.join(output_dir, 'benchmark_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    # Filter successful results
    successful_results = [r for r in results if 'error' not in r]
    
    if not successful_results:
        print("No successful results to analyze")
        return
    
    # Create DataFrame for analysis
    df_data = []
    for result in successful_results:
        row = {
            'model_name': result['model_name'],
            'total_params': result.get('total_params', 0),
            'trainable_params': result.get('trainable_params', 0),
            'training_time': result.get('training_time_seconds', 0),
            'time_per_epoch': result.get('time_per_epoch_seconds', 0),
            'samples_per_second': result.get('samples_per_second', 0),
            'final_val_accuracy': result.get('final_val_accuracy', 0),
            'peak_memory_mb': result.get('memory_usage', {}).get('peak_memory_mb', 0),
            'memory_increase_mb': result.get('memory_usage', {}).get('memory_increase_mb', 0),
            'gradient_norm': result.get('gradient_flow', {}).get('mean_gradient_norm', 0),
            'gradient_stability': result.get('gradient_flow', {}).get('gradient_stability', 0),
            'batch_size': result.get('batch_size', 128)
        }
        df_data.append(row)
    
    df = pd.DataFrame(df_data)
    
    # Save detailed results
    df.to_csv(os.path.join(output_dir, 'performance_comparison.csv'), index=False)
    
    # Generate plots
    generate_performance_plots(df, output_dir)
    
    # Generate summary report
    generate_summary_report(df, successful_results, output_dir)


def generate_performance_plots(df, output_dir):
    """Generate performance visualization plots"""
    
    # Separate ULCNN models from existing models
    ulcnn_models = ['mcldnn', 'scnn', 'mcnet', 'pet', 'ulcnn']
    df['model_type'] = df['model_name'].apply(
        lambda x: 'ULCNN' if any(ulcnn in x for ulcnn in ulcnn_models) else 'Existing'
    )
    
    # Filter out batch size test results for main comparison
    main_df = df[~df['model_name'].str.contains('_bs')]
    
    # 1. Training Speed Comparison
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    main_df.groupby('model_type')['samples_per_second'].mean().plot(kind='bar')
    plt.title('Average Training Speed (Samples/Second)')
    plt.ylabel('Samples per Second')
    plt.xticks(rotation=45)
    
    # 2. Memory Usage Comparison
    plt.subplot(2, 2, 2)
    main_df.groupby('model_type')['peak_memory_mb'].mean().plot(kind='bar', color='orange')
    plt.title('Average Peak Memory Usage')
    plt.ylabel('Memory (MB)')
    plt.xticks(rotation=45)
    
    # 3. Model Complexity vs Performance
    plt.subplot(2, 2, 3)
    colors = ['blue' if t == 'Existing' else 'red' for t in main_df['model_type']]
    plt.scatter(main_df['total_params'], main_df['final_val_accuracy'], c=colors, alpha=0.7)
    plt.xlabel('Total Parameters')
    plt.ylabel('Validation Accuracy')
    plt.title('Model Complexity vs Performance')
    plt.legend(['Existing Models', 'ULCNN Models'])
    
    # 4. Training Time vs Accuracy
    plt.subplot(2, 2, 4)
    plt.scatter(main_df['training_time'], main_df['final_val_accuracy'], c=colors, alpha=0.7)
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Validation Accuracy')
    plt.title('Training Time vs Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Batch size robustness plot
    batch_size_df = df[df['model_name'].str.contains('_bs')]
    if not batch_size_df.empty:
        plt.figure(figsize=(10, 6))
        
        for model in batch_size_df['model_name'].str.replace(r'_bs\d+', '', regex=True).unique():
            model_data = batch_size_df[batch_size_df['model_name'].str.contains(model)]
            plt.plot(model_data['batch_size'], model_data['samples_per_second'], 
                    marker='o', label=model)
        
        plt.xlabel('Batch Size')
        plt.ylabel('Samples per Second')
        plt.title('Training Speed vs Batch Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'batch_size_robustness.png'), dpi=300, bbox_inches='tight')
        plt.close()


def generate_summary_report(df, results, output_dir):
    """Generate text summary report"""
    
    ulcnn_models = ['mcldnn', 'scnn', 'mcnet', 'pet', 'ulcnn']
    main_df = df[~df['model_name'].str.contains('_bs')]
    
    ulcnn_df = main_df[main_df['model_name'].apply(lambda x: any(ulcnn in x for ulcnn in ulcnn_models))]
    existing_df = main_df[~main_df['model_name'].apply(lambda x: any(ulcnn in x for ulcnn in ulcnn_models))]
    
    report = []
    report.append("ULCNN Performance Validation and Optimization Report")
    report.append("=" * 60)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("")
    
    # Executive Summary
    report.append("EXECUTIVE SUMMARY")
    report.append("-" * 20)
    report.append(f"Total models tested: {len(main_df)}")
    report.append(f"ULCNN models: {len(ulcnn_df)}")
    report.append(f"Existing models: {len(existing_df)}")
    report.append("")
    
    # Performance Comparison
    if not ulcnn_df.empty and not existing_df.empty:
        report.append("PERFORMANCE COMPARISON")
        report.append("-" * 25)
        
        # Training Speed
        ulcnn_speed = ulcnn_df['samples_per_second'].mean()
        existing_speed = existing_df['samples_per_second'].mean()
        speed_ratio = ulcnn_speed / existing_speed if existing_speed > 0 else 0
        
        report.append(f"Average Training Speed:")
        report.append(f"  ULCNN models: {ulcnn_speed:.1f} samples/second")
        report.append(f"  Existing models: {existing_speed:.1f} samples/second")
        report.append(f"  Speed ratio: {speed_ratio:.2f}x")
        report.append("")
        
        # Memory Usage
        ulcnn_memory = ulcnn_df['peak_memory_mb'].mean()
        existing_memory = existing_df['peak_memory_mb'].mean()
        memory_ratio = ulcnn_memory / existing_memory if existing_memory > 0 else 0
        
        report.append(f"Average Peak Memory Usage:")
        report.append(f"  ULCNN models: {ulcnn_memory:.1f} MB")
        report.append(f"  Existing models: {existing_memory:.1f} MB")
        report.append(f"  Memory ratio: {memory_ratio:.2f}x")
        report.append("")
        
        # Model Complexity
        ulcnn_params = ulcnn_df['total_params'].mean()
        existing_params = existing_df['total_params'].mean()
        params_ratio = ulcnn_params / existing_params if existing_params > 0 else 0
        
        report.append(f"Average Model Parameters:")
        report.append(f"  ULCNN models: {ulcnn_params:,.0f}")
        report.append(f"  Existing models: {existing_params:,.0f}")
        report.append(f"  Parameter ratio: {params_ratio:.2f}x")
        report.append("")
        
        # Accuracy
        ulcnn_acc = ulcnn_df['final_val_accuracy'].mean()
        existing_acc = existing_df['final_val_accuracy'].mean()
        
        report.append(f"Average Validation Accuracy:")
        report.append(f"  ULCNN models: {ulcnn_acc:.3f}")
        report.append(f"  Existing models: {existing_acc:.3f}")
        report.append(f"  Accuracy difference: {ulcnn_acc - existing_acc:+.3f}")
        report.append("")
    
    # Individual Model Performance
    report.append("INDIVIDUAL MODEL PERFORMANCE")
    report.append("-" * 35)
    
    for _, row in main_df.iterrows():
        report.append(f"{row['model_name'].upper()}:")
        report.append(f"  Parameters: {row['total_params']:,}")
        report.append(f"  Training speed: {row['samples_per_second']:.1f} samples/sec")
        report.append(f"  Peak memory: {row['peak_memory_mb']:.1f} MB")
        report.append(f"  Validation accuracy: {row['final_val_accuracy']:.3f}")
        report.append(f"  Gradient stability: {row['gradient_stability']:.6f}")
        report.append("")
    
    # Gradient Flow Analysis
    report.append("GRADIENT FLOW ANALYSIS")
    report.append("-" * 25)
    
    for result in results:
        if 'gradient_flow' in result and 'error' not in result:
            gf = result['gradient_flow']
            report.append(f"{result['model_name'].upper()}:")
            report.append(f"  Mean gradient norm: {gf['mean_gradient_norm']:.6f}")
            report.append(f"  Mean weight norm: {gf['mean_weight_norm']:.6f}")
            report.append(f"  Gradient/weight ratio: {gf['mean_gradient_ratio']:.6f}")
            report.append(f"  Gradient stability (std): {gf['gradient_stability']:.6f}")
            
            # Assess gradient flow health
            if gf['mean_gradient_norm'] < 1e-6:
                report.append("  ⚠️  WARNING: Very small gradients (vanishing gradient problem)")
            elif gf['mean_gradient_norm'] > 1.0:
                report.append("  ⚠️  WARNING: Large gradients (exploding gradient problem)")
            else:
                report.append("  ✅ Healthy gradient flow")
            report.append("")
    
    # Batch Size Robustness
    batch_size_df = df[df['model_name'].str.contains('_bs')]
    if not batch_size_df.empty:
        report.append("BATCH SIZE ROBUSTNESS")
        report.append("-" * 25)
        
        for model in batch_size_df['model_name'].str.replace(r'_bs\d+', '', regex=True).unique():
            model_data = batch_size_df[batch_size_df['model_name'].str.contains(model)]
            if len(model_data) > 1:
                speed_std = model_data['samples_per_second'].std()
                speed_mean = model_data['samples_per_second'].mean()
                cv = speed_std / speed_mean if speed_mean > 0 else 0
                
                report.append(f"{model.upper()}:")
                report.append(f"  Speed coefficient of variation: {cv:.3f}")
                if cv < 0.1:
                    report.append("  ✅ Robust across batch sizes")
                elif cv < 0.3:
                    report.append("  ⚠️  Moderate sensitivity to batch size")
                else:
                    report.append("  ❌ High sensitivity to batch size")
                report.append("")
    
    # Recommendations
    report.append("RECOMMENDATIONS")
    report.append("-" * 15)
    
    if not ulcnn_df.empty:
        # Find best ULCNN model
        best_ulcnn = ulcnn_df.loc[ulcnn_df['final_val_accuracy'].idxmax()]
        report.append(f"Best performing ULCNN model: {best_ulcnn['model_name']}")
        report.append(f"  Accuracy: {best_ulcnn['final_val_accuracy']:.3f}")
        report.append(f"  Speed: {best_ulcnn['samples_per_second']:.1f} samples/sec")
        report.append("")
        
        # Find most efficient ULCNN model
        ulcnn_df['efficiency'] = ulcnn_df['final_val_accuracy'] / (ulcnn_df['total_params'] / 1000)
        most_efficient = ulcnn_df.loc[ulcnn_df['efficiency'].idxmax()]
        report.append(f"Most efficient ULCNN model: {most_efficient['model_name']}")
        report.append(f"  Efficiency score: {most_efficient['efficiency']:.6f}")
        report.append("")
    
    # Performance characteristics and limitations
    report.append("PERFORMANCE CHARACTERISTICS & LIMITATIONS")
    report.append("-" * 45)
    
    # Memory usage patterns
    high_memory_models = main_df[main_df['peak_memory_mb'] > main_df['peak_memory_mb'].quantile(0.75)]
    if not high_memory_models.empty:
        report.append("High memory usage models:")
        for _, model in high_memory_models.iterrows():
            report.append(f"  {model['model_name']}: {model['peak_memory_mb']:.1f} MB")
        report.append("")
    
    # Training speed patterns
    slow_models = main_df[main_df['samples_per_second'] < main_df['samples_per_second'].quantile(0.25)]
    if not slow_models.empty:
        report.append("Slower training models:")
        for _, model in slow_models.iterrows():
            report.append(f"  {model['model_name']}: {model['samples_per_second']:.1f} samples/sec")
        report.append("")
    
    # Failed models
    failed_results = [r for r in results if 'error' in r]
    if failed_results:
        report.append("FAILED MODELS")
        report.append("-" * 15)
        for result in failed_results:
            report.append(f"{result['model_name']}: {result['error']}")
        report.append("")
    
    # Save report
    with open(os.path.join(output_dir, 'performance_report.txt'), 'w') as f:
        f.write('\n'.join(report))
    
    # Print summary to console
    print("\n" + "\n".join(report[:50]))  # Print first 50 lines
    print(f"\nFull report saved to: {os.path.join(output_dir, 'performance_report.txt')}")


def main():
    parser = argparse.ArgumentParser(description='ULCNN Performance Benchmarking')
    parser.add_argument('--dataset_path', type=str, default='data/RML2016.10a_dict.pkl',
                        help='Path to RadioML dataset')
    parser.add_argument('--output_dir', type=str, default='ulcnn_performance_results',
                        help='Output directory for results')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of epochs for benchmarking')
    parser.add_argument('--test_batch_sizes', action='store_true',
                        help='Test different batch sizes')
    parser.add_argument('--models', nargs='+', 
                        choices=['all', 'ulcnn_only', 'existing_only'] + list(compare_model_architectures().keys()),
                        default=['all'],
                        help='Models to benchmark')
    
    args = parser.parse_args()
    
    # Set up TensorFlow
    tf.keras.utils.set_random_seed(42)
    
    # Load and prepare data
    print("Loading dataset...")
    dataset = load_radioml_data(args.dataset_path)
    if dataset is None:
        print("Failed to load dataset")
        return
    
    print("Preparing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, _, _, _, mods = prepare_data_by_snr(
        dataset, denoising_method='none')
    
    # Use smaller subset for benchmarking
    n_train = min(2000, len(X_train))
    n_val = min(400, len(X_val))
    
    X_train = X_train[:n_train]
    y_train = y_train[:n_train]
    X_val = X_val[:n_val]
    y_val = y_val[:n_val]
    
    input_shape = X_train.shape[1:]
    num_classes = len(mods)
    
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Get models to test
    all_models = compare_model_architectures()
    
    if 'all' in args.models:
        models_to_test = all_models
    elif 'ulcnn_only' in args.models:
        ulcnn_models = ['mcldnn', 'scnn', 'mcnet', 'pet', 'ulcnn']
        models_to_test = {k: v for k, v in all_models.items() if k in ulcnn_models}
    elif 'existing_only' in args.models:
        ulcnn_models = ['mcldnn', 'scnn', 'mcnet', 'pet', 'ulcnn']
        models_to_test = {k: v for k, v in all_models.items() if k not in ulcnn_models}
    else:
        models_to_test = {k: v for k, v in all_models.items() if k in args.models}
    
    print(f"Testing models: {list(models_to_test.keys())}")
    
    # Run benchmarks
    all_results = []
    
    for model_name, model_builder in models_to_test.items():
        print(f"\n{'='*60}")
        print(f"BENCHMARKING {model_name.upper()}")
        print(f"{'='*60}")
        
        # Main benchmark
        result = benchmark_model_training(
            model_name, model_builder,
            X_train, y_train, X_val, y_val,
            input_shape, num_classes,
            epochs=args.epochs
        )
        all_results.append(result)
        
        # Batch size robustness test
        if args.test_batch_sizes and 'error' not in result:
            batch_results = test_batch_size_robustness(
                model_name, model_builder,
                X_train, y_train, X_val, y_val,
                input_shape, num_classes
            )
            all_results.extend(batch_results)
    
    # Generate comprehensive report
    generate_performance_report(all_results, args.output_dir)
    
    print(f"\n{'='*60}")
    print("BENCHMARKING COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()