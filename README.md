# RadioML

This repository contains machine learning models for radio modulation classification.

## Important Note

**This repository includes all source code, models, and data files. Large files are managed using Git LFS.**

## Project Structure

- `src/`: Source code for all models
  - Models implementations (CNN 1D, CNN 2D, Complex NN, ResNet, Transformer)
  - Utility functions and callbacks
- `model_weight_saved/`: Saved model weights (managed with Git LFS)
- `output/models/`: Output model files (managed with Git LFS)
- `RML2016.10a_dict.pkl`: The RadioML dataset (managed with Git LFS)
- `projects/`: Contains submodules of related projects
  - `AMC-Net`: Implementation of the AMC-Net architecture
  - `ULCNN`: Implementation of the ULCNN architecture

## Getting Started

To use this code, you will need to:

1. Download the RadioML dataset (RML2016.10a) from the official website
2. Place the dataset file in the repository root
3. Run the training scripts to train the models or use pre-trained weights

## Models

The repository includes implementations of various models for radio modulation classification:

- CNN 1D: One-dimensional convolutional neural network
- CNN 2D: Two-dimensional convolutional neural network 
- Complex NN: Neural network with complex-valued operations
- ResNet: Residual network architecture
- Transformer: Attention-based transformer model

## Usage

To run the project, use `src/main.py`. Key command-line arguments include:

*   `--mode`: Mode of operation. Choices: `explore`, `train`, `evaluate`, `all`. Default: `all`.
*   `--model_type`: Model architecture to use. Choices: `cnn1d`, `cnn2d`, `resnet`, `complex_nn`, `transformer`, `all`. Default: `resnet`.
*   `--dataset_path`: Path to the RadioML dataset. Default: `../RML2016.10a_dict.pkl`.
*   `--epochs`: Number of training epochs. Default: 400.
*   `--batch_size`: Batch size for training. Default: 128.
*   `--augment_data`: Enable data augmentation.
*   `--denoising_method`: Denoising method to apply to the input signals. Default: `gpr`.
    *   `gpr`: Gaussian Process Regression. (Default kernel is RBF; Matern and RationalQuadratic also available).
    *   `wavelet`: Wavelet-based denoising.
    *   `ddae`: Deep Denoising Autoencoder. Uses a pre-trained model. See below for training your own DDAE.
    *   `none`: No denoising is applied.
*   `--ddae_model_path`: Path to the trained Denoising Autoencoder model. Default: `model_weight_saved/ddae_model.keras`.

Example:
```bash
python src/main.py --model_type resnet --denoising_method wavelet --epochs 50
```

### Training the Denoising Autoencoder

The Deep Denoising Autoencoder (DDAE) model can be (re)trained using the `src/train_autoencoder.py` script. This script trains the DAE on signals from the dataset, where high-SNR signals are treated as clean targets and noise is added to create noisy inputs.

Key arguments for `src/train_autoencoder.py`:

*   `--dataset_path`: Path to the RadioML dataset (default: `../RML2016.10a_dict.pkl`, assumes script is run from `src/`).
*   `--epochs`: Number of training epochs (default: 100).
*   `--batch_size`: Batch size for training (default: 256).
*   `--learning_rate`: Learning rate for the optimizer (default: 1e-3).
*   `--high_snr_threshold`: SNR value (dB) above which signals are considered "clean" targets (default: 10).
*   `--noisy_snr_target`: Target SNR (dB) for the noisy signals generated for training (default: 0).
*   `--output_model_dir`: Directory to save the trained model weights (default: `../model_weight_saved/`).
*   `--output_model_name`: Filename for the saved model (default: `ddae_model.keras`).

Example training command (assuming you are in the `src` directory):

```bash
python train_autoencoder.py --dataset_path ../RML2016.10a_dict.pkl --epochs 50 --output_model_name ddae_custom_model.keras
```

Alternatively, if running from the project root:
```bash
python src/train_autoencoder.py --dataset_path RML2016.10a_dict.pkl --epochs 50 --output_model_name ddae_custom_model.keras --output_model_dir model_weight_saved/
```

The trained model can then be used with `src/main.py` by specifying the `--denoising_method ddae` and, if using a custom name/path, the `--ddae_model_path` argument (e.g., `--ddae_model_path model_weight_saved/ddae_custom_model.keras` if run from project root).