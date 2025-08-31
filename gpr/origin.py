# 进行gpr去噪处理,实现方式和 src/preprocess.py 中完全相同,将去噪后数据和处理时间保存到 gpr/origin 中,可以通过命令行参数选择处理数据为原数据集 data/RML2016.10a_dict.pkl 或 gpr 下的1000条数据.

import numpy as np
import pickle
import os
import argparse
import time
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def load_data(file_path):
    """Load RadioML dataset."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

def calculate_power(i_component, q_component):
    """Calculate the power of the signal from I and Q components."""
    return np.mean(i_component**2 + q_component**2)

def estimate_noise_std(signal_power, snr_db):
    """Estimate noise standard deviation from signal power and SNR."""
    snr_linear = 10**(snr_db / 10)
    noise_power = signal_power / (snr_linear + 1)
    return np.sqrt(noise_power / 2)

def apply_gp_regression(complex_signal, noise_std, kernel_name='rbf', length_scale=50, matern_nu=1.5, rational_quadratic_alpha=1.0):
    """
    Apply Gaussian Process Regression to denoise a complex signal.
    Args:
        complex_signal (np.ndarray): Array of complex numbers representing the signal.
        noise_std (float): Estimated standard deviation of the noise.
        kernel_name (str): Type of kernel ('rbf', 'matern', 'rational_quadratic').
        length_scale (float): Length scale parameter for RBF, Matern, RationalQuadratic kernels.
        matern_nu (float): Nu parameter for Matern kernel.
        rational_quadratic_alpha (float): Alpha parameter for RationalQuadratic kernel.
    Returns:
        np.ndarray: Denoised complex signal.
    """
    X = np.arange(len(complex_signal)).reshape(-1, 1)
    y_real = complex_signal.real
    y_imag = complex_signal.imag

    # Kernel selection (using RBF as in src/preprocess.py)
    kernel = RBF(length_scale=length_scale, length_scale_bounds="fixed")

    # Apply GPR to the real part
    gpr_real = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2, normalize_y=True)
    gpr_real.fit(X, y_real)
    y_real_denoised, _ = gpr_real.predict(X, return_std=True)

    # Apply GPR to the imaginary part
    gpr_imag = GaussianProcessRegressor(kernel=kernel, alpha=noise_std**2, normalize_y=True)
    gpr_imag.fit(X, y_imag)
    y_imag_denoised, _ = gpr_imag.predict(X, return_std=True)
    
    return y_real_denoised + 1j * y_imag_denoised

def apply_gpr_denoising(dataset):
    """对整个数据集应用GPR去噪"""
    denoised_dataset = {}
    total_samples = sum(len(samples) for samples in dataset.values())
    processed_samples = 0
    start_time = time.time()
    
    print(f"开始处理 {total_samples} 个样本...")
    
    for key, samples in dataset.items():
        mod, snr_val = key
        print(f"处理 {mod} at {snr_val}dB: {len(samples)} 个样本")
        
        denoised_samples = []
        
        for sample in samples:
            i_component = sample[0, :]
            q_component = sample[1, :]
            complex_signal = i_component + 1j * q_component
            
            # 计算功率和噪声标准差
            total_power = calculate_power(i_component, q_component)
            noise_std = estimate_noise_std(total_power, snr_val)
            
            # 根据SNR调整length_scale (与src/preprocess.py中逻辑相同)
            length_scale_val = 5.0 if snr_val >= 0 else min(10, 5.0 - snr_val * 0.25)
            
            # 应用GPR去噪
            denoised_signal = apply_gp_regression(complex_signal, noise_std, 
                                                kernel_name='rbf', 
                                                length_scale=length_scale_val)
            
            # 重构去噪后的样本
            denoised_sample = np.zeros_like(sample)
            denoised_sample[0, :] = np.real(denoised_signal)
            denoised_sample[1, :] = np.imag(denoised_signal)
            
            denoised_samples.append(denoised_sample)
            processed_samples += 1
            
            # 显示进度
            if processed_samples % 100 == 0 or processed_samples == total_samples:
                elapsed_time = time.time() - start_time
                progress = processed_samples / total_samples * 100
                print(f"进度: {processed_samples}/{total_samples} ({progress:.1f}%), "
                      f"用时: {elapsed_time:.1f}s")
        
        denoised_dataset[key] = np.array(denoised_samples)
    
    total_time = time.time() - start_time
    print(f"GPR去噪完成，总用时: {total_time:.2f} 秒")
    
    return denoised_dataset, total_time

def save_results(denoised_data, processing_time, output_dir):
    """保存去噪结果和处理时间"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存去噪后的数据
    data_file = os.path.join(output_dir, "denoised_data.pkl")
    with open(data_file, 'wb') as f:
        pickle.dump(denoised_data, f)
    print(f"去噪后数据已保存到: {data_file}")
    
    # 保存处理时间信息
    time_file = os.path.join(output_dir, "processing_time.txt")
    with open(time_file, 'w') as f:
        f.write(f"GPR去噪处理时间: {processing_time:.2f} 秒\n")
        f.write(f"处理样本数: {sum(len(v) for v in denoised_data.values())}\n")
        f.write(f"平均每样本处理时间: {processing_time/sum(len(v) for v in denoised_data.values()):.4f} 秒\n")
    print(f"处理时间信息已保存到: {time_file}")

def main():
    parser = argparse.ArgumentParser(description='GPR去噪处理脚本')
    parser.add_argument('--source', choices=['original', 'sampled'], default='sampled',
                        help='数据源选择: original (原数据集) 或 sampled (1000条采样数据)')
    parser.add_argument('--output', default='origin', 
                        help='输出目录 (默认: origin)')
    
    args = parser.parse_args()
    
    # 根据参数选择数据源
    if args.source == 'original':
        input_file = "../data/RML2016.10a_dict.pkl"
        print("使用原始数据集进行GPR去噪...")
    else:
        input_file = "RML2016_1000_samples.pkl"
        print("使用1000条采样数据进行GPR去噪...")
    
    # 检查文件是否存在
    if not os.path.exists(input_file):
        print(f"错误: 找不到输入文件 {input_file}")
        if args.source == 'sampled':
            print("请先运行 'python data.py' 生成采样数据")
        return
    
    # 加载数据
    print(f"加载数据: {input_file}")
    dataset = load_data(input_file)
    
    print(f"数据集包含 {len(dataset)} 个(调制类型, SNR)组合")
    total_samples = sum(len(samples) for samples in dataset.values())
    print(f"总样本数: {total_samples}")
    
    # 应用GPR去噪
    denoised_data, processing_time = apply_gpr_denoising(dataset)
    
    # 保存结果
    save_results(denoised_data, processing_time, args.output)
    
    print("处理完成!")

if __name__ == "__main__":
    main()