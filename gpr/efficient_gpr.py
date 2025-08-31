# 优化GPR去噪过程，缩短时间，实现高效GPR去噪

import numpy as np
import pickle
import os
import argparse
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from multiprocessing import cpu_count
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import warnings
warnings.filterwarnings("ignore")

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

def apply_gp_regression_fast(complex_signal, noise_std, length_scale=5.0, subsample_factor=2):
    """
    优化的GPR去噪，使用子采样和快速预测加速处理
    Args:
        complex_signal: 复数信号
        noise_std: 噪声标准差
        length_scale: 长度参数
        subsample_factor: 子采样因子，减少训练点数量
    """
    signal_len = len(complex_signal)
    
    # 子采样训练数据以加速GPR训练
    if signal_len > 64:  # 只对长信号进行子采样
        train_indices = np.arange(0, signal_len, subsample_factor)
        X_train = train_indices.reshape(-1, 1)
        X_predict = np.arange(signal_len).reshape(-1, 1)
    else:
        X_train = np.arange(signal_len).reshape(-1, 1)
        X_predict = X_train
        train_indices = np.arange(signal_len)
    
    y_real_train = complex_signal.real[train_indices]
    y_imag_train = complex_signal.imag[train_indices]
    
    # 使用固定的核参数避免优化
    kernel = RBF(length_scale=length_scale, length_scale_bounds="fixed")
    
    # 处理实部
    gpr_real = GaussianProcessRegressor(
        kernel=kernel, 
        alpha=noise_std**2, 
        normalize_y=True,
        optimizer=None  # 禁用超参数优化
    )
    gpr_real.fit(X_train, y_real_train)
    y_real_denoised, _ = gpr_real.predict(X_predict, return_std=True)
    
    # 处理虚部
    gpr_imag = GaussianProcessRegressor(
        kernel=kernel, 
        alpha=noise_std**2, 
        normalize_y=True,
        optimizer=None  # 禁用超参数优化
    )
    gpr_imag.fit(X_train, y_imag_train)
    y_imag_denoised, _ = gpr_imag.predict(X_predict, return_std=True)
    
    return y_real_denoised + 1j * y_imag_denoised

def process_single_sample(args):
    """处理单个样本的函数，用于并行处理"""
    sample, snr_val = args
    
    i_component = sample[0, :]
    q_component = sample[1, :]
    complex_signal = i_component + 1j * q_component
    
    # 计算功率和噪声标准差
    total_power = calculate_power(i_component, q_component)
    noise_std = estimate_noise_std(total_power, snr_val)
    
    # 根据SNR和信号长度动态调整参数
    signal_len = len(complex_signal)
    length_scale_val = 5.0 if snr_val >= 0 else min(10, 5.0 - snr_val * 0.25)
    
    # 根据信号长度调整子采样因子
    if signal_len <= 64:
        subsample_factor = 1
    elif signal_len <= 128:
        subsample_factor = 2
    else:
        subsample_factor = 3
    
    # 应用快速GPR去噪
    denoised_signal = apply_gp_regression_fast(
        complex_signal, 
        noise_std, 
        length_scale=length_scale_val,
        subsample_factor=subsample_factor
    )
    
    # 重构去噪后的样本
    denoised_sample = np.zeros_like(sample)
    denoised_sample[0, :] = np.real(denoised_signal)
    denoised_sample[1, :] = np.imag(denoised_signal)
    
    return denoised_sample

def process_samples_batch(samples_and_snr, batch_size=50):
    """批量处理样本，使用线程池"""
    results = []
    
    with ThreadPoolExecutor(max_workers=min(4, cpu_count())) as executor:
        # 分批处理以控制内存使用
        for i in range(0, len(samples_and_snr), batch_size):
            batch = samples_and_snr[i:i+batch_size]
            batch_results = list(executor.map(process_single_sample, batch))
            results.extend(batch_results)
            
            # 显示进度
            progress = min(i + batch_size, len(samples_and_snr))
            print(f"  批处理进度: {progress}/{len(samples_and_snr)}")
    
    return results

def apply_efficient_gpr_denoising(dataset, use_parallel=True, batch_size=100):
    """高效GPR去噪实现"""
    denoised_dataset = {}
    total_samples = sum(len(samples) for samples in dataset.values())
    processed_samples = 0
    start_time = time.time()
    
    print(f"开始高效GPR处理 {total_samples} 个样本...")
    print(f"并行处理: {'是' if use_parallel else '否'}")
    print(f"CPU核心数: {cpu_count()}")
    
    for key, samples in dataset.items():
        mod, snr_val = key
        print(f"处理 {mod} at {snr_val}dB: {len(samples)} 个样本")
        
        if use_parallel and len(samples) > 10:
            # 准备并行处理的数据
            samples_and_snr = [(sample, snr_val) for sample in samples]
            denoised_samples = process_samples_batch(samples_and_snr, batch_size)
        else:
            # 串行处理小批量数据
            denoised_samples = []
            for sample in samples:
                denoised_sample = process_single_sample((sample, snr_val))
                denoised_samples.append(denoised_sample)
        
        denoised_dataset[key] = np.array(denoised_samples)
        processed_samples += len(samples)
        
        elapsed_time = time.time() - start_time
        progress = processed_samples / total_samples * 100
        samples_per_second = processed_samples / elapsed_time if elapsed_time > 0 else 0
        print(f"整体进度: {processed_samples}/{total_samples} ({progress:.1f}%), "
              f"用时: {elapsed_time:.1f}s, 速度: {samples_per_second:.2f} samples/s")
    
    total_time = time.time() - start_time
    print(f"高效GPR去噪完成！")
    print(f"总用时: {total_time:.2f} 秒")
    print(f"平均速度: {total_samples/total_time:.2f} samples/s")
    
    return denoised_dataset, total_time

def apply_ultra_fast_gpr(dataset, target_samples_per_key=50):
    """超快速GPR去噪 - 只处理每个键的部分样本进行演示"""
    denoised_dataset = {}
    total_processed = 0
    start_time = time.time()
    
    print(f"开始超快速GPR处理...")
    print(f"每个(调制类型,SNR)组合最多处理 {target_samples_per_key} 个样本")
    
    for key, samples in dataset.items():
        mod, snr_val = key
        
        # 限制处理的样本数量以加速演示
        samples_to_process = min(len(samples), target_samples_per_key)
        selected_samples = samples[:samples_to_process]
        
        print(f"处理 {mod} at {snr_val}dB: {samples_to_process} 个样本 (共{len(samples)}个)")
        
        # 使用简化的GPR参数
        denoised_samples = []
        for sample in selected_samples:
            i_component = sample[0, :]
            q_component = sample[1, :]
            complex_signal = i_component + 1j * q_component
            
            total_power = calculate_power(i_component, q_component)
            noise_std = estimate_noise_std(total_power, snr_val)
            
            # 使用大的子采样因子和简化参数
            denoised_signal = apply_gp_regression_fast(
                complex_signal, 
                noise_std, 
                length_scale=5.0,
                subsample_factor=4  # 更大的子采样因子
            )
            
            denoised_sample = np.zeros_like(sample)
            denoised_sample[0, :] = np.real(denoised_signal)
            denoised_sample[1, :] = np.imag(denoised_signal)
            denoised_samples.append(denoised_sample)
        
        denoised_dataset[key] = np.array(denoised_samples)
        total_processed += samples_to_process
    
    total_time = time.time() - start_time
    print(f"超快速GPR去噪完成！")
    print(f"总用时: {total_time:.2f} 秒")
    print(f"处理样本: {total_processed} 个")
    print(f"平均速度: {total_processed/total_time:.2f} samples/s")
    
    return denoised_dataset, total_time

def save_results(denoised_data, processing_time, output_dir, method_name):
    """保存去噪结果和处理时间"""
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存去噪后的数据
    data_file = os.path.join(output_dir, f"{method_name}_denoised_data.pkl")
    with open(data_file, 'wb') as f:
        pickle.dump(denoised_data, f)
    print(f"去噪后数据已保存到: {data_file}")
    
    # 保存处理时间信息
    total_samples = sum(len(v) for v in denoised_data.values())
    time_file = os.path.join(output_dir, f"{method_name}_processing_time.txt")
    with open(time_file, 'w') as f:
        f.write(f"{method_name} GPR去噪处理时间: {processing_time:.2f} 秒\n")
        f.write(f"处理样本数: {total_samples}\n")
        f.write(f"平均每样本处理时间: {processing_time/total_samples:.4f} 秒\n")
        f.write(f"处理速度: {total_samples/processing_time:.2f} samples/s\n")
    print(f"处理时间信息已保存到: {time_file}")

def main():
    parser = argparse.ArgumentParser(description='高效GPR去噪处理脚本')
    parser.add_argument('--source', choices=['original', 'sampled'], default='sampled',
                        help='数据源选择: original (原数据集) 或 sampled (1000条采样数据)')
    parser.add_argument('--output', default='efficient_results', 
                        help='输出目录 (默认: efficient_results)')
    parser.add_argument('--method', choices=['efficient', 'ultra_fast'], default='efficient',
                        help='处理方法: efficient (高效) 或 ultra_fast (超快速)')
    parser.add_argument('--no-parallel', action='store_true',
                        help='禁用并行处理')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='批处理大小 (默认: 100)')
    
    args = parser.parse_args()
    
    # 根据参数选择数据源
    if args.source == 'original':
        input_file = "../data/RML2016.10a_dict.pkl"
        print("使用原始数据集进行高效GPR去噪...")
    else:
        input_file = "RML2016_1000_samples.pkl"
        print("使用1000条采样数据进行高效GPR去噪...")
    
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
    
    # 选择处理方法
    if args.method == 'ultra_fast':
        print("使用超快速模式 (处理部分样本用于演示)")
        denoised_data, processing_time = apply_ultra_fast_gpr(dataset)
        method_name = 'ultra_fast'
    else:
        print("使用高效模式")
        use_parallel = not args.no_parallel
        denoised_data, processing_time = apply_efficient_gpr_denoising(
            dataset, 
            use_parallel=use_parallel,
            batch_size=args.batch_size
        )
        method_name = 'efficient'
    
    # 保存结果
    save_results(denoised_data, processing_time, args.output, method_name)
    
    print("处理完成!")
    print(f"\n性能比较:")
    print(f"方法: {method_name}")
    print(f"总用时: {processing_time:.2f} 秒")
    processed_samples = sum(len(v) for v in denoised_data.values())
    print(f"处理样本: {processed_samples}")
    print(f"平均速度: {processed_samples/processing_time:.2f} samples/s")

if __name__ == "__main__":
    main()