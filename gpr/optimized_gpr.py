# 高效GPR去噪（批处理 + 重用核分解）
# 基于 gpr/origin.py 的逻辑进行优化：
# - 仍按 (modulation, SNR) 分组处理，length_scale 与 origin 保持一致
# - 将同一组内所有样本按时间轴堆叠成矩阵，一次性求解（多右端三角求解）
# - 对噪声方差采用“组级估计（per-key）”作为默认模式，极大减少重复分解
# - 可切换为“逐样本估计（per-sample）”以获得与 origin 更一致的结果（但更慢）

import argparse
import os
import pickle
import time
from typing import Dict, Tuple

import numpy as np


# ----------------------------
# 数据加载/保存
# ----------------------------

def load_data(file_path: str):
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


def save_results(denoised_data: Dict[Tuple[str, int], np.ndarray], processing_time: float, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    data_file = os.path.join(output_dir, 'denoised_data.pkl')
    with open(data_file, 'wb') as f:
        pickle.dump(denoised_data, f)
    print(f'去噪后数据已保存到: {data_file}')

    total_samples = sum(len(v) for v in denoised_data.values())
    time_file = os.path.join(output_dir, 'processing_time.txt')
    with open(time_file, 'w') as f:
        f.write(f'GPR去噪处理时间: {processing_time:.2f} 秒\n')
        f.write(f'处理样本数: {total_samples}\n')
        f.write(f'平均每样本处理时间: {processing_time/max(total_samples,1):.6f} 秒\n')
    print(f'处理时间信息已保存到: {time_file}')


# ----------------------------
# 核函数与GPR辅助
# ----------------------------

def calculate_power(i_component: np.ndarray, q_component: np.ndarray) -> float:
    return float(np.mean(i_component ** 2 + q_component ** 2))


def estimate_noise_std(signal_power: float, snr_db: float) -> float:
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / (snr_linear + 1)
    return float(np.sqrt(noise_power / 2))


def length_scale_from_snr(snr_db: float) -> float:
    # 与 origin.py 相同
    return 5.0 if snr_db >= 0 else min(10.0, 5.0 - snr_db * 0.25)


def rbf_kernel_same_grid(n: int, length_scale: float) -> np.ndarray:
    # X = [[0],[1],...,[n-1]]
    idx = np.arange(n, dtype=np.float64)
    # (i-j)^2 矩阵
    d2 = (idx[:, None] - idx[None, :]) ** 2
    ls2 = (length_scale ** 2)
    K = np.exp(-0.5 * d2 / max(ls2, 1e-12))
    return K


def batched_gp_denoise_same_inputs(
    K_no_noise: np.ndarray,
    Y: np.ndarray,
    noise_var: float,
    jitter: float = 1e-8,
) -> np.ndarray:
    """
    对训练点自身求后验均值（X* = X），多列Y批量求解。
    K_no_noise: (n,n) 无噪声核矩阵 k(X,X)
    Y: (n,m) 多个目标列（可同时包含多个样本的实部与虚部）
    noise_var: 标量噪声方差 sigma^2
    返回: (n,m) 后验均值矩阵
    公式：mean = K @ (K + sigma^2 I)^{-1} Y
    """
    n = K_no_noise.shape[0]
    Ky = K_no_noise + (noise_var + jitter) * np.eye(n, dtype=np.float64)
    # Cholesky 分解
    L = np.linalg.cholesky(Ky)
    # 解 alpha = Ky^{-1} Y，使用两次三角求解（多右端）
    v = np.linalg.solve(L, Y)
    alpha = np.linalg.solve(L.T, v)
    # 训练点的均值：K @ alpha
    mean = K_no_noise @ alpha
    return mean


# ----------------------------
# 主流程（分组 + 批处理）
# ----------------------------

def apply_gpr_denoising_efficient(
    dataset: Dict[Tuple[str, int], np.ndarray],
    noise_mode: str = 'per-key',  # 'per-key' 或 'per-sample'
    batch_limit: int = 4096,      # 控制单次列数，防止极端情况下内存峰值
) -> Tuple[Dict[Tuple[str, int], np.ndarray], float]:
    """
    返回 (denoised_dataset, total_time)
    dataset[key] 形状: (num_samples, 2, n)
    """
    assert noise_mode in {'per-key', 'per-sample'}

    total_samples = sum(len(v) for v in dataset.values())
    processed = 0

    t0 = time.time()
    denoised_dataset: Dict[Tuple[str, int], np.ndarray] = {}

    print(f'开始高效GPR去噪，总样本数: {total_samples}')

    for key, samples in dataset.items():
        mod, snr_db = key
        num = len(samples)
        if num == 0:
            denoised_dataset[key] = samples
            continue

        n = samples.shape[-1]
        ls = length_scale_from_snr(float(snr_db))
        K = rbf_kernel_same_grid(n, ls)

        # 估计噪声方差
        if noise_mode == 'per-key':
            # 使用该组样本的功率的中位数进行估计，兼顾稳健性与速度
            powers = np.mean(samples[:, 0, :] ** 2 + samples[:, 1, :] ** 2, axis=1)
            median_power = float(np.median(powers))
            sigma = estimate_noise_std(median_power, float(snr_db))
            noise_var = sigma ** 2
            # 全组一次（或分块）求解
            # 将实部与虚部拼为列：Y = [y1_real, y1_imag, y2_real, y2_imag, ...]
            Y = np.empty((n, num * 2), dtype=np.float64)
            Y[:, 0::2] = samples[:, 0, :].T  # 实部列
            Y[:, 1::2] = samples[:, 1, :].T  # 虚部列

            # 分块以控制内存峰值
            denoised_cols = np.empty_like(Y)
            for start in range(0, Y.shape[1], batch_limit):
                end = min(start + batch_limit, Y.shape[1])
                denoised_cols[:, start:end] = batched_gp_denoise_same_inputs(K, Y[:, start:end], noise_var)

            # 还原为样本数组
            denoised = np.empty_like(samples)
            denoised[:, 0, :] = denoised_cols[:, 0::2].T
            denoised[:, 1, :] = denoised_cols[:, 1::2].T

        else:  # per-sample: 与 origin 更一致，但要为每个样本单独分解 Ky
            denoised = np.empty_like(samples)
            # 仍然可以小批量进行（每个样本噪声不同 -> 不能共享分解，但可共享 K_no_noise）
            for i, sample in enumerate(samples):
                i_comp = sample[0, :]
                q_comp = sample[1, :]
                power = calculate_power(i_comp, q_comp)
                sigma = estimate_noise_std(power, float(snr_db))
                noise_var = sigma ** 2
                Y = np.stack([i_comp, q_comp], axis=1).astype(np.float64)  # (n,2)
                mu = batched_gp_denoise_same_inputs(K, Y, noise_var)      # (n,2)
                denoised[i, 0, :] = mu[:, 0]
                denoised[i, 1, :] = mu[:, 1]

                processed += 1
                if processed % 200 == 0:
                    elapsed = time.time() - t0
                    print(f'进度: {processed}/{total_samples} ({processed/total_samples*100:.1f}%), 用时: {elapsed:.1f}s')

        denoised_dataset[key] = denoised
        processed += len(samples) if noise_mode == 'per-key' else 0
        elapsed = time.time() - t0
        print(f'完成 {mod}@{snr_db}dB: {len(samples)} 个样本，用时累计: {elapsed:.1f}s')

    total_time = time.time() - t0
    print(f'高效GPR去噪完成，总用时: {total_time:.2f} 秒')
    return denoised_dataset, total_time


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description='高效GPR去噪（批处理 + 重用核分解）')
    parser.add_argument('--source', choices=['original', 'sampled'], default='sampled',
                        help='数据源选择: original(原数据集) 或 sampled(1000条采样数据)')
    parser.add_argument('--output', default='gpr/efficient_results',
                        help='输出目录（建议位于 gpr/efficient_results）')
    parser.add_argument('--noise-mode', choices=['per-key', 'per-sample'], default='per-key',
                        help='噪声估计模式: per-key(默认，速度快) 或 per-sample(更接近 origin，较慢)')
    parser.add_argument('--batch-limit', type=int, default=4096,
                        help='批处理列数上限，用于控制内存峰值，默认4096')

    args = parser.parse_args()

    # 选择数据源
    if args.source == 'original':
        input_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'RML2016.10a_dict.pkl')
        print('使用原始数据集进行高效GPR去噪...')
    else:
        input_file = os.path.join(os.path.dirname(__file__), 'RML2016_1000_samples.pkl')
        print('使用1000条采样数据进行高效GPR去噪...')

    input_file = os.path.normpath(input_file)

    if not os.path.exists(input_file):
        print(f'错误: 找不到输入文件 {input_file}')
        if args.source == 'sampled':
            print("请先运行 'python gpr/data.py' 生成采样数据")
        return

    # 加载
    print(f'加载数据: {input_file}')
    dataset = load_data(input_file)
    print(f'数据集包含 {len(dataset)} 个 (调制, SNR) 组合，总样本数: {sum(len(v) for v in dataset.values())}')

    # 去噪
    denoised_data, total_time = apply_gpr_denoising_efficient(
        dataset,
        noise_mode=args.noise_mode,
        batch_limit=max(1, int(args.batch_limit)),
    )

    # 输出目录
    out_dir = args.output
    if not os.path.isabs(out_dir):
        out_dir = os.path.normpath(os.path.join(os.getcwd(), out_dir))

    save_results(denoised_data, total_time, out_dir)
    print('处理完成!')


if __name__ == '__main__':
    main()