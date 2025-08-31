# 从 data/RML2016.10a_dict.pkl 中选取1000条数据保存到 gpr 中.

import numpy as np
import pickle
import os
import random

def load_data(file_path):
    """Load RadioML dataset."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data

def sample_data(dataset, num_samples=1000, random_seed=42):
    """从数据集中随机选取指定数量的样本"""
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    # 获取所有调制类型和SNR值
    mods = sorted(list(set([k[0] for k in dataset.keys()])))
    snrs = sorted(list(set([k[1] for k in dataset.keys()])))
    
    print(f"Found {len(mods)} modulation types: {mods}")
    print(f"Found {len(snrs)} SNR values: {snrs}")
    
    # 收集所有样本
    all_samples = []
    all_keys = []
    
    for key in dataset.keys():
        samples = dataset[key]
        for i, sample in enumerate(samples):
            all_samples.append(sample)
            all_keys.append((key, i))
    
    print(f"Total samples in dataset: {len(all_samples)}")
    
    # 随机选择样本
    selected_indices = random.sample(range(len(all_samples)), min(num_samples, len(all_samples)))
    
    # 构建选取的数据集
    sampled_dataset = {}
    for idx in selected_indices:
        key, sample_idx = all_keys[idx]
        if key not in sampled_dataset:
            sampled_dataset[key] = []
        sampled_dataset[key].append(all_samples[idx])
    
    # 转换为numpy数组
    for key in sampled_dataset:
        sampled_dataset[key] = np.array(sampled_dataset[key])
    
    print(f"Selected {sum(len(v) for v in sampled_dataset.values())} samples")
    print(f"Distribution by (modulation, SNR):")
    for key, samples in sorted(sampled_dataset.items()):
        print(f"  {key}: {len(samples)} samples")
    
    return sampled_dataset

def save_data(data, output_path):
    """保存数据到文件"""
    # 只有当路径包含目录时才创建目录
    dir_path = os.path.dirname(output_path)
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    # 设置路径
    input_file = "../data/RML2016.10a_dict.pkl"
    output_file = "RML2016_1000_samples.pkl"
    
    print("Loading original dataset...")
    dataset = load_data(input_file)
    
    print("Sampling 1000 data points...")
    sampled_data = sample_data(dataset, num_samples=1000)
    
    print("Saving sampled dataset...")
    save_data(sampled_data, output_file)
    
    print("Done!")