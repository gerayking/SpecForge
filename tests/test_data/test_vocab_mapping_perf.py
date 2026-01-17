# coding=utf-8
"""
测试 vocab mapping token 统计的正确性和性能

用法:
    python tests/test_vocab_mapping_perf.py
"""

import time
from collections import Counter
from typing import Dict, List, Tuple
from functools import partial
import multiprocessing as mp

import torch
from tqdm import tqdm


# ==============================
# 模拟数据集生成
# ==============================
def generate_mock_dataset(
    num_samples: int = 460000,
    seq_length: int = 2048,
    vocab_size: int = 152000,
    loss_mask_ratio: float = 0.3,
    seed: int = 42,
) -> List[Dict[str, torch.Tensor]]:
    """
    生成模拟数据集用于测试

    Args:
        num_samples: 样本数量
        seq_length: 序列长度
        vocab_size: 词表大小
        loss_mask_ratio: loss_mask 中为 1 的比例
        seed: 随机种子
    """
    torch.manual_seed(seed)
    
    dataset = []
    for _ in tqdm(range(num_samples), desc="Generating mock dataset"):
        # 实际数据 shape 是 (1, seq_length)
        input_ids = torch.randint(0, vocab_size, (1, seq_length))
        # 模拟 loss_mask：后半部分更可能为 1（模拟 assistant 回复）
        loss_mask = torch.zeros(1, seq_length, dtype=torch.long)
        start_idx = int(seq_length * (1 - loss_mask_ratio))
        loss_mask[0, start_idx:] = 1
        
        dataset.append({
            "input_ids": input_ids,
            "loss_mask": loss_mask,
        })
    
    return dataset


# ==============================
# 原始实现 (单进程)
# ==============================
def count_tokens_original(dataset: List[Dict[str, torch.Tensor]]) -> Counter:
    """原始的单进程实现"""
    token_dict = Counter()
    for item in tqdm(dataset, desc="Counting tokens (original)"):
        input_ids = item["input_ids"]
        loss_mask = item["loss_mask"]
        import pdb; pdb.set_trace()
        masked_ids = input_ids[loss_mask == 1]
        unique_ids, counts = masked_ids.unique(return_counts=True)
        batch_token_dict = dict(zip(unique_ids.tolist(), counts.tolist()))
        token_dict.update(batch_token_dict)
    return token_dict


# ==============================
# 优化实现 1: 批处理 + 向量化
# ==============================
def count_tokens_vectorized(dataset: List[Dict[str, torch.Tensor]]) -> Counter:
    """向量化批处理实现"""
    # 预先收集所有 masked tokens
    all_masked_ids = []
    for item in tqdm(dataset, desc="Counting tokens (vectorized)"):
        input_ids = item["input_ids"]
        loss_mask = item["loss_mask"]
        masked_ids = input_ids[loss_mask == 1]
        all_masked_ids.append(masked_ids)
    
    # 合并并统计
    all_tokens = torch.cat(all_masked_ids)
    unique_ids, counts = all_tokens.unique(return_counts=True)
    token_dict = Counter(dict(zip(unique_ids.tolist(), counts.tolist())))
    
    return token_dict


# ==============================
# 优化实现 2: 多进程
# ==============================
def _count_tokens_worker(items: List[Dict[str, torch.Tensor]]) -> Counter:
    """多进程 worker 函数"""
    local_counter = Counter()
    for item in items:
        input_ids = item["input_ids"]
        loss_mask = item["loss_mask"]
        masked_ids = input_ids[loss_mask == 1]
        unique_ids, counts = masked_ids.unique(return_counts=True)
        batch_token_dict = dict(zip(unique_ids.tolist(), counts.tolist()))
        local_counter.update(batch_token_dict)
    return local_counter


def count_tokens_multiprocess(
    dataset: List[Dict[str, torch.Tensor]],
    num_workers: int = 4,
) -> Counter:
    """多进程实现"""
    # 将数据集分成 chunks
    chunk_size = len(dataset) // num_workers
    chunks = []
    for i in range(num_workers):
        start = i * chunk_size
        end = start + chunk_size if i < num_workers - 1 else len(dataset)
        chunks.append(dataset[start:end])
    
    print(f"Using {num_workers} workers, chunk sizes: {[len(c) for c in chunks]}")
    
    # 多进程处理
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(_count_tokens_worker, chunks),
            total=num_workers,
            desc="Counting tokens (multiprocess)",
        ))
    
    # 合并结果
    token_dict = Counter()
    for result in results:
        token_dict.update(result)
    
    return token_dict


# ==============================
# 优化实现 3: 多进程 + 向量化
# ==============================
def _count_tokens_worker_vectorized(items: List[Dict[str, torch.Tensor]]) -> Counter:
    """多进程 worker 函数 (向量化版)"""
    all_masked_ids = []
    for item in items:
        input_ids = item["input_ids"]
        loss_mask = item["loss_mask"]
        masked_ids = input_ids[loss_mask == 1]
        all_masked_ids.append(masked_ids)
    
    if all_masked_ids:
        all_tokens = torch.cat(all_masked_ids)
        unique_ids, counts = all_tokens.unique(return_counts=True)
        return Counter(dict(zip(unique_ids.tolist(), counts.tolist())))
    return Counter()


def count_tokens_multiprocess_vectorized(
    dataset: List[Dict[str, torch.Tensor]],
    num_workers: int = 4,
) -> Counter:
    """多进程 + 向量化实现"""
    # 将数据集分成 chunks
    chunk_size = len(dataset) // num_workers
    chunks = []
    for i in range(num_workers):
        start = i * chunk_size
        end = start + chunk_size if i < num_workers - 1 else len(dataset)
        chunks.append(dataset[start:end])
    
    print(f"Using {num_workers} workers, chunk sizes: {[len(c) for c in chunks]}")
    
    # 多进程处理
    with mp.Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(_count_tokens_worker_vectorized, chunks),
            total=num_workers,
            desc="Counting tokens (mp+vectorized)",
        ))
    
    # 合并结果
    token_dict = Counter()
    for result in results:
        token_dict.update(result)
    
    return token_dict


# ==============================
# 正确性验证
# ==============================
def verify_correctness(
    result: Counter,
    baseline: Counter,
    method_name: str,
) -> bool:
    """验证结果正确性"""
    if result == baseline:
        print(f"✅ {method_name}: 结果正确")
        return True
    else:
        # 详细对比差异
        diff_keys = set(result.keys()) ^ set(baseline.keys())
        if diff_keys:
            print(f"❌ {method_name}: Key 差异: {len(diff_keys)} 个不同的 key")
        
        common_keys = set(result.keys()) & set(baseline.keys())
        diff_values = sum(1 for k in common_keys if result[k] != baseline[k])
        if diff_values:
            print(f"❌ {method_name}: Value 差异: {diff_values} 个 key 的值不同")
        
        return False


# ==============================
# 性能测试
# ==============================
def benchmark(
    func,
    dataset: List[Dict[str, torch.Tensor]],
    method_name: str,
    num_runs: int = 3,
    **kwargs,
) -> Tuple[Counter, float, float]:
    """
    性能基准测试

    Returns:
        result: Counter 结果
        avg_time: 平均耗时
        std_time: 耗时标准差
    """
    times = []
    result = None
    
    for i in range(num_runs):
        start = time.perf_counter()
        result = func(dataset, **kwargs)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {i+1}/{num_runs}: {elapsed:.3f}s")
    
    avg_time = sum(times) / len(times)
    std_time = (sum((t - avg_time) ** 2 for t in times) / len(times)) ** 0.5
    
    return result, avg_time, std_time


def main():
    print("=" * 80)
    print("Vocab Mapping Token 统计 - 正确性与性能测试")
    print("=" * 80)
    
    # 配置参数
    NUM_SAMPLES = 460000  # 样本数量，可以根据需要调整
    SEQ_LENGTH = 1203   # 序列长度
    VOCAB_SIZE = 152000 # 词表大小 (Qwen 词表)
    NUM_RUNS = 3        # 每个方法运行次数
    NUM_WORKERS = 4     # 多进程 worker 数量
    
    print(f"\n测试配置:")
    print(f"  - 样本数量: {NUM_SAMPLES}")
    print(f"  - 序列长度: {SEQ_LENGTH}")
    print(f"  - 词表大小: {VOCAB_SIZE}")
    print(f"  - 运行次数: {NUM_RUNS}")
    print(f"  - Worker 数: {NUM_WORKERS}")
    print(f"  - CPU 核心数: {mp.cpu_count()}")
    
    # 生成模拟数据集
    print("\n" + "=" * 80)
    print("Step 1: 生成模拟数据集")
    print("=" * 80)
    dataset = generate_mock_dataset(
        num_samples=NUM_SAMPLES,
        seq_length=SEQ_LENGTH,
        vocab_size=VOCAB_SIZE,
    )
    print(f"数据集大小: {len(dataset)} 样本")
    
    # 性能测试
    results = {}
    
    # 1. 原始实现 (baseline)
    print("\n" + "=" * 80)
    print("Step 2: 原始实现 (baseline)")
    print("=" * 80)
    baseline, baseline_time, baseline_std = benchmark(
        count_tokens_original,
        dataset,
        "original",
        num_runs=NUM_RUNS,
    )
    results["original"] = {
        "result": baseline,
        "avg_time": baseline_time,
        "std_time": baseline_std,
    }
    print(f"\n原始实现: {baseline_time:.3f}s ± {baseline_std:.3f}s")
    print(f"Token 种类数: {len(baseline)}")
    print(f"Token 总数: {sum(baseline.values())}")
    
    # 2. 向量化实现
    print("\n" + "=" * 80)
    print("Step 3: 向量化实现")
    print("=" * 80)
    vectorized_result, vectorized_time, vectorized_std = benchmark(
        count_tokens_vectorized,
        dataset,
        "vectorized",
        num_runs=NUM_RUNS,
    )
    results["vectorized"] = {
        "result": vectorized_result,
        "avg_time": vectorized_time,
        "std_time": vectorized_std,
    }
    verify_correctness(vectorized_result, baseline, "向量化实现")
    
    # 3. 多进程实现
    print("\n" + "=" * 80)
    print("Step 4: 多进程实现")
    print("=" * 80)
    mp_result, mp_time, mp_std = benchmark(
        count_tokens_multiprocess,
        dataset,
        "multiprocess",
        num_runs=NUM_RUNS,
        num_workers=NUM_WORKERS,
    )
    results["multiprocess"] = {
        "result": mp_result,
        "avg_time": mp_time,
        "std_time": mp_std,
    }
    verify_correctness(mp_result, baseline, "多进程实现")
    
    # 4. 多进程 + 向量化实现
    print("\n" + "=" * 80)
    print("Step 5: 多进程 + 向量化实现")
    print("=" * 80)
    mp_vec_result, mp_vec_time, mp_vec_std = benchmark(
        count_tokens_multiprocess_vectorized,
        dataset,
        "multiprocess_vectorized",
        num_runs=NUM_RUNS,
        num_workers=NUM_WORKERS,
    )
    results["multiprocess_vectorized"] = {
        "result": mp_vec_result,
        "avg_time": mp_vec_time,
        "std_time": mp_vec_std,
    }
    verify_correctness(mp_vec_result, baseline, "多进程+向量化实现")
    
    # 汇总结果
    print("\n" + "=" * 80)
    print("性能对比汇总")
    print("=" * 80)
    print(f"\n{'方法':<25} {'平均耗时':>12} {'标准差':>10} {'加速比':>10}")
    print("-" * 60)
    
    for method, data in results.items():
        speedup = baseline_time / data["avg_time"]
        print(f"{method:<25} {data['avg_time']:>10.3f}s {data['std_time']:>10.3f}s {speedup:>9.2f}x")
    
    print("\n" + "=" * 80)
    print("测试完成!")
    print("=" * 80)


if __name__ == "__main__":
    # 设置多进程启动方式 (macOS 需要 spawn)
    mp.set_start_method("spawn", force=True)
    main()

