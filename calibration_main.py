import re
import math
import json
import torch
import requests
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from opencompass.evaluator import MATHVerifyEvaluator
from datasets import concatenate_datasets

from tqdm import tqdm
from datasets import Dataset

# from math_equ import is_equiv

from concurrent.futures import ThreadPoolExecutor, as_completed
import requests

def load_math_dataset(data_path, num_samples = 500):
    print("Loading MATH dataset...")
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # 保留常用字段
            data.append({
                "problem": item.get("problem", ""),
                "solution": item.get("solution", ""),
                "answer": item.get("answer", "")
            })
    dataset = Dataset.from_list(data[:num_samples])
    print(f"Loaded {len(dataset)} samples.")
    
    return dataset
    
def load_aime_dataset(data_path, num_samples = 30):
    print("Loading AIME24 dataset...")
    df = pd.read_parquet(data_path)
    
    questions = [p[0]["content"] for p in df["prompt"]]
    y_true = np.array([int(rm["ground_truth"]) for rm in df["reward_model"]])
    
    data = []
    for i in range(num_samples):
        data.append({"problem": questions[i],
                     "answer": str(y_true[i])
        })
        
    dataset = Dataset.from_list(data[:num_samples])
    print(f"Loaded {len(dataset)} samples.")
    
    return dataset

def load_aime25_dataset(data_path, num_samples = 30):
    print("Loading AIME25 dataset...")
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            # 保留常用字段
            data.append({
                "problem": item.get("question", ""),
                "answer": item.get("answer", "")
            })
    dataset = Dataset.from_list(data[:num_samples])
    print(f"Loaded {len(dataset)} samples.")
    
    return dataset
    
def load_amc_dataset(data_path, num_samples = 45):
    print("Loading AMC dataset...")

    df = pd.read_parquet(data_path)
    print(df.columns)
    print(df.head(1))

    questions = [p for p in df["problem"]]
    y_true = [rm for rm in df["answer"]]
    data = []
    NUM_SAMPLES = len(questions)
    print(NUM_SAMPLES)
    for i in range(NUM_SAMPLES):
        data.append({"problem": questions[i],
                     "answer": str(y_true[i])
        })
    dataset = Dataset.from_list(data[:num_samples])
    print(f"Loaded {len(dataset)} samples.")
    return dataset

def model_generate(session, model_name, model_url, prompt, max_tokens, gen_type = "verbal"):
    #需要实现一个基于本地部署一个vllm的方法
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 1.5,
        "max_tokens": max_new_tokens,
        "logprobs": True,
        "top_logprobs": 1,
        "chat_template_kwargs": {"enable_thinking": False}
    }

    resp = session.post(model_url, json=payload, timeout=300)
    resp.raise_for_status()
    result = resp.json()

    # vLLM(OpenAI API) 返回格式
    gen_text = result["choices"][0]["message"]["content"]
    
    return gen_text

def get_confidence_and_answer(session, model_name, model_url, prompt, max_tokens, gen_type = 'verbal'):
    directive = (
        "\n\nPlease put your final answer within \boxed{}.\n"
        "\nAlso output a singal line at the end of the answer：CONFIDENCE: <float number between 0 and 1>\n"
        "e.g. ：CONFIDENCE: 0.83\n"
        "Please make sure CONFIDENCE part is in a singal line with the exact same format。\n"
    )

    full_prompt = prompt + directive
    # print("\n\n")
    # print(full_prompt)

    gen_text = model_generate(session, model_name, model_url, full_prompt, max_tokens, gen_type)

    conf = None
    match = re.search(r"CONFIDENCE:\s*([01]?\.\d+|\d+)", gen_text)

    if match:
        try:
            conf = float(match.group(1))
        except:
            conf = -0.2

    if conf is None:
        conf = -0.1

    # 去掉 CONFIDENCE 行只保留回答内容
    answer = re.sub(r"CONFIDENCE:\s*[0-9.]+", "", gen_text).strip()
    #answer = gen_text
    # print(answer)
    return answer, conf, gen_text

def get_answer_and_logit_confidence(session, model_name, model_url, prompt, max_new_tokens):
    # prompt = prompt + "\nPlease reason step by step, and put your final answer within \boxed{}."
    payload = {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "presence_penalty": 1.5,
        "max_tokens": max_new_tokens,
        "logprobs": True,
        "top_logprobs": 1,
        "chat_template_kwargs": {"enable_thinking": False}
    }
    resp = session.post(model_url, json=payload, timeout=300)
    resp.raise_for_status()
    result = resp.json()

    choice = result["choices"][0]
    content = choice["message"]["content"]

    logprob_items = choice["logprobs"]["content"]

    tokens = []
    token_logprobs = []

    for item in logprob_items:
        if item is None:
            continue
        tokens.append(item["token"])
        token_logprobs.append(item["logprob"])

    token_probs = [math.exp(p) for p in token_logprobs]
    avg_logprob = sum(token_logprobs) / len(token_logprobs)
    confidence = math.exp(avg_logprob)
    
    # print(content)

    return content, confidence, token_probs

def map_to_100_percent_average(data):
    """
    将输入列表按位置映射到 100 个百分比桶，
    每个桶的值为该百分比区间内元素的平均值。
    若某个桶为空，则返回 None。
    """
    if not data:
        return [None] * 100

    n = len(data)
    buckets = [[] for _ in range(100)]

    # 将数据按位置分配到百分比桶
    for i, value in enumerate(data):
        for j in range(100):
            percent_index = int(i / n * 100)
            percent_index = min(percent_index, 99)  # 防止边界问题
            buckets[percent_index].append(value)

    # 计算每个桶的平均值
    result = []
    temp = 0
    for bucket in buckets:
        if bucket:
            temp = sum(bucket) / len(bucket)
            result.append(temp)
        else:
            result.append(temp)

    return result

def compute_ece(confidences, accuracies, n_bins=20):
    """计算 Expected Calibration Error (ECE)"""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bins[i], bins[i + 1]

        if i == 0:
            # 第一个 bin 包含 confidence == 0.0
            mask = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            mask = (confidences > bin_lower) & (confidences <= bin_upper)

        if np.any(mask):
            acc = accuracies[mask].mean()
            conf = confidences[mask].mean()
            ece += np.abs(acc - conf) * mask.mean()

    return ece

def compute_overconf_ece(confidences, accuracies, n_bins=20):
    """计算 Over-Confidence Expected Calibration Error"""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bins[i], bins[i + 1]

        if i == 0:
            # 第一个 bin 包含 confidence == 0.0
            mask = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            mask = (confidences > bin_lower) & (confidences <= bin_upper)

        if np.any(mask):
            acc = accuracies[mask].mean()
            conf = confidences[mask].mean()

            # 只在 over-confidence 时累加
            if conf > acc:
                ece += (conf - acc) * mask.mean()

    return ece

def compute_BS(confidences, accuracies, n_bins=20):
    """计算 BS"""
    bs = 0.0
    mask = (confidences >= 0) & (confidences <= 1)
    bs = np.mean((confidences[mask] - accuracies[mask]) ** 2)
    return bs

def compute_MCE(confidences, accuracies, n_bins=20):
    """计算"""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    mce = 0.0

    for i in range(n_bins):
        bin_lower, bin_upper = bins[i], bins[i + 1]

        if i == 0:
            # 第一个 bin 包含 confidence == 0.0
            mask = (confidences >= bin_lower) & (confidences <= bin_upper)
        else:
            mask = (confidences > bin_lower) & (confidences <= bin_upper)

        if np.any(mask):
            acc = accuracies[mask].mean()
            conf = confidences[mask].mean()
            mce = max(mce, np.abs(acc - conf))
    return mce

def compute_auroc(confidences, accuracies):
    """计算 AUROC"""
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(accuracies, confidences)

def evaluate_dataset_parallel(
    dataset,
    model_name,
    model_url,
    conf_type,
    max_tokens,
    num_workers=16,
    repeat=4
):
    records = []
    evaluator = MATHVerifyEvaluator()

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_item = {}

        for i in range(repeat):
            for item in dataset:
                if conf_type == "verbal":
                    fut = executor.submit(
                        get_confidence_and_answer,
                        session,
                        model_name,
                        model_url,
                        item["problem"],
                        max_tokens,
                    )
                else:
                    fut = executor.submit(
                        get_answer_and_logit_confidence,
                        session,
                        model_name,
                        model_url,
                        item["problem"],
                        max_tokens,
                    )
                future_to_item[fut] = item

        for fut in tqdm(as_completed(future_to_item), total=len(future_to_item)):
            item = future_to_item[fut]
            if conf_type == "verbal":
                model_answer, conf, gen_text = fut.result()
            else:
                model_answer, conf, token_probs = fut.result()

            # print(model_answer)
            correctness = evaluator.score(
                [model_answer],
                [item["solution"] if "solution" in item else item["answer"]]
            )
            
            records.append({
                "prediction": model_answer,
                "reference": item["solution"] if "solution" in item else item["answer"],
                "confidence": conf,
                "correct": correctness['accuracy'] / 100,
                "token_probs": token_probs if conf_type == "logits" else None
            })

    # ===== OpenCompass 正确性评估 =====

    confidences = []
    accuracies = []
    token_probs_list = []

    for r in records:
        confidences.append(float(r["confidence"]))
        accuracies.append(float(r["correct"]))
        if conf_type == "logits":
            token_probs_list.append(map_to_100_percent_average(r["token_probs"]))
    
    if conf_type == "logits":
        return np.array(confidences), np.array(accuracies), np.mean(np.array(token_probs_list), axis=0)

    return np.array(confidences), np.array(accuracies), None

def plot_reliability(
    confidences,
    accuracies,
    ece,
    cal_bins,
    save_path,
):
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "axes.linewidth": 0.8,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })

    bins = np.linspace(0.0, 1.0, cal_bins + 1)
    centers = (bins[:-1] + bins[1:]) / 2
    width = 1.0 / cal_bins * 0.9

    bin_acc = np.zeros(cal_bins)
    bin_cnt = np.zeros(cal_bins)

    for i in range(cal_bins):
        if i == 0:
            mask = (confidences >= bins[i]) & (confidences <= bins[i + 1])
        else:
            mask = (confidences > bins[i]) & (confidences <= bins[i + 1])

        bin_cnt[i] = mask.sum()
        if bin_cnt[i] > 0:
            bin_acc[i] = accuracies[mask].mean()

    valid = bin_cnt > 0
    x = centers[valid]
    y = bin_acc[valid]
    cnt = bin_cnt[valid]

    # ===== Color by density (darker = more samples) =====
    norm = (cnt - cnt.min()) / (cnt.max() - cnt.min() + 1e-8)
    colors = cm.Blues(0.3 + 0.7 * norm)

    plt.figure(figsize=(3.25, 3.0), dpi=300)

    # Perfect calibration line
    plt.plot(
        [0, 1], [0, 1],
        linestyle="--",
        linewidth=0.8,
        color="black",
        zorder=1,
    )

    # Reliability bars
    plt.bar(
        x,
        y,
        width=width,
        color=colors,
        edgecolor="black",
        linewidth=0.6,
        zorder=2,
    )

    # Axes
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence Bin")
    plt.ylabel("Accuracy in Bin")

    plt.tight_layout(pad=0.6)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    print("SCRIPT STARTED")
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, default='models/Qwen3-8B')
    args.add_argument("--model_url", type=str, default="http://localhost:8000/v1/chat/completions")
    args.add_argument("--dataset_name", type=str, default='AMC24')
    args.add_argument("--dataset_path", type=str, default='/ceph_home/mazhengzhao2024/data/AMC/data/amc24-00000-of-00001.parquet')
    args.add_argument("--conf_type", type=str, default='verbal')
    args.add_argument("--save_fig", type=bool, default=True)
    args.add_argument("--max_new_tokens", type=int, default=6500)
    args.add_argument("--cal_bins", type=int, default=15)
    args = args.parse_args()
    model_name = args.model_name
    model_url = args.model_url
    dataset_name = args.dataset_name
    dataset_path = args.dataset_path
    conf_type = args.conf_type
    save_fig = args.save_fig
    max_new_tokens = args.max_new_tokens
    cal_bins = args.cal_bins

    session = requests.Session()
    
    print(f"Model: {model_name}, Dataset: {dataset_name}, Confidence Type: {conf_type}")
    if dataset_name == "MATH-500":
        dataset = load_math_dataset(dataset_path)
    if dataset_name == "AIME24":
        dataset = load_aime_dataset(dataset_path)
    if dataset_name == "AIME25":
        dataset = load_aime25_dataset(dataset_path)
    if dataset_name == "AMC24" or dataset_name == "AMC23":
        dataset = load_amc_dataset(dataset_path)
    if dataset_name == "MIXED":
        print("Loading MIXED dataset...")
        dataset_math = load_math_dataset('data/MATH-500/test.jsonl', num_samples=500)
        dataset_aime24 = load_aime_dataset('data/aime24.parquet', num_samples=30)
        dataset_aime25 = load_aime25_dataset('data/aime25/aime2025.jsonl', num_samples=30)
        dataset_amc23 = load_amc_dataset('data/AMC/data/amc23-00000-of-00001.parquet', num_samples=45)
        dataset_amc24 = load_amc_dataset('data/AMC/data/amc24-00000-of-00001.parquet', num_samples=45)
        dataset = concatenate_datasets([
            # dataset_math,
            # dataset_aime24,
            # dataset_aime25,
            dataset_amc23,
            dataset_amc24
        ])

    confidences, accuracies = [], []

    confidences, accuracies, token_probs_list = evaluate_dataset_parallel(
        dataset,
        model_name,
        model_url,
        conf_type,
        max_tokens=max_new_tokens,     # ← 大幅缩小
        num_workers=32,     # 可根据 GPU / vLLM 并发能力调整
    )

    bins = np.linspace(0.0, 1.0, cal_bins + 1)

    frequency = [0] * cal_bins

    for conf in confidences:
        for i in range(cal_bins):
            if i == 0:
                # 第一个 bin 包含 conf == 0.0
                if conf >= bins[i] and conf <= bins[i + 1]:
                    frequency[i] += 1
            else:
                if conf > bins[i] and conf <= bins[i + 1]:
                    frequency[i] += 1

    # print(frequency)

    print(f"Mean Accuracy: {np.mean(accuracies):.4f}")
    # print(confidences, accuracies)
    ece = compute_ece(confidences, accuracies)
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    bs = compute_BS(confidences, accuracies)
    print(f"Brier Score (BS): {bs:.4f}")
    auroc = compute_auroc(confidences, accuracies)
    print(f"Area Under ROC (AUROC): {auroc:.4f}")
    over_ece = compute_overconf_ece(confidences, accuracies)
    print(f"Over Confidence Expected Calibration Error (OverECE): {over_ece:.4f}")

    bin_centers = (bins[:-1] + bins[1:]) / 2

    bin_accs = []
    for i in range(cal_bins):
        if i == 0:
            mask = (confidences >= bins[i]) & (confidences <= bins[i + 1])
        else:
            mask = (confidences > bins[i]) & (confidences <= bins[i + 1])

        if np.any(mask):
            bin_accs.append(accuracies[mask].mean())
        else:
            bin_accs.append(0)

    print(bin_centers)
    print(bin_accs)

    if save_fig:
        plot_reliability(confidences, accuracies,  ece, cal_bins, save_path = f"Figs/{model_name}/{cal_bins}_calibration_{conf_type}_{dataset_name}.pdf",
)
        

