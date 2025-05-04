# plot_utils.py
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

def plot_test_seed(df: pd.DataFrame, outfile="algorithm_comparison.png"):
    """柱状图：各算法在不同随机种子上的 Test 结果"""
    plt.figure(figsize=(10,6))
    offsets = {"ABC": -0.1, "PSO": 0, "PSO-SA": 0.1}
    for algo, grp in df.groupby("Algorithm"):
        plt.bar(grp["Seed"] + offsets[algo], grp["Test"], width=0.2, label=algo)
    plt.xlabel("Seed"); plt.ylabel("Test-period Value")
    plt.title("Algorithm Test Performance across Seeds")
    plt.legend(); plt.tight_layout()
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile); plt.close()

def plot_train_vs_test(summary: pd.DataFrame, outfile="train_test_comparison.png"):
    """双柱图：平均 Train vs Test"""
    plt.figure(figsize=(10,6))
    bar_width = 0.35
    x = range(len(summary))
    plt.bar([i - bar_width/2 for i in x], summary["Train_mean"],
            width=bar_width, label="Train")
    plt.bar([i + bar_width/2 for i in x], summary["Test_mean"],
            width=bar_width, label="Test")
    plt.xticks(x, summary["Algorithm"])
    plt.ylabel("Value")
    plt.title("Mean Train vs Test Performance by Algorithm")
    plt.legend(); plt.tight_layout()
    Path(outfile).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(outfile); plt.close()

def plot_convergence_curves(curve_dict: dict, outfile="convergence_curves.png"):
    """绘制每个算法在每个种子下的收敛过程"""
    plt.figure(figsize=(10, 6))
    for label, curve in curve_dict.items():
        plt.plot(curve, label=label)
    plt.xlabel("Iteration")
    plt.ylabel("Best Fitness So Far")
    plt.title("Convergence Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()

def plot_equity_curves(equity_dict: dict, outfile="equity_curves.png"):
    """绘制每个算法每个种子对应的净值曲线"""
    plt.figure(figsize=(10, 6))
    for label, curve in equity_dict.items():
        plt.plot(curve, label=label)
    plt.xlabel("Time")
    plt.ylabel("Equity Value")
    plt.title("Equity Curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close()