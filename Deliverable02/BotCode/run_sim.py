from trading_bot import TradingBotOptimiser, TradingBotInstance
from optimisers.artificial_bee_colony import ArtificialBeeColony   # 这是继承 Optimiser 的版本
from optimisers.pso import PSO
from optimisers.pso_sa import PSOSA
import pandas as pd
import matplotlib.pyplot as plt
import plot_utils as putils
from plot_utils import plot_convergence_curves, plot_equity_curves

plt.switch_backend("Agg")  # avoid blocking if no display

dataset_path = "/Users/chenzijian/codespace/cits-4404-project/Deliverable02/Data/btc_train.csv"   # 确认路径正确
test_file  = "/Users/chenzijian/codespace/cits-4404-project/Deliverable02/Data/btc_test.csv"

# --- benchmark configuration ---
iter_per_run = 20        # iterations inside each optimiser run (was 50)
seeds = [1, 7, 13, 21, 42, 50, 65, 77, 88, 99]  # five seeds for stability check
# --------------------------------

# seeds defined above
algos = [
    ("ABC", lambda seed: ArtificialBeeColony(
        search_space=None, rng_seed=seed,
        population_size=30, pos_age_limit=10)),
    ("PSO", lambda seed: PSO(
        search_space=None, rng_seed=seed,
        population_size=30, p_increment=0.5, g_increment=0.5)),
    ("PSO-SA", lambda seed: PSOSA(search_space=None, rng_seed=seed,
      population_size=30, p_increment=0.5, g_increment=0.5,
      iters=20, max_inner=200))              # ← 关键字 iters
]

results = []
convergence_data = {}
equity_data = {}
import contextlib, io

for algo_name, algo_factory in algos:
    for seed in seeds:
        print(f"\n--> {algo_name}  seed={seed}")
        algo = algo_factory(seed)
        opt = TradingBotOptimiser(dataset_path, algo)
        # suppress internal verbose prints from optimiser
        with contextlib.redirect_stdout(io.StringIO()):
            params, train_score, convergence_curve = opt.optimise(iter=iter_per_run, return_curve=True)
        bot = TradingBotInstance(test_file, *params)
        test_score, equity_curve = bot.simulate_run(return_equity=True)
        results.append({"Algorithm": algo_name, "Seed": seed,
                        "Train": float(train_score), "Test": float(test_score)})
        label = f"{algo_name}_seed_{seed}"
        convergence_data[label] = convergence_curve
        equity_data[label] = equity_curve

# Save convergence and equity data for future analysis
import pickle
with open("convergence_data.pkl", "wb") as f:
    pickle.dump(convergence_data, f)
with open("equity_data.pkl", "wb") as f:
    pickle.dump(equity_data, f)

df = pd.DataFrame(results)
# Save results
df.to_csv("comparison_results.csv", index=False)

# Plot
putils.plot_test_seed(df, "algorithm_comparison.png")
# ---- summary statistics ----
summary = df.groupby("Algorithm").agg(
    Train_mean=("Train", "mean"),
    Train_std=("Train", "std"),
    Test_mean=("Test", "mean"),
    Test_std=("Test", "std")
).reset_index()
summary.to_csv("summary_stats.csv", index=False)
print("\n=== Mean / Std by Algorithm ===")
print(summary)

# ---- combined Train/Test bar plot ----
putils.plot_train_vs_test(summary, "train_test_comparison.png")

plot_convergence_curves(convergence_data, "convergence_curves.png")
plot_equity_curves(equity_data, "equity_curves.png")