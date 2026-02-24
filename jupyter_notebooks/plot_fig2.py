import matplotlib.pyplot as plt
from scipy.stats import wilcoxon, ttest_rel
import pandas as pd
import numpy as np

def run_plot_fig2(confocal_df=None, colours=None):

    # Filter for pressure == 0.0 (baseline)
    df_box = confocal_df[confocal_df["Pressure"] == 0.0][[
        "Mesh ID",
        "Midsection AR left",
        "Midsection AR right",
        "Tip AR left",
        "Tip AR right"
    ]].copy()

    mid_all = pd.concat([df_box['Midsection AR left'], df_box['Midsection AR right']])
    tip_all = pd.concat([df_box['Tip AR left'], df_box['Tip AR right']])

    box_data = [mid_all, tip_all]
    labels = ['Midsection', 'Tip']

    fig, ax = plt.subplots(figsize=(6, 5))
    # Plot boxplot first
    bp = ax.boxplot(box_data, positions=[1,2], widths=0.8, patch_artist=True,
                    boxprops=dict(facecolor='white', linewidth=2),
                    medianprops=dict(color='black', linewidth=2))

    # Overlay points with slight jitter for visibility (plotted after boxplot)

    np.random.seed(0)
    jitter = np.random.uniform(-0.08, 0.08, size=len(mid_all))
    ax.scatter(1 + jitter, mid_all, color=colours["midsection"], alpha=0.8, s=60, label='Midsection', edgecolor = "k", zorder=3)
    jitter = np.random.uniform(-0.08, 0.08, size=len(tip_all))
    ax.scatter(2 + jitter, tip_all, color=colours["tip"], alpha=0.8, s=60, label='Tip', edgecolor = "k", zorder=3)

    # Remove NaNs for paired test
    mid_vals = pd.concat([df_box['Midsection AR left'], df_box['Midsection AR right']]).dropna()
    tip_vals = pd.concat([df_box['Tip AR left'], df_box['Tip AR right']]).dropna()

    # Ensure equal length for paired test (if not, use unpaired test)
    min_len = min(len(mid_vals), len(tip_vals))
    mid_vals = mid_vals.iloc[:min_len]
    tip_vals = tip_vals.iloc[:min_len]

    # Paired t-test
    t_stat, t_p = ttest_rel(mid_vals, tip_vals)
    # Wilcoxon signed-rank test (nonparametric)
    w_stat, w_p = wilcoxon(mid_vals, tip_vals)

    print(f"Paired t-test: p = {t_p:.3g}")
    print(f"Wilcoxon signed-rank: p = {w_p:.3g}")

    # Annotate p-value on plot
    y_max = max(mid_vals.max(), tip_vals.max())
    ax.text(1.5, y_max * 1.05, f"p = {w_p:.3g} (Wilcoxon)", ha='center', va='bottom', fontsize=12)

    ax.set_xticks([1,2])
    ax.set_xticklabels(labels)
    ax.set_ylabel('Aspect Ratio')
    plt.tight_layout()
    plt.savefig("../Figures/Fig2/AR_box.png", dpi = 300)
    plt.show()

    mid_all = pd.concat([df_box['Midsection AR left'], df_box['Midsection AR right']])
    tip_all = pd.concat([df_box['Tip AR left'], df_box['Tip AR right']])

    plt.figure(figsize=(6,6))
    plt.scatter(mid_all, tip_all, color='#56B4E9', edgecolor='black', s=60)
    plt.xlabel('Midsection Aspect Ratio')
    plt.ylabel('Tip Aspect Ratio')
    plt.plot([mid_all.min(), mid_all.max()], [mid_all.min(), mid_all.max()], 'r--', alpha=0.5)  # Diagonal reference
    plt.tight_layout()
    plt.savefig("../Figures/Fig2/tip_mid_AR_corr.png", dpi = 300)
    plt.show()