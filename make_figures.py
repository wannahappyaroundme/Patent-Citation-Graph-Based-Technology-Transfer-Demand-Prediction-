"""
make_figures.py — Publication figures for the paper, generated from the FINAL
10-seed (average-rank tie-break) results. Numbers are transcribed from
ipm_results_final/run_ipm_results.md and are kept here so the figures are
reproducible without the Drive artifacts. Output: ./paper_figures/*.png
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

OUT = "paper_figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 200})

# (model, ndcg10, ndcg_std, auc, group)  — group: pop, mf/cf, text, gnn, mit
TABLE4 = [
    ("MostPop", 0.1968, 0.0000, 0.5826, "pop"),
    ("MostPop-IPC", 0.1990, 0.0000, 0.4158, "pop"),
    ("Recency", 0.1488, 0.0000, 0.6022, "pop"),
    ("SVD", 0.0366, 0.0000, 0.5197, "cf"),
    ("MLP", 0.1502, 0.0041, 0.5760, "text"),
    ("LightGCN", 0.0578, 0.0005, 0.5090, "cf"),
    ("NGCF", 0.1964, 0.0003, 0.5836, "cf"),
    ("GraphSAGE", 0.0961, 0.0523, 0.4817, "gnn"),
    ("GAT", 0.0720, 0.0110, 0.5159, "gnn"),
    ("GraphSAGE+Debias", 0.0750, 0.0393, 0.4914, "mit"),
    ("GraphSAGE+logQ", 0.1352, 0.0350, 0.5463, "mit"),
    ("GraphSAGE+DropEdge", 0.1142, 0.0462, 0.4861, "mit"),
    ("GraphSAGE+Time", 0.1060, 0.0483, 0.4912, "mit"),
    ("GraphSAGE+IPS", 0.1277, 0.0224, 0.4296, "mit"),
    ("GAT+Debias", 0.0627, 0.0233, 0.5088, "mit"),
    ("GAT+logQ", 0.1068, 0.0012, 0.5497, "mit"),
    ("GAT+DropEdge", 0.0709, 0.0142, 0.5161, "mit"),
    ("GAT+Time", 0.0740, 0.0154, 0.5155, "mit"),
    ("GAT+IPS", 0.1571, 0.0007, 0.4174, "mit"),
    ("GraphSAGE+Debias+IPS", 0.1550, 0.0053, 0.4171, "mit"),
    ("GraphSAGE+Time+IPS", 0.1285, 0.0256, 0.4351, "mit"),
    ("GAT+Debias+IPS", 0.1574, 0.0003, 0.4174, "mit"),
    ("GAT+Time+IPS", 0.1554, 0.0055, 0.4172, "mit"),
]
GROUP_COLOR = {"pop": "#888888", "cf": "#4477AA", "text": "#66CCEE", "gnn": "#EE6677", "mit": "#CCBB44"}
GROUP_LABEL = {"pop": "Popularity/Recency", "cf": "MF / CF", "text": "Text MLP", "gnn": "GNN", "mit": "Mitigation"}
MOSTPOP = 0.1968


def fig1_main():
    rows = sorted(TABLE4, key=lambda r: r[1])
    names = [r[0] for r in rows]
    ndcg = [r[1] for r in rows]
    err = [r[2] for r in rows]
    auc = [r[3] for r in rows]
    colors = [GROUP_COLOR[r[4]] for r in rows]
    y = np.arange(len(rows))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 8), sharey=True)
    ax1.barh(y, ndcg, xerr=err, color=colors, capsize=2, edgecolor="white", linewidth=0.4)
    ax1.axvline(MOSTPOP, color="black", ls="--", lw=1)
    ax1.text(MOSTPOP, len(rows) - 0.3, " MostPop 0.197", fontsize=8, va="top")
    ax1.set_yticks(y); ax1.set_yticklabels(names, fontsize=8)
    ax1.set_xlabel("NDCG@10"); ax1.set_title("(a) Ranking accuracy (NDCG@10)")

    ax2.barh(y, auc, color=colors, edgecolor="white", linewidth=0.4)
    ax2.axvline(0.5, color="black", ls="--", lw=1)
    ax2.text(0.5, len(rows) - 0.3, " chance 0.5", fontsize=8, va="top")
    ax2.set_xlabel("AUC (tie-aware)"); ax2.set_title("(b) Discrimination (AUC)")
    ax2.set_xlim(0.40, 0.62)

    handles = [Patch(color=GROUP_COLOR[g], label=GROUP_LABEL[g]) for g in GROUP_COLOR]
    ax1.legend(handles=handles, fontsize=7, loc="lower right")
    fig.suptitle("Every learned model falls below the popularity baseline; AUC near chance", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(f"{OUT}/fig1_main_performance.png"); plt.close(fig)


def fig2_tiebreak():
    models = ["SVD", "GraphSAGE+logQ", "GAT"]
    strict = [0.953, 1.000, 0.451]   # pre-fix (strict '>')
    avg = [0.037, 0.135, 0.072]      # post-fix (average-rank)
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - w / 2, strict, w, label="Strict '>' tie-break (artifact)", color="#EE6677")
    ax.bar(x + w / 2, avg, w, label="Average-rank tie-break (correct)", color="#4477AA")
    for i, v in enumerate(strict): ax.text(i - w / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(avg): ax.text(i + w / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_ylabel("NDCG@10"); ax.set_ylim(0, 1.1)
    ax.set_title("Tie-break artifact: cold-start ties inflate no-information models to ~1.0")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig2_tiebreak_artifact.png"); plt.close(fig)


def fig3_popbias():
    # (model, spearman, inversion%, ndcg)
    rows = [
        ("MostPop", 1.00, 56.4, 0.197), ("NGCF", 0.96, 56.3, 0.196), ("Recency", 0.69, 50.0, 0.149),
        ("MLP", 0.19, 18.7, 0.150), ("GraphSAGE", 0.07, 49.8, 0.096), ("SVD", 0.02, 2.4, 0.037),
        ("GAT", -0.05, 28.3, 0.072),
    ]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    # (a) Spearman rho vs NDCG scatter
    for m, rho, inv, nd in rows:
        ax1.scatter(rho, nd, s=60, color="#4477AA")
        ax1.annotate(m, (rho, nd), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax1.axhline(MOSTPOP, color="gray", ls="--", lw=1); ax1.text(-0.05, MOSTPOP, " MostPop", fontsize=7, va="bottom")
    ax1.set_xlabel("Spearman ρ (score vs train popularity)"); ax1.set_ylabel("NDCG@10")
    ax1.set_title("(a) Popularity correlation does not buy accuracy")
    # (b) inversion rate bar
    names = [r[0] for r in rows]; inv = [r[2] for r in rows]
    yb = np.arange(len(rows))
    ax2.barh(yb, inv, color="#CCBB44")
    ax2.axvline(50, color="black", ls="--", lw=1); ax2.text(50, len(rows) - 0.3, " 50%", fontsize=7, va="top")
    ax2.set_yticks(yb); ax2.set_yticklabels(names, fontsize=8); ax2.invert_yaxis()
    ax2.set_xlabel("Hard-negative inversion rate (%)")
    ax2.set_title("(b) Popular negatives outrank the positive")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig3_popularity_bias.png"); plt.close(fig)


def fig4_stratified():
    strata = ["Head\n(popular)", "Torso", "Tail/new\n(unpopular)"]
    vals = [0.0161, 0.0818, 0.3551]; errs = [0.0021, 0.0045, 0.0746]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(strata, vals, yerr=errs, capsize=5, color=["#EE6677", "#CCBB44", "#4477AA"])
    for i, v in enumerate(vals): ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("GAT NDCG@10"); ax.set_ylim(0, 0.45)
    ax.set_title("GAT does best on unpopular targets — but absolute values stay low")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig4_stratified.png"); plt.close(fig)


def fig5_coldstart():
    models = ["SVD", "GAT", "GraphSAGE", "MostPop"]
    newp = [0.000, 0.021, 0.107, 0.212]
    seenp = [0.439, 0.021, 0.129, 0.324]
    x = np.arange(len(models)); w = 0.38
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(x - w / 2, newp, w, label="New patent (91.7% of test)", color="#EE6677")
    ax.bar(x + w / 2, seenp, w, label="Seen patent (~8%)", color="#4477AA")
    for i, v in enumerate(newp): ax.text(i - w / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(seenp): ax.text(i + w / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(models); ax.set_ylabel("NDCG@10")
    ax.set_title("Patent cold-start: SVD scores 0.000 on unseen patents")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig5_coldstart.png"); plt.close(fig)


def fig6_error():
    models = ["GraphSAGE", "GAT"]
    popular = [98.5, 57.8]; rare = [0.2, 8.3]; resid = [1.3, 33.9]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(models, popular, label="Lost to popular hard-negative", color="#EE6677")
    ax.bar(models, rare, bottom=popular, label="Rare / new positive", color="#CCBB44")
    ax.bar(models, resid, bottom=np.array(popular) + np.array(rare), label="Semantic residual", color="#4477AA")
    for i, v in enumerate(popular): ax.text(i, v / 2, f"{v:.0f}%", ha="center", color="white", fontsize=9)
    ax.set_ylabel("Share of failed queries (%)"); ax.set_ylim(0, 105)
    ax.set_title("Error-source decomposition")
    ax.legend(fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.32), ncol=1)
    fig.tight_layout(); fig.savefig(f"{OUT}/fig6_error_sources.png"); plt.close(fig)


def fig7_sweeps():
    betas = [0.0, 0.5, 1.0, 2.0, 4.0]
    ips = {"GraphSAGE": [0.0961, 0.1181, 0.1277, 0.1473, 0.1565], "GAT": [0.0720, 0.1515, 0.1571, 0.1573, 0.1573]}
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    deb = {"GraphSAGE": [0.1358, 0.0845, 0.0868, 0.1073, 0.1293], "GAT": [0.0775, 0.0634, 0.0540, 0.0615, 0.0567]}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for m, c in [("GraphSAGE", "#4477AA"), ("GAT", "#EE6677")]:
        ax1.plot(betas, ips[m], "-o", color=c, label=m)
        ax2.plot(alphas, deb[m], "-s", color=c, label=m)
    for ax in (ax1, ax2):
        ax.axhline(MOSTPOP, color="gray", ls="--", lw=1); ax.set_ylabel("NDCG@10"); ax.legend(fontsize=8)
    ax1.text(0, MOSTPOP, " MostPop", fontsize=7, va="bottom")
    ax1.set_xlabel("IPS penalty β"); ax1.set_title("(a) IPS re-ranking sweep")
    ax2.set_xlabel("Debiased-NS exponent α"); ax2.set_title("(b) Popularity-debiased sampling sweep")
    fig.suptitle("Mitigation sweeps: no setting reaches the popularity baseline", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{OUT}/fig7_mitigation_sweeps.png"); plt.close(fig)


if __name__ == "__main__":
    fig1_main(); fig2_tiebreak(); fig3_popbias(); fig4_stratified()
    fig5_coldstart(); fig6_error(); fig7_sweeps()
    print("Figures written to", OUT + "/:")
    for f in sorted(os.listdir(OUT)):
        print("  ", f)
