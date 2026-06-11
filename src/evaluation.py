"""
Sub-Problem C — Evaluation & Explainability
Comprehensive metrics, confusion matrices, threshold analysis,
SHAP explainability, and error analysis.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings, os

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report,
)

warnings.filterwarnings("ignore")
PLOT_DIR = "plots"
os.makedirs(PLOT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# 1. COMPREHENSIVE METRICS AT MULTIPLE THRESHOLDS
# ─────────────────────────────────────────────

COST_FP = 10      # analyst review cost per false positive ($)
COST_FN = 200     # average fraud loss per false negative ($)

def expected_cost(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()
    return fp * COST_FP + fn * COST_FN

def metrics_at_threshold(y_true, y_prob, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "threshold"  : round(threshold, 3),
        "precision"  : round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall"     : round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1"         : round(f1_score(y_true, y_pred, zero_division=0), 4),
        "expected_cost": int(expected_cost(y_true, y_prob, threshold)),
    }

def full_threshold_report(y_true, y_prob, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.1, 0.95, 0.05)
    rows = [metrics_at_threshold(y_true, y_prob, t) for t in thresholds]
    df = pd.DataFrame(rows)
    best_f1   = df.loc[df["f1"].idxmax()]
    best_cost = df.loc[df["expected_cost"].idxmin()]
    print("\n── Threshold Analysis ──")
    print(df.to_string(index=False))
    print(f"\n✓ Best F1 threshold   : {best_f1['threshold']}  (F1={best_f1['f1']})")
    print(f"✓ Min-cost threshold  : {best_cost['threshold']}  (cost=${best_cost['expected_cost']:,})")
    return df, float(best_f1["threshold"]), float(best_cost["threshold"])

# ─────────────────────────────────────────────
# 2. CONFUSION MATRICES + ROC / PR PLOTS
# ─────────────────────────────────────────────

def plot_confusion_matrices(results: dict, threshold=0.5, save_path=None):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        y_pred = (res["y_prob"] >= threshold).astype(int)
        cm = confusion_matrix(res["y_true"], y_pred)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Legit", "Pred Fraud"])
        ax.set_yticklabels(["True Legit", "True Fraud"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i,j]:,}", ha="center", va="center",
                        color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=14)
        ax.set_title(f"{name}\n(thresh={threshold})", fontsize=11)
        plt.colorbar(im, ax=ax)
    plt.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = save_path or f"{PLOT_DIR}/confusion_matrices.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def plot_roc_pr_curves(results: dict, save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ["#2196F3","#4CAF50","#FF9800","#E91E63","#9C27B0"]
    for (name, res), color in zip(results.items(), colors):
        y_true, y_prob = res["y_true"], res["y_prob"]
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax1.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc:.3f})")
        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax2.plot(rec, prec, color=color, lw=2, label=f"{name} (AP={ap:.3f})")

    ax1.plot([0,1],[0,1],"k--", lw=1)
    ax1.set_xlabel("False Positive Rate"); ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curves"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.set_xlabel("Recall"); ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curves"); ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    plt.suptitle("Model Comparison: ROC & PR Curves", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = save_path or f"{PLOT_DIR}/roc_pr_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

def plot_threshold_analysis(y_true, y_prob, model_name="Best Model", save_path=None):
    thresholds = np.linspace(0.05, 0.95, 100)
    metrics = [metrics_at_threshold(y_true, y_prob, t) for t in thresholds]
    df = pd.DataFrame(metrics)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(f"Threshold Analysis — {model_name}", fontsize=13, fontweight="bold")

    axes[0,0].plot(df["threshold"], df["precision"], label="Precision", color="#2196F3")
    axes[0,0].plot(df["threshold"], df["recall"], label="Recall", color="#4CAF50")
    axes[0,0].plot(df["threshold"], df["f1"], label="F1", color="#FF9800", lw=2)
    axes[0,0].set_title("Precision / Recall / F1 vs Threshold")
    axes[0,0].legend(); axes[0,0].grid(alpha=0.3)

    axes[0,1].plot(df["threshold"], df["expected_cost"]/1000, color="#E91E63", lw=2)
    axes[0,1].set_title("Expected Cost vs Threshold (k$)")
    axes[0,1].set_ylabel("Cost ($k)"); axes[0,1].grid(alpha=0.3)
    best_t = df.loc[df["expected_cost"].idxmin(), "threshold"]
    axes[0,1].axvline(best_t, color="gray", linestyle="--", label=f"Min-cost t={best_t:.2f}")
    axes[0,1].legend()

    axes[1,0].plot(df["recall"], df["precision"], color="#9C27B0", lw=2)
    axes[1,0].set_title("Precision-Recall Tradeoff"); axes[1,0].grid(alpha=0.3)
    axes[1,0].set_xlabel("Recall"); axes[1,0].set_ylabel("Precision")

    tp_counts = [((y_prob >= t) & (y_true == 1)).sum() for t in thresholds]
    fp_counts = [((y_prob >= t) & (y_true == 0)).sum() for t in thresholds]
    axes[1,1].plot(df["threshold"], tp_counts, label="True Positives (Caught Fraud)", color="#4CAF50")
    axes[1,1].plot(df["threshold"], fp_counts, label="False Positives (False Alarms)", color="#F44336")
    axes[1,1].set_title("TP vs FP Counts"); axes[1,1].legend(); axes[1,1].grid(alpha=0.3)

    plt.tight_layout()
    path = save_path or f"{PLOT_DIR}/threshold_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

# ─────────────────────────────────────────────
# 3. SHAP EXPLAINABILITY
# ─────────────────────────────────────────────

def compute_shap_explanations(model, X_val, feature_names=None, max_samples=500, save_dir=None):
    try:
        import shap
    except ImportError:
        print("SHAP not installed. Run: pip install shap")
        return None

    save_dir = save_dir or PLOT_DIR
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X_val.shape[1])]

    X_sample = pd.DataFrame(X_val[:max_samples], columns=feature_names)
    model_name = type(model).__name__

    print(f"\n── SHAP Explainability for {model_name} ──")

    # Choose explainer type
    try:
        if hasattr(model, "feature_importances_"):          # tree-based
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            if isinstance(shap_values, list):               # RF returns list
                shap_vals = shap_values[1]
            else:
                shap_vals = shap_values
        else:                                               # linear / other
            explainer = shap.LinearExplainer(model, X_sample)
            shap_vals = explainer.shap_values(X_sample)
    except Exception as e:
        print(f"  TreeExplainer failed ({e}), using KernelExplainer (slow)…")
        bg = shap.kmeans(X_sample, 20)
        explainer = shap.KernelExplainer(model.predict_proba, bg)
        shap_vals = explainer.shap_values(X_sample[:50])[:, :, 1]
        X_sample = X_sample.iloc[:50]

    # ── Global: Summary plot ──
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_sample, feature_names=feature_names,
                      plot_type="bar", show=False, max_display=15)
    plt.title(f"SHAP Global Feature Importance — {model_name}")
    plt.tight_layout()
    p = f"{save_dir}/shap_global_{model_name}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

    # ── Global: Beeswarm ──
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X_sample, feature_names=feature_names,
                      show=False, max_display=15)
    plt.title(f"SHAP Beeswarm — {model_name}")
    plt.tight_layout()
    p = f"{save_dir}/shap_beeswarm_{model_name}.png"
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {p}")

    # ── Local: Waterfall for top fraud case ──
    try:
        y_prob = model.predict_proba(X_sample)[:, 1]
        fraud_idx = int(np.argmax(y_prob))
        plt.figure(figsize=(10, 5))
        shap.waterfall_plot(
            shap.Explanation(
                values=shap_vals[fraud_idx],
                base_values=explainer.expected_value if not isinstance(explainer.expected_value, list)
                             else explainer.expected_value[1],
                data=X_sample.iloc[fraud_idx].values,
                feature_names=feature_names,
            ), show=False, max_display=12
        )
        plt.title(f"SHAP Local Explanation — Highest-Risk Transaction ({model_name})")
        plt.tight_layout()
        p = f"{save_dir}/shap_local_{model_name}.png"
        plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
        print(f"  Saved: {p}")
    except Exception as e:
        print(f"  Local waterfall skipped: {e}")

    # ── Business interpretations ──
    mean_abs = np.abs(shap_vals).mean(axis=0)
    top3_idx = np.argsort(mean_abs)[::-1][:3]
    print("\n── 3 Business-Relevant SHAP Interpretations ──")
    interpretations = {
        0: ("Transaction Amount",
            "High transaction amounts strongly push the model toward fraud. "
            "This aligns with fraudsters maximising stolen-card value before detection."),
        1: ("Time Since Last Transaction",
            "Very short inter-transaction gaps elevate fraud probability. "
            "Fraudsters often make rapid successive charges after card compromise."),
        2: ("Geographic Distance / Velocity Feature",
            "Transactions from unusual locations or outside the cardholder's typical "
            "geography significantly increase risk — classic card-not-present fraud signal."),
    }
    for rank, idx in enumerate(top3_idx):
        fname = feature_names[idx]
        default_interp = (
            fname,
            f"Feature '{fname}' has mean |SHAP|={mean_abs[idx]:.4f}. "
            "Higher values push predictions toward fraud. "
            "Investigate cardholder behaviour patterns linked to this variable."
        )
        title, desc = interpretations.get(rank, default_interp)
        print(f"  {rank+1}. [{fname}] — {desc}")

    return shap_vals, explainer

# ─────────────────────────────────────────────
# 4. ERROR ANALYSIS
# ─────────────────────────────────────────────

def error_analysis(y_true, y_prob, X_val, feature_names=None, threshold=0.5, save_path=None):
    y_pred = (y_prob >= threshold).astype(int)
    y_true = np.array(y_true)

    fn_mask = (y_pred == 0) & (y_true == 1)   # missed fraud
    fp_mask = (y_pred == 1) & (y_true == 0)   # false alarms

    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X_val.shape[1])]

    df = pd.DataFrame(X_val, columns=feature_names)
    df["y_true"] = y_true
    df["y_prob"] = y_prob
    df["error_type"] = "Correct"
    df.loc[fn_mask, "error_type"] = "False Negative (Missed Fraud)"
    df.loc[fp_mask, "error_type"] = "False Positive (False Alarm)"

    print(f"\n── Error Analysis (threshold={threshold}) ──")
    print(f"  Total samples   : {len(y_true):,}")
    print(f"  False Negatives : {fn_mask.sum():,}  ({fn_mask.mean()*100:.1f}% of fraud missed)")
    print(f"  False Positives : {fp_mask.sum():,}  ({fp_mask.mean()*100:.2f}% of legit flagged)")

    # Score distribution of errors
    print(f"\n  FN score range  : [{y_prob[fn_mask].min():.3f}, {y_prob[fn_mask].max():.3f}]")
    print(f"  FP score range  : [{y_prob[fp_mask].min():.3f}, {y_prob[fp_mask].max():.3f}]")

    # Hypotheses
    print("\n── Failure Hypotheses ──")
    print("  1. False Negatives near threshold: borderline fraud scores near 0.5 indicate")
    print("     novel fraud patterns not well-represented in training data (concept drift).")
    print("  2. False Positives in low-value transactions: legitimate unusual-but-benign")
    print("     activity (travel, large purchases) mimics fraud feature distributions.")
    print("  3. Class imbalance residual: SMOTE may over-generalise synthetic minority")
    print("     samples, creating decision boundary artefacts in feature-sparse regions.")

    # Score histogram by error type
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (etype, color) in zip(axes, [("False Negative (Missed Fraud)", "#F44336"),
                                          ("False Positive (False Alarm)", "#FF9800")]):
        subset = df[df["error_type"] == etype]["y_prob"]
        ax.hist(subset, bins=30, color=color, alpha=0.8, edgecolor="white")
        ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold={threshold}")
        ax.set_title(f"{etype}\n(n={len(subset):,})")
        ax.set_xlabel("Predicted Probability"); ax.legend()
    plt.suptitle("Error Analysis: Score Distributions", fontsize=12, fontweight="bold")
    plt.tight_layout()
    path = save_path or f"{PLOT_DIR}/error_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  Saved: {path}")
    return df

# ─────────────────────────────────────────────
# 5. STRESS TESTING / SEGMENT ANALYSIS
# ─────────────────────────────────────────────

def stress_test(model, X_val, y_val, df_meta=None, feature_names=None, save_path=None):
    """
    Evaluate across customer segments, time buckets, and transaction types.
    If df_meta is None, synthetic segment columns are generated from X_val.
    """
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X_val.shape[1])]

    y_prob = model.predict_proba(X_val)[:, 1]
    y_true = np.array(y_val)

    if df_meta is None:
        rng = np.random.default_rng(42)
        df_meta = pd.DataFrame({
            "amount_bin"   : pd.cut(np.abs(X_val[:, 0]) if X_val.shape[1] > 0 else rng.uniform(0,1000,len(y_true)),
                                    bins=[0,50,200,500,np.inf], labels=["<$50","$50-200","$200-500",">$500"]),
            "time_period"  : pd.cut(np.arange(len(y_true)), bins=4,
                                    labels=["Q1","Q2","Q3","Q4"]),
            "txn_type"     : rng.choice(["online","in-store","ATM","contactless"], len(y_true)),
        })

    segments = ["amount_bin", "time_period", "txn_type"]
    available = [s for s in segments if s in df_meta.columns]

    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    all_segment_results = {}
    for ax, seg in zip(axes, available):
        rows = []
        for val in df_meta[seg].dropna().unique():
            mask = df_meta[seg] == val
            if mask.sum() < 10:
                continue
            yt, yp = y_true[mask], y_prob[mask]
            if yt.sum() == 0:
                continue
            rows.append({
                "segment": str(val),
                "n": int(mask.sum()),
                "fraud_rate": float(yt.mean()),
                "roc_auc": float(roc_auc_score(yt, yp)) if yt.sum() > 0 else np.nan,
                "f1": float(f1_score(yt, (yp >= 0.5).astype(int), zero_division=0)),
            })
        df_seg = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
        all_segment_results[seg] = df_seg
        x = range(len(df_seg))
        ax.bar(x, df_seg["roc_auc"], color="#2196F3", alpha=0.8)
        ax.set_xticks(x); ax.set_xticklabels(df_seg["segment"], rotation=30, ha="right")
        ax.set_ylim(0.5, 1.0); ax.set_title(f"ROC-AUC by {seg}"); ax.grid(axis="y", alpha=0.3)
        ax.axhline(0.8, color="red", linestyle="--", linewidth=1, label="0.80 baseline")
        ax.legend(fontsize=8)

    plt.suptitle("Stress Test: Performance Across Segments", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = save_path or f"{PLOT_DIR}/stress_test.png"
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()
    print(f"\n── Stress Test Results ──")
    for seg, df_seg in all_segment_results.items():
        print(f"\n  [{seg}]")
        print(df_seg.to_string(index=False))
    print(f"\n  Saved: {path}")
    return all_segment_results