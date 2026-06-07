import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from metrics_core import match_pulses, gate
import os


sns.set_theme(style="whitegrid", context="notebook")


fig_size = (14, 11)      
grid_fig_size = (18, 14)   
font_size = 28             
line_width = 9         
ref_line_width = 3     

plt.rcParams.update({
    "figure.figsize": fig_size,
    "axes.titlesize": font_size,
    "axes.labelsize": font_size,
    "xtick.labelsize": font_size,
    "ytick.labelsize": font_size,
    "legend.fontsize": font_size,
    "lines.linewidth": line_width,
})

heatmap_cmap = "Blues"                                 
palette = sns.color_palette("Blues", 6)                 
primary = palette[4]                                    
accent = sns.color_palette("deep")[3]                   
detection_colors = [palette[5], palette[3], palette[1]] 

types = ["cw", "lfm", "hfm"]    
count_values = [0, 1, 2, 3, 4]  

reg_keys = ["t_start", "t_stop", "f1", "f2"]
display_labels = {"t_start": "t_start", "t_stop": "t_stop", "f1": "f_start", "f2": "f_stop"}


def unit(key):

    """
    Returns the physical unit string associated with a regression key.

    ----------

    Parameters:
        key (str) - one of the reg_keys ('t_start', 't_stop', 'f1', 'f2').

    Returns:
        (str) - 's' for time keys, 'Hz' for frequency keys.
    """

    return "s" if key.startswith("t") else "Hz"


def collect(all_pred, all_truth):

    """
    Collects the data needed for all figures: the count and empty/non-empty confusions, the type confusion, 
    and the regression arrays for matched pulses.

    ----------

    Parameters:
        all_pred (list) - list of samples. Each sample is a list of predicted pulse dicts with keys type, t_start, t_stop, f1, f2.
        all_truth (list) - list of samples. Each sample is a list of ground-truth pulse dicts, same keys.

    Returns:
        count_cm (ndarray) - 2-D array of shape (5, 5), rows = true count, cols = predicted count.
        type_cm (ndarray) - 2-D array of shape (3, 3), rows = true type, cols = predicted type (matched pulses).
        reg (dict) - dict mapping each quantity ('t_start', 't_stop', 'f1', 'f2') to {'pred': array, 'true': array}.
    """

    count_cm = np.zeros((len(count_values), len(count_values)), dtype=int)
    type_cm = np.zeros((len(types), len(types)), dtype=int)
    reg = {k: {"pred": [], "true": []} for k in reg_keys}
    t_index = {t: i for i, t in enumerate(types)}

    for pred, truth in zip(all_pred, all_truth):
        tc = min(len(truth), count_values[-1])
        pc = min(len(pred), count_values[-1])
        count_cm[tc, pc] += 1

        pairs, _, _ = match_pulses(pred, truth)
        for (i, j) in pairs:
            type_cm[t_index[truth[j]["type"]], t_index[pred[i]["type"]]] += 1
            for key in reg:
                reg[key]["pred"].append(pred[i][key])
                reg[key]["true"].append(truth[j][key])

    for key in reg:
        reg[key]["pred"] = np.array(reg[key]["pred"], float)
        reg[key]["true"] = np.array(reg[key]["true"], float)

    return count_cm, type_cm, reg


def plot_confusion(cm, labels, title, path, value_label="count"):

    """
    Draws a confusion-matrix heatmap. 

    ----------

    Parameters:
        cm (ndarray) - confusion matrix, rows = true, cols = predicted.
        labels (list) - tick labels for both axes.
        title (str) - figure title.
        path (str) - output file path for the saved PNG.
        value_label (str) - default set to 'count'.

    Returns:
        None
    """

    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm, float), where=row_sums > 0)

    annot = np.empty(cm.shape, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = (f"{cm[i, j]}\n{norm[i, j]*100:.0f}%"
                           if row_sums[i] > 0 else f"{cm[i, j]}")

    fig, ax = plt.subplots(figsize=(2.5 * len(labels) + 4, 2.5 * len(labels) + 3))
    sns.heatmap(norm, ax=ax, cmap=heatmap_cmap, vmin=0, vmax=1,
                annot=annot, fmt="", annot_kws={"fontsize": font_size},
                xticklabels=labels, yticklabels=labels, square=True,
                linewidths=0.5, linecolor="white",
                cbar_kws={"label": "row-normalized", "fraction": 0.046, "pad": 0.04})
    ax.set_xlabel(f"Predicted {value_label}")
    ax.set_ylabel(f"True {value_label}")
    ax.set_title(title)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_detection(metrics, path):

    """
    Draws a bar chart of the detection outcomes (true positives, false alarms, misses) with the
    precision, recall and F1 score annotated in legend.

    ----------

    Parameters:
        metrics (dict) - output of metrics_core.evaluate; the 'detection' entry is used.
        path (str) - output file path for the saved PNG.

    Returns:
        None
    """

    d = metrics["detection"]
    bars = ["TP\n(found)", "FP\n(false alarm)", "FN\n(missed)"]
    vals = [d["TP"], d["FP"], d["FN"]]

    fig, ax = plt.subplots(figsize=fig_size)
    ax.bar(range(len(bars)), vals, color=detection_colors)
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels(bars)
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=font_size)
    ax.set_ylabel("pulse count")
    ax.set_title("Detection outcomes (match threshold= %.2f s)" % gate)

    txt = (f"precision = {d['precision']:.3f}\n"
           f"recall    = {d['recall']:.3f}\n"
           f"F1        = {d['f1']:.3f}")
    ax.text(0.97, 0.95, txt, transform=ax.transAxes, ha="right", va="top",
            family="monospace", fontsize=font_size,
            bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_error_histograms(reg, path):

    """
    Draws a 2x2 grid of signed-error histograms (predicted minus true) for the four regression
    quantities. A solid line marks zero error and a dashed line marks the mean error.

    ----------

    Parameters:
        reg (dict) - regression arrays from collect, keyed by reg_keys with 'pred' and 'true' entries.
        path (str) - output file path for the saved PNG.

    Returns:
        None
    """

    fig, axes = plt.subplots(2, 2, figsize=grid_fig_size)
    for ax, key in zip(axes.ravel(), reg_keys):
        name = display_labels[key]
        err = reg[key]["pred"] - reg[key]["true"]
        if err.size == 0:
            ax.set_title(f"{name}  (no matched pulses)"); ax.axis("off"); continue
        sns.histplot(err, bins=30, ax=ax, color=primary, edgecolor="white")
        ax.axvline(0, color="0.2", lw=ref_line_width)
        ax.axvline(err.mean(), color=accent, ls="--", lw=ref_line_width, label=f"mean error={err.mean():+.3g}")
        ax.set_title(f"{name} signed error  (n={err.size})")
        ax.set_xlabel(f"predicted - true [{unit(key)}]")
        ax.set_ylabel("count")
        ax.legend()
    fig.suptitle("Regression error distributions (matched pulses)", fontsize=font_size, y=0.99)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_pred_vs_true(reg, path):

    """
    Draws a 2x2 grid of predicted-versus-true scatter plots for the four regression quantities, each
    with the y = x reference line.

    ----------

    Parameters:
        reg (dict) - regression arrays from collect, keyed by reg_keys with 'pred' and 'true' entries.
        path (str) - output file path for the saved PNG.

    Returns:
        None
    """

    fig, axes = plt.subplots(2, 2, figsize=grid_fig_size)
    for ax, key in zip(axes.ravel(), reg_keys):
        name = display_labels[key]
        p, t = reg[key]["pred"], reg[key]["true"]
        if p.size == 0:
            ax.set_title(f"{name}  (no matched pulses)"); ax.axis("off"); continue
        ax.scatter(t, p, color=primary, s=40, alpha=0.5, edgecolors="none")
        lo = float(min(p.min(), t.min())); hi = float(max(p.max(), t.max()))
        ax.plot([lo, hi], [lo, hi], ls="--", lw=ref_line_width, color="0.2", label="y = x")
        ax.set_title(f"{name}  (n={p.size})")
        ax.set_xlabel(f"true [{unit(key)}]")
        ax.set_ylabel(f"predicted [{unit(key)}]")
        ax.set_aspect("equal", adjustable="datalim")
        ax.legend()
    fig.suptitle("Predicted vs. true (matched pulses)", fontsize=font_size, y=0.99)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_loss_curves(outdir, prefix, train_path="train_losses.npy", val_path="val_losses.npy"):

    """
    Plots and saves the training and validation loss curves, marking the best epoch (the validation
    minimum). Skipped if the saved loss arrays are not found.

    ----------

    Parameters:
        outdir (str) - directory the figure is written to.
        prefix (str) - filename prefix, matching the other evaluation figures.
        train_path (str) - default 'train_losses.npy'. Path to the saved training-loss array.
        val_path (str) - default 'val_losses.npy'. Path to the saved validation-loss array.

    Returns:
        path (str) - file path of the saved figure, or None if the arrays were not found.
    """

    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        print("  [plots] no loss arrays found -> skipping loss curves")
        return None

    train = np.load(train_path)
    val = np.load(val_path)
    best = int(np.argmin(val)) if len(val) else None

    fig, ax = plt.subplots(figsize=fig_size)
    ax.plot(range(len(train)), train, label="Training Loss", marker="o", markersize=10)
    ax.plot(range(len(val)), val, label="Validation Loss", marker="s", markersize=10)
    if best is not None:
        ax.axvline(best, color="green", ls="--", lw=ref_line_width, alpha=0.6)
        ax.scatter([best], [val[best]], color="green", zorder=5, s=200,
                   label=f"Best model (epoch {best}, val {val[best]:.4f})")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend(); ax.grid(True, alpha=0.3)

    path = os.path.join(outdir, f"{prefix}_loss_curves.png")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

    return path


def make_all_plots(all_pred, all_truth, metrics, outdir=".", prefix="eval"):

    """
    Builds and saves every evaluation figure.

    ----------

    Parameters:
        all_pred (list) - list of samples. Each sample is a list of predicted pulse dicts with keys type, t_start, t_stop, f1, f2.
        all_truth (list) - list of samples. Each sample is a list of ground-truth pulse dicts, same keys.
        metrics (dict) - output of metrics_core.evaluate.
        outdir (str) - default set to '.'. Directory the figures are written to (created if absent).
        prefix (str) - default set to 'eval'. Filename prefix for every saved figure.

    Returns:
        paths (list) - list of file paths written.
    """

    import os
    os.makedirs(outdir, exist_ok=True)
    count_cm, type_cm, reg = collect(all_pred, all_truth)

    paths = []
    p = os.path.join(outdir, f"{prefix}_count_confusion.png")
    plot_confusion(count_cm, [str(c) for c in count_values],
                    "Pulse-count confusion", p, value_label="count"); paths.append(p)

    p = os.path.join(outdir, f"{prefix}_type_confusion.png")
    if type_cm.sum() > 0:
        plot_confusion(type_cm, types, "Pulse-type confusion (matched pulses)",
                        p, value_label="type"); paths.append(p)
    else:
        print("  [plots] no matched pulses -> skipping type confusion matrix")

    p = os.path.join(outdir, f"{prefix}_detection.png")
    plot_detection(metrics, p); paths.append(p)

    p = os.path.join(outdir, f"{prefix}_error_histograms.png")
    plot_error_histograms(reg, p); paths.append(p)

    p = os.path.join(outdir, f"{prefix}_pred_vs_true.png")
    plot_pred_vs_true(reg, p); paths.append(p)

    p = plot_loss_curves(outdir, prefix)
    if p:
        paths.append(p)

    print(f"  [plots] wrote {len(paths)} figures to '{outdir}/'")
    
    return paths
