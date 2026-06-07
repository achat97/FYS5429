import numpy as np



gate = 0.5  #max start-time gap to still call two pulses the same pulse


def match_pulses(pred, truth):

    """
    Matches predicted pulses to ground-truth pulses one-to-one for a single sample, pairing
    the closest start times first. Pairs more than treshold seconds apart are rejected.

    ----------

    Parameters:
        pred (list) - the pulses predicted for one spectrogram, e.g. [{"type": "lfm", "t_start": 0.5, "t_stop": 1.2, "f1": 3000,...].
                      One dict per pulse.
        truth (list) - the true pulses for that spectrogram, in the same form.

    Returns:
        pairs (list) - list of (pred_index, truth_index) tuples for matched pulses.
        false_alarms (list) - predicted indices left unmatched.
        misses (list) - ground-truth indices left unmatched.
    """

    n_p, n_t = len(pred), len(truth)
    if n_p == 0 or n_t == 0:
        return [], list(range(n_p)), list(range(n_t))

    C = np.array([[abs(p["t_start"] - t["t_start"]) for t in truth] for p in pred])

    pairs, used_p, used_t = [], set(), set()
    for c, i, j in sorted((C[i, j], i, j) for i in range(n_p) for j in range(n_t)):
        if i in used_p or j in used_t:
            continue
        if c > gate:
            break
        pairs.append((i, j)); used_p.add(i); used_t.add(j)

    return (pairs,
            [i for i in range(n_p) if i not in used_p],
            [j for j in range(n_t) if j not in used_t])


def evaluate(all_pred, all_truth):

    """
    Computes the evaluation metrics over the whole test set: pulse count, detection,
    classification, and regression error. Counts are compared directly. The other metrics use the
    pulses paired by match_pulses.

    ----------

    Parameters:
        all_pred (list of list of dict) - the predicted pulses for every spectrogram: one inner
                                          list per spectrogram, one dict per pulse.
        all_truth (list of list of dict) - the true pulses, in the same form.

    Returns:
        (dict) - nested metrics under the keys 'n_samples', 'count', 'detection', 'regression',
                and 'classification_on_matched'.
    """

    n = len(all_truth)
    exact = 0
    count_err = []          
    TP = FP = FN = 0
    err = {"t_start": [], "t_stop": [], "f1": [], "f2": []}  
    cls_correct = cls_total = 0

    for pred, truth in zip(all_pred, all_truth):
        if len(pred) == len(truth):
            exact += 1
        count_err.append(len(pred) - len(truth))

        pairs, fa, miss = match_pulses(pred, truth)
        TP += len(pairs); FP += len(fa); FN += len(miss)

        for (i, j) in pairs:
            for key in err:
                err[key].append(pred[i][key] - truth[j][key])
            cls_total += 1
            cls_correct += int(pred[i]["type"] == truth[j]["type"])

    def stats(d):

        """Summarizes a list of signed errors; returns None when the list is empty."""

        a = np.array(d, float)
        if a.size == 0:
            return None
        return {
            "n": int(a.size),
            "bias": float(a.mean()),
            "mae": float(np.abs(a).mean()),
            "rmse": float(np.sqrt((a**2).mean())),
            "std": float(a.std()),
            "p50": float(np.percentile(np.abs(a), 50)),
            "p90": float(np.percentile(np.abs(a), 90)),
        }

    prec = TP / (TP + FP) if (TP + FP) else float("nan")
    rec = TP / (TP + FN) if (TP + FN) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else float("nan")
    ce = np.array(count_err)

    return {
        "n_samples": n,
        "count": {
            "exact_match_rate": exact / n if n else float("nan"),
            "count_bias": float(ce.mean()),
            "count_mae": float(np.abs(ce).mean()),
            "within_1": float((np.abs(ce) <= 1).mean()),
        },
        "detection": {"TP": TP, "FP": FP, "FN": FN,
                      "precision": prec, "recall": rec, "f1": f1},
        "regression": {k: stats(v) for k, v in err.items()},
        "classification_on_matched": {
            "accuracy": cls_correct / cls_total if cls_total else float("nan"),
            "n": cls_total,
        },
    }
