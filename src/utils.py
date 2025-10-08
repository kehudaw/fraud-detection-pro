import numpy as np

def best_threshold_by_cost(y_true, y_prob, cost_fp=5.0, cost_fn=500.0, steps=1001):
    ts = np.linspace(0, 1, steps)
    best_t, best_savings = 0.5, -1e18
    for t in ts:
        y_hat = (y_prob >= t).astype(int)
        fp = ((y_hat == 1) & (y_true == 0)).sum()
        fn = ((y_hat == 0) & (y_true == 1)).sum()
        savings = -(fp * cost_fp + fn * cost_fn)
        if savings > best_savings:
            best_savings, best_t = savings, t
    return float(best_t), float(best_savings)
