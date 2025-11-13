import numpy as np
import pandas as pd

def extract_gaussians(component, parent_weight=1.0):
    """
    Recursively flatten nested mixture components into base (mean, std, weight) tuples.
    Skip non-Gaussian (non-'norm') densities safely.
    """
    gaussians = []
    density = component["density"]
    weight = component.get("weight", 1.0) * parent_weight

    if "type" not in density:
        density["type"] = "builtin"

    # Case 1: Nested mixture — recurse
    if density["type"] == "mixture":
        for subcomp in density["components"]:
            gaussians.extend(extract_gaussians(subcomp, weight))

    # Case 2: Base Gaussian
    elif density["type"] == "builtin":
        if density.get("name") == "norm":
            params = density["params"]
            gaussians.append((params["loc"], params["scale"], weight))
        else:
            # skip unsupported distributions
            # (t-distribution or others)
            pass

    return gaussians


def densities_to_scales(predictions):

    results = []

    for pred_dict in predictions:

        # Flatten recursively
        if "components" in pred_dict:
            components = pred_dict["components"]
        elif isinstance(pred_dict, dict):
            components = [pred_dict]
        elif isinstance(pred_dict, list):
            components = pred_dict
        gaussians = []
        for comp in components:
            gaussians.extend(extract_gaussians(comp))

        if not gaussians:
            # Skip if no Gaussian component found
            continue

        # Normalize weights
        weights = np.array([w for _, _, w in gaussians])
        weights /= np.sum(weights)

        means = np.array([m for m, _, _ in gaussians])
        stds = np.array([s for _, s, _ in gaussians])

        # Weighted mean
        mean = np.average(means, weights=weights)

        # Monte Carlo quantile estimation
        n_samples = 10000
        samples = np.hstack([
            np.random.normal(m, s, int(n_samples * w))
            for m, s, w in zip(means, stds, weights)
        ])

        q05, q95 = np.percentile(samples, [5, 95])

        results.append({
            "mean": mean,
            "q05": q05,
            "q95": q95
        })

    return pd.DataFrame(results)