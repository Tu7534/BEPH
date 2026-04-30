import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score


def compute_clustering_metrics(true_labels, pred_labels, embeddings=None):
    metrics = {}
    metrics['ARI'] = adjusted_rand_score(true_labels, pred_labels)
    metrics['NMI'] = normalized_mutual_info_score(true_labels, pred_labels)
    if embeddings is not None and len(set(pred_labels)) > 1:
        try:
            metrics['silhouette'] = silhouette_score(embeddings, pred_labels)
        except Exception:
            metrics['silhouette'] = float('nan')
    else:
        metrics['silhouette'] = float('nan')
    return metrics
