from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Counter as CounterT

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels

from sitod.data import TurnPrediction, Intent


def compute_optimal_alignment(predicted_labels: List[str], reference_labels: List[str]) -> Dict[str, str]:
    """
    Find an optimal assignment of predicted labels (e.g. cluster labels) to corresponding reference
    labels (ground truth labels) by maximizing overlap between each predicted and ground truth label.
    :param predicted_labels: predicted labels, e.g. cluster IDs
    :param reference_labels: corresponding reference labels, such as ground truth cluster labels
    :return: mapping of predicted labels to reference labels
    """
    # (1) assign unique labels to indices
    unique_predicted_labels, cluster_label_indices = np.unique(predicted_labels, return_inverse=True)
    unique_ref_labels, reference_label_indices = np.unique(reference_labels, return_inverse=True)

    # (2) build matrix counting overlap between predicted and reference labels
    cost_matrix = np.zeros((len(unique_predicted_labels), len(unique_ref_labels)))
    for predicted, reference in zip(cluster_label_indices, reference_label_indices):
        cost_matrix[predicted][reference] += 1

    # (3) compute optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

    # (4) return optimal 1:1 mapping of cluster labels to reference labels
    return {
        unique_predicted_labels[row]: unique_ref_labels[col] for row, col in zip(row_ind.tolist(), col_ind.tolist())
    }


def align_labels(predicted_labels: List[str], alignment: Dict[str, str], default_label=None) -> List[str]:
    """
    Apply alignment to predicted labels.
    :param predicted_labels: predicted labels, e.g. cluster IDs
    :param alignment: alignment of predicted labels to reference labels
    :param default_label: default label to be used if predicted label is not present in alignment
    :return: aligned predicted labels
    """
    return [alignment.get(label, default_label) for label in predicted_labels]


def count_cluster_label_overlap(
    first_clustering: List[str], second_clustering: List[str]
) -> Dict[str, CounterT[str]]:
    """
    Return the label overlap counts between two clusterings.
    """
    overlap_counts = defaultdict(Counter)
    for first_label, second_label in zip(first_clustering, second_clustering):
        overlap_counts[first_label][second_label] += 1
    return overlap_counts


class ClusteringMetric(object):

    def metric_name(self) -> str:
        """
        Returns the name of the clustering metric for reporting.
        """
        raise NotImplementedError

    def compute_metric(self, cluster_labels: List[str], reference_labels: List[str]) -> float:
        """
        Compute extrinsic cluster metric given cluster labels and corresponding reference (ground truth) labels.
        :param cluster_labels: predicted cluster labels
        :param reference_labels: ground truth labels
        :return: cluster metric result
        """
        raise NotImplementedError


class NMI(ClusteringMetric):
    """
    Normalized mutual information between predicted and reference clusterings.
    """

    def metric_name(self) -> str:
        return 'NMI'

    def compute_metric(self, cluster_labels: List[str], reference_labels: List[str]) -> float:
        return 100 * normalized_mutual_info_score(reference_labels, cluster_labels)


class ARI(ClusteringMetric):
    """
    Adjusted Rand index between predicted and reference clusterings.
    """

    def metric_name(self) -> str:
        return 'ARI'

    def compute_metric(self, cluster_labels: List[str], reference_labels: List[str]) -> float:
        return 100 * adjusted_rand_score(reference_labels, cluster_labels)


class ClusteringAccuracy(ClusteringMetric):
    """
    Clustering accuracy, in which an optimal 1:1 alignment is found between predicted cluster labels
    and reference labels.
    """

    def metric_name(self) -> str:
        return 'ACC'

    def compute_metric(self, cluster_labels: List[str], reference_labels: List[str]) -> float:
        alignment = compute_optimal_alignment(cluster_labels, reference_labels)
        aligned_labels = align_labels(cluster_labels, alignment)
        total_correct = sum(1 for aligned, reference in zip(aligned_labels,
                                                            reference_labels) if aligned == reference)
        accuracy = total_correct / len(reference_labels) if reference_labels else 0
        return 100 * accuracy


class ClusteringPrecision(ClusteringMetric):
    """
    Clustering precision, in which a many-to-one alignment is computed from *cluster* labels to *reference* labels
    such that the number of correct aligned labels is maximized.
    """

    def metric_name(self) -> str:
        return 'Precision'

    def compute_metric(self, cluster_labels: List[str], reference_labels: List[str]) -> float:
        total = 0
        for cluster_label, ref_counts in count_cluster_label_overlap(cluster_labels, reference_labels).items():
            max_ref, max_ref_count = ref_counts.most_common()[0]
            total += max_ref_count
        precision = total / len(reference_labels) if reference_labels else 0
        return 100 * precision


class ClusteringRecall(ClusteringMetric):
    """
    Clustering recall, in which a many-to-one alignment is computed from *reference* labels to *cluster* labels
    such that the number of correct aligned labels is maximized.
    """

    def metric_name(self) -> str:
        return 'Recall'

    def compute_metric(self, cluster_labels: List[str], reference_labels: List[str]) -> float:
        return ClusteringPrecision().compute_metric(reference_labels, cluster_labels)


class ClusteringF1(ClusteringMetric):
    """
    Clustering F1, a harmonic mean between `ClusteringPrecision` and `ClusteringRecall` metrics.
    """

    def metric_name(self) -> str:
        return 'F1'

    def compute_metric(self, cluster_labels: List[str], reference_labels: List[str]) -> float:
        # this is considered precision because we maximize counts over predicted clusters
        precision = ClusteringPrecision().compute_metric(cluster_labels, reference_labels)
        # this is considered recall because we maximize counts over reference clusters
        recall = ClusteringRecall().compute_metric(cluster_labels, reference_labels)
        denom = precision + recall
        return (2 * precision * recall / denom) if denom > 0 else 0


class NumberOfClusters(ClusteringMetric):
    def metric_name(self) -> str:
        return 'K'

    def compute_metric(self, cluster_labels: List[str], reference_labels: List[str]) -> float:
        return len(set(cluster_labels))


class NumberOfReferenceLabels(ClusteringMetric):
    def metric_name(self) -> str:
        return 'Reference K'

    def compute_metric(self, cluster_labels: List[str], reference_labels: List[str]) -> float:
        return len(set(reference_labels))


class NumberOfInstances(ClusteringMetric):
    def metric_name(self) -> str:
        return '# Instances'

    def compute_metric(self, cluster_labels: List[str], reference_labels: List[str]) -> float:
        return len(reference_labels)


class NumberOfCoveredIntents(ClusteringMetric):
    """
    The number of reference clusters/intents covered by one or more predicted clusters
    after performing a many-to-one alignment from predicted clusters to reference clusters.
    """

    def metric_name(self) -> str:
        return '# Covered Intents'

    def compute_metric(self, cluster_labels: List[str], reference_labels: List[str]) -> float:
        cluster_alignment = compute_many_to_one_alignment(cluster_labels, reference_labels)
        covered_intents = set(cluster_alignment.values())
        return len(covered_intents)


class ExampleCoverage(ClusteringMetric):
    """
    Example coverage, defined as percent of examples whose reference intent has a corresponding
    predicted cluster after performing a many-to-one alignment from predicted clusters to reference clusters.
    """

    def metric_name(self) -> str:
        return 'Example Coverage'

    def compute_metric(self, cluster_labels: List[str], reference_labels: List[str]) -> float:
        cluster_alignment = compute_many_to_one_alignment(cluster_labels, reference_labels)
        covered_intents = set(cluster_alignment.values())
        covered_count = sum([1 for label in reference_labels if label in covered_intents])
        coverage = (100 * covered_count / len(reference_labels)) if reference_labels else 0
        return coverage


def compute_many_to_one_alignment(cluster_labels: List[str], reference_labels: List[str]) -> Dict[str, str]:
    """
    Compute a many-to-one alignment from `cluster_labels` to `reference_labels` such that the total number
    of aligned cluster labels that match reference labels is maximized. Ties are broken alphanumerically.
    :param cluster_labels: cluster labels to align
    :param reference_labels: labels to align onto
    :return: many-to-one alignment
    """
    ref_counts_by_cluster_label: Dict[str, CounterT[str]] = count_cluster_label_overlap(
        cluster_labels, reference_labels)
    return {
        cluster: max(ref_counts.items(), key=lambda item: (item[1], item[0]))[0]
        for cluster, ref_counts in ref_counts_by_cluster_label.items()
    }


def compute_mean_and_stdev(
    metrics: Dict[str, List[float]],
    skip_keys: List[str] = None
) -> Tuple[Dict[str, float], Dict[str, float]]:
    mean_dict = {}
    stdev_dict = {}
    for key, vals in metrics.items():
        if skip_keys and key in skip_keys:
            continue
        mean_dict[key] = float(np.mean(vals))
        stdev_dict[key] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0
    return mean_dict, stdev_dict


def format_mean_and_stdev(
    mean_dict: Dict[str, float],
    stdev_dict: Dict[str, float],
    plus_minus_symbol: str = '+/-'
) -> Dict[str, str]:
    result = {}
    for key, mean in mean_dict.items():
        if key in stdev_dict:
            stdev = stdev_dict[key]
            result[key] = f'{mean:.1f} {plus_minus_symbol} {stdev:.1f}'
    return result


def classification_metrics(predicted_labels: List[str], reference_labels: List[str]) -> Dict[str, Any]:
    predicted_labels = [str(label) for label in predicted_labels]
    reference_labels = [str(label) for label in reference_labels]
    labels = unique_labels(reference_labels, predicted_labels).tolist()
    precision, recall, f1, support = precision_recall_fscore_support(reference_labels, predicted_labels,
                                                                     zero_division=0)
    micro_precision, micro_recall, micro_f1, _ = precision_recall_fscore_support(
        reference_labels, predicted_labels, zero_division=0, average='micro'
    )
    precision = precision.tolist() + [micro_precision.tolist()]
    recall = recall.tolist() + [micro_recall.tolist()]
    f1 = f1.tolist() + [micro_f1.tolist()]
    support = support.tolist() + [sum(support.tolist())]
    labels += ['Micro Averaged']
    metrics = {}
    for label, label_precision, label_recall, label_f1, label_support in zip(labels, precision, recall, f1, support):
        metrics.update({
            f'{label} P': label_precision,
            f'{label} R': label_recall,
            f'{label} F1': label_f1,
            f'{label} Support': label_support,
        })
    return metrics


def filter_labels(
    cluster_labels: List[str], reference_labels: List[str], ignored_labels: List[str]
) -> Tuple[List[str], List[str]]:
    """
    Filter out any ignored instances and corresponding cluster labels given a list of ignored reference labels.
    """
    filtered_cluster_labels = []
    filtered_reference_labels = []
    for cluster_label, reference_label in zip(cluster_labels, reference_labels):
        if reference_label in ignored_labels:
            continue
        filtered_cluster_labels.append(cluster_label)
        filtered_reference_labels.append(reference_label)
    return filtered_cluster_labels, filtered_reference_labels


def schema_metrics(schema: List[Intent]) -> Dict[str, Any]:
    labels = []
    for intent in schema:
        labels += len(intent.utterances) * [intent.intent_id]
    total_utterances = sum(len(intent.utterances) for intent in schema)
    counts = {
        '# Intents': len(schema),
        '# Utterances': total_utterances,
        '# Utterances per Intent': total_utterances / len(schema) if schema else 0,
    }
    return counts


def clustering_info(
    cluster_labels: List[str],
    reference_labels: List[str]
) -> Dict[str, Any]:
    cluster_alignment = compute_many_to_one_alignment(cluster_labels, reference_labels)
    ref_alignment = compute_many_to_one_alignment(reference_labels, cluster_labels)
    one_to_one_alignment = compute_optimal_alignment(cluster_labels, reference_labels)

    metrics = {
        '1:1 Alignment': one_to_one_alignment,
        'Cluster:Ref Many:1 Alignment': cluster_alignment,
        'Ref:Cluster Many:1 Alignment': ref_alignment,
    }
    return metrics


def compute_metrics_from_turn_predictions(
    turn_predictions: List[TurnPrediction],
    metrics: List[ClusteringMetric] = None,
    ignore_labels: List[str] = None,
) -> Dict[str, Any]:
    if not metrics:
        metrics = [
            NMI(),
            ARI(),
            ClusteringAccuracy(),
            ClusteringPrecision(),
            ClusteringRecall(),
            ClusteringF1(),
            ExampleCoverage(),
            NumberOfReferenceLabels(),
            NumberOfClusters()
        ]

    cluster_labels = []
    reference_labels = []
    for prediction in turn_predictions:
        cluster_labels.append(prediction.predicted_label)
        reference_labels.append(prediction.reference_label)

    cluster_labels, reference_labels = filter_labels(cluster_labels, reference_labels, ignore_labels)
    metrics = {
        metric.metric_name(): metric.compute_metric(cluster_labels, reference_labels) for metric in metrics
    }

    # classification metrics for different alignments
    alignment = compute_optimal_alignment(cluster_labels, reference_labels)
    metrics['Classification 1:1'] = classification_metrics(align_labels(cluster_labels, alignment,
                                                                        default_label='N/A'), reference_labels)
    alignment = compute_many_to_one_alignment(cluster_labels, reference_labels)
    metrics['Classification Many:1'] = classification_metrics(align_labels(cluster_labels, alignment,
                                                                           default_label='N/A'), reference_labels)

    metrics.update(clustering_info(cluster_labels, reference_labels))
    return metrics
