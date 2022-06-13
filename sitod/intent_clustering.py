"""
Intent clustering interfaces and baseline code.
"""
import hashlib
import json
import logging
import pickle
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Dict, List, Set, Any, Optional

import hyperopt.hp as hp
import numpy as np
# noinspection PyPackageRequirements
from allennlp.common import Registrable
from hyperopt import STATUS_OK, Trials, fmin, tpe, STATUS_FAIL
from hyperopt.early_stop import no_progress_loss
from hyperopt.pyll import scope
from numpy import ndarray, argmax
from sentence_transformers import SentenceTransformer
from sklearn import metrics
from sklearn.cluster import KMeans, DBSCAN, OPTICS

from sitod.data import DialogueDataset

logger = logging.getLogger(__name__)


@dataclass
class IntentClusteringContext:
    """
    Dialogue clustering context consisting of a list of dialogues and set of target turn IDs to be labeled
    with clusters.
    """

    dataset: DialogueDataset
    intent_turn_ids: Set[str]
    # output intermediate clustering results/metadata here
    output_dir: Path = None


class IntentClusteringModel(Registrable):

    def cluster_intents(self, context: IntentClusteringContext) -> Dict[str, str]:
        """
        Assign cluster IDs to intent turns within a collection of dialogues.

        :param context: dialogue clustering context

        :return: assignment of turn IDs to cluster labels
        """
        raise NotImplementedError


@dataclass
class ClusterData:
    """
    Wrapper class for cluster labels.
    """
    clusters: List[int]


@dataclass
class ClusteringContext:
    """
    Wrapper for ndarray containing clustering inputs.
    """
    features: ndarray
    # output intermediate clustering results/metadata here
    output_dir: Path = None
    # dynamically inject parameters to clustering algorithm here
    parameters: Dict[str, Any] = field(default_factory=dict)


class ClusteringAlgorithm(Registrable):

    def cluster(self, context: ClusteringContext) -> ClusterData:
        """
        Predict cluster labels given a clustering context consisting of raw features and any parameters
        to dynamically pass to the clustering algorithm.
        :param context: clustering context
        :return: cluster labels
        """
        raise NotImplementedError


@ClusteringAlgorithm.register('sklearn_clustering_algorithm')
class SkLearnClusteringAlgorithm(ClusteringAlgorithm):
    CLUSTERING_ALGORITHM_BY_NAME = {
        'kmeans': KMeans,
        'dbscan': DBSCAN,
        'optics': OPTICS,
    }

    def __init__(
        self,
        clustering_algorithm_name: str,
        clustering_algorithm_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a clustering algorithm with scikit-learn `ClusterMixin` interface.
        :param clustering_algorithm_name: key for algorithm, currently supports 'kmeans', 'dbscan', and 'optics'
        :param clustering_algorithm_params: optional constructor parameters used to initialize clustering algorithm
        """
        super().__init__()
        # look up clustering algorithm by key
        clustering_algorithm_name = clustering_algorithm_name.lower()
        if clustering_algorithm_name not in SkLearnClusteringAlgorithm.CLUSTERING_ALGORITHM_BY_NAME:
            raise ValueError(f'Clustering algorithm "{clustering_algorithm_name}" not supported')
        self._constructor = SkLearnClusteringAlgorithm.CLUSTERING_ALGORITHM_BY_NAME[clustering_algorithm_name]
        if not clustering_algorithm_params:
            clustering_algorithm_params = {}
        self._clustering_algorithm_params = clustering_algorithm_params

    def cluster(self, context: ClusteringContext) -> ClusterData:
        # combine base parameters with any clustering parameters from the clustering context
        params = {**self._clustering_algorithm_params.copy(), **context.parameters}
        # initialize the clustering algorithm
        algorithm = self._constructor(**params)
        # predict and return cluster labels
        labels = algorithm.fit_predict(context.features).tolist()
        return ClusterData(labels)


class ClusteringMetric(Registrable):
    default_implementation = 'sklearn_clustering_metric'

    def compute(self, cluster_labels: List[int], clustering_context: ClusteringContext) -> float:
        """
        Compute clustering validity score/intrinsic metric for a clustering (higher is better).
        :param cluster_labels: candidate clustering of inputs
        :param clustering_context: clustering context, containing clustering inputs
        :return: validity score
        """
        raise NotImplementedError


@ClusteringMetric.register('sklearn_clustering_metric')
class SklearnClusteringMetric(ClusteringMetric):
    METRIC_BY_NAME = {
        'silhouette_score': metrics.silhouette_score,
        'calinski_harabasz_score': metrics.calinski_harabasz_score,
        'davies_bouldin_score': metrics.davies_bouldin_score,
    }

    def __init__(self, metric_name: str, metric_params: Dict[str, Any] = None) -> None:
        """
        Initialize a `ClusteringMetric` based on built-in scikit-learn cluster validity metrics.
        :param metric_name: name of metric to use
        :param metric_params: any parameters to pass to validity emtric
        """
        super().__init__()
        metric_name = metric_name.lower()
        if metric_name not in SklearnClusteringMetric.METRIC_BY_NAME:
            raise ValueError(f'Metric "{metric_name}" not supported')
        self._metric = SklearnClusteringMetric.METRIC_BY_NAME[metric_name]
        self._metric_params = dict(metric_params) if metric_params else {}

    def compute(
        self,
        cluster_labels: List[int],
        clustering_context: ClusteringContext,
    ) -> float:
        # skip cases where metrics may not be defined
        n_labels = len(set(cluster_labels))
        if n_labels <= 1:
            return -1
        features = clustering_context.features
        if n_labels == len(features):
            return 0
        params = self._metric_params.copy()
        return self._metric(features, cluster_labels, **params)


@ClusteringAlgorithm.register('hyperopt_tuned_clustering_algorithm')
class HyperoptTunedClusteringAlgorithm(ClusteringAlgorithm):
    NAME_TO_EXPRESSION = {
        'choice': hp.choice,
        'randint': hp.randint,
        'uniform': hp.uniform,
        'quniform': lambda *args: scope.int(hp.quniform(*args)),
        'loguniform': hp.loguniform,
        'qloguniform': lambda *args: scope.int(hp.qloguniform(*args)),
        'normal': hp.normal,
        'qnormal': lambda *args: scope.int(hp.qnormal(*args)),
        'lognormal': hp.lognormal,
        'qlognormal': lambda *args: scope.int(hp.qlognormal(*args)),
    }

    def __init__(
        self,
        clustering_algorithm: ClusteringAlgorithm,
        metric: ClusteringMetric,
        parameter_search_space: Dict[str, List[Any]],
        max_evals: int = 100,
        timeout: Optional[int] = None,
        trials_per_eval: int = 1,
        patience: int = 25,
        min_clusters: int = 5,
        max_clusters: int = 50,
        tpe_startup_jobs: int = 10,
    ) -> None:
        """
        Initialize a clustering algorithm wrapper that finds optimal clustering parameters based on
        an intrinsic cluster validity metric using hyperopt.

        :param clustering_algorithm: clustering algorithm to for which to optimize hyperparameters
        :param metric: clustering metric used to define loss
        :param parameter_search_space: parameter search space dictionary
        :param max_evals: maximum total number of trials
        :param timeout: timeout in seconds on hyperparameter search
        :param trials_per_eval: number of trials for each unique parameter setting
        :param patience: maximum number of trials to continue after no progress on loss function
        :param min_clusters: minimum number of clusters for a valid clustering
        :param max_clusters: maximum number of clusters for a valid clustering
        :param tpe_startup_jobs: number of random trials to explore search space
        """
        self._clustering_algorithm = clustering_algorithm
        self._metric = metric
        self._space = {}
        for key, value in parameter_search_space.items():
            self._space[key] = self.NAME_TO_EXPRESSION[value[0]](key, *value[1:])
        self._max_evals = max_evals
        self._timeout = timeout
        self._trials_per_eval = trials_per_eval
        self._patience = patience
        self._min_clusters = min_clusters
        self._max_clusters = max_clusters
        self._tpe_startup_jobs = tpe_startup_jobs
        # avoid verbose hyperopt logging
        loggers_to_ignore = [
            "hyperopt.tpe",
            "hyperopt.fmin",
            "hyperopt.pyll.base",
        ]
        for ignored in loggers_to_ignore:
            logging.getLogger(ignored).setLevel(logging.ERROR)

    def cluster(self, context: ClusteringContext) -> ClusterData:
        trials = Trials()
        results_by_params = {}

        def _objective(params):
            params_key = json.dumps(params, sort_keys=True)
            if params_key in results_by_params:
                # skip repeated params
                return results_by_params[params_key]

            scores = []
            labelings = []
            try:
                for seed in range(self._trials_per_eval):
                    trial_context = replace(context, parameters=params)
                    result = self._clustering_algorithm.cluster(trial_context)
                    score = self._metric.compute(result.clusters, context)
                    scores.append(score)
                    labelings.append(result.clusters)
            except ValueError:
                return {
                    'loss': -1,
                    'status': STATUS_FAIL
                }

            score = float(np.mean(scores))
            labels = labelings[int(argmax(scores))]
            n_predicted_clusters = len(set(labels))
            if not (self._min_clusters <= n_predicted_clusters <= self._max_clusters):
                return {
                    'loss': 1,
                    'status': STATUS_FAIL
                }

            result = {
                'loss': -score,
                'n_predicted_clusters': n_predicted_clusters,
                'status': STATUS_OK,
                'labels': labels
            }
            if len(scores) > 1:
                result['loss_variance'] = np.var(scores, ddof=1)
            results_by_params[params_key] = result
            return result

        tpe_partial = partial(tpe.suggest, n_startup_jobs=self._tpe_startup_jobs)
        fmin(
            _objective,
            space=self._space,
            algo=tpe_partial,
            max_evals=self._max_evals,
            trials=trials,
            timeout=self._timeout,
            rstate=np.random.default_rng(42),
            early_stop_fn=no_progress_loss(self._patience)
        )
        return ClusterData(trials.best_trial['result']['labels'])


class SentenceEmbeddingModel(Registrable):
    def encode(self, utterances: List[str]) -> np.ndarray:
        """
        Encode a list of utterances as an array of real-valued vectors.
        :param utterances: original utterances
        :return: output encoding
        """
        raise NotImplementedError


@SentenceEmbeddingModel.register('sentence_transformers_model')
class SentenceTransformersModel(SentenceEmbeddingModel):

    def __init__(self, model_name_or_path: str) -> None:
        """
        Initialize SentenceTransformers model for a given path or model name.
        :param model_name_or_path: model name or path for SentenceTransformers sentence encoder
        """
        super().__init__()
        self._sentence_transformer = model_name_or_path

    def encode(self, utterances: List[str]) -> np.ndarray:
        encoder = SentenceTransformer(self._sentence_transformer)
        return encoder.encode(utterances)


@IntentClusteringModel.register('baseline_intent_clustering_model')
class BaselineIntentClusteringModel(IntentClusteringModel):

    def __init__(
        self,
        clustering_algorithm: ClusteringAlgorithm,
        embedding_model: SentenceEmbeddingModel,
    ) -> None:
        """
        Initialize intent clustering model based on clustering utterance embeddings.
        :param clustering_algorithm: clustering algorithm applied to sentence embeddings
        :param embedding_model: sentence embedding model
        """
        super().__init__()
        self._clustering_algorithm = clustering_algorithm
        self._embedding_model = embedding_model

    def cluster_intents(self, context: IntentClusteringContext) -> Dict[str, str]:
        # collect utterances corresponding to intents
        utterances = []
        turn_ids = []
        labels = set()
        for dialogue in context.dataset.dialogues:
            for turn in dialogue.turns:
                if turn.turn_id in context.intent_turn_ids:
                    utterances.append(turn.utterance)
                    turn_ids.append(turn.turn_id)
                    labels.update(turn.intents)

        # compute sentence embeddings
        features = self._embedding_model.encode(utterances)
        # cluster sentence embeddings
        context = ClusteringContext(
            features,
            output_dir=context.output_dir,
        )
        result = self._clustering_algorithm.cluster(context)
        # map turn IDs to cluster labels
        return {turn_id: str(label) for turn_id, label in zip(turn_ids, result.clusters)}


@SentenceEmbeddingModel.register('caching_sentence_embedding_model')
class CachingSentenceEmbeddingModelSentenceTransformersModel(SentenceEmbeddingModel):

    def __init__(
        self,
        sentence_embedding_model: SentenceEmbeddingModel,
        cache_path: str,
        prefix: str,
    ) -> None:
        """
        `SentenceEmbeddingModel` wrapper that caches sentence embeddings to disk.
        :param sentence_embedding_model: wrapped sentence embedding model
        :param cache_path: path to cache sentence embeddings
        :param prefix: cache key prefix for this model
        """
        super().__init__()
        self._sentence_embedding_model = sentence_embedding_model
        self._cache_path = Path(cache_path)
        self._cache_path.mkdir(exist_ok=True, parents=True)
        self._cache_key_prefix = prefix

    def _cache_key(self, utterances: List[str]) -> Path:
        doc = '|||'.join(utterances)
        return self._cache_path / f'{self._cache_key_prefix}_{hashlib.sha256(doc.encode("utf-8")).hexdigest()}.pkl'

    def encode(self, utterances: List[str]) -> np.ndarray:
        cache_path = self._cache_key(utterances)
        if cache_path.exists():
            logger.info(f'Sentence encoder cache hit for {len(utterances)} utterances')
            with open(cache_path, "rb") as fin:
                stored_data = pickle.load(fin)
                stored_sentences = stored_data['sentences']
                stored_embeddings = stored_data['embeddings']
                if all(stored == utterance for stored, utterance in zip(stored_sentences, utterances)):
                    return stored_embeddings
                logger.info(f'Stored utterances do not match input utterances for cache key')

        logger.info(f'Sentence encoder cache miss for {len(utterances)} utterances')
        embeddings = self._sentence_embedding_model.encode(utterances)
        with open(cache_path, "wb") as fout:
            pickle.dump({'sentences': utterances, 'embeddings': embeddings}, fout, protocol=pickle.HIGHEST_PROTOCOL)

        return embeddings
