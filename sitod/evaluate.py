"""
Classes and functions used for end-to-end evaluation of open intent induction.
"""
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Set, Dict, Union, Any

from allennlp.common import Registrable
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
# noinspection PyProtectedMember
from sklearn.utils._testing import ignore_warnings

from sitod.constants import OutputPaths
from sitod.data import Intent, Dialogue, write_turn_predictions, TurnPrediction, read_turn_predictions
from sitod.intent_clustering import SentenceEmbeddingModel
from sitod.metric import (
    compute_metrics_from_turn_predictions,
)


class ClassifierEvaluator(Registrable):
    default_implementation = 'logistic_regression_classifier_evaluator'

    def predict_labels(
        self,
        intent_schema: List[Intent],
        test_dialogues: List[Dialogue],
        target_turns: Set[str],
        output_dir: Union[str, bytes, os.PathLike],
        seed: int = 42,
    ) -> Dict[str, str]:
        """
        Train a classifier and predict labels on target turns within test dialogues given an intent schema.

        :param intent_schema: intent schema, intents with associated training sample utterances
        :param test_dialogues: dialogues used for evaluation of intent schema
        :param target_turns: target turn IDs used for evaluation within test dialogues
        :param output_dir: output directory to store any intermediate results
        :param seed: random seed
        :return: predicted labels for each target turn (by turn ID)
        """
        raise NotImplementedError


@ClassifierEvaluator.register('logistic_regression_classifier_evaluator')
class LogisticRegressionClassifierEvaluator(ClassifierEvaluator):

    def __init__(
        self,
        sentence_encoder: SentenceEmbeddingModel,
        classifier_settings: Dict[str, Any] = None,
    ) -> None:
        """
        `ClassifierEvaluator` implementation that uses logistic regression with fixed sentence embeddings
        to predict cluster labels on test data.

        :param sentence_encoder: sentence embedding model
        :param classifier_settings: classifier training hyperparameters
        """
        super().__init__()
        self._sentence_encoder = sentence_encoder
        self._classifier_settings = classifier_settings if classifier_settings else {'class_weight': 'balanced'}

    @ignore_warnings(category=ConvergenceWarning)
    def predict_labels(
        self,
        intent_schema: List[Intent],
        test_dialogues: List[Dialogue],
        target_turns: Set[str],
        output_dir: Union[str, bytes, os.PathLike],
        seed: int = 42
    ) -> Dict[str, str]:
        # load data
        utterances = []
        labels = []
        for schema in intent_schema:
            utterances.extend(schema.utterances)
            labels.extend([schema.intent_id] * len(schema.utterances))
        test_utterances = []
        turn_ids = []
        for dialogue in test_dialogues:
            for turn in dialogue.turns:
                if turn.turn_id in target_turns:
                    test_utterances.append(turn.utterance)
                    turn_ids.append(turn.turn_id)

        # initialize and train classifier
        train_embeddings = self._sentence_encoder.encode(utterances)
        classifier = LogisticRegression(**self._classifier_settings, random_state=seed)
        le = LabelEncoder()
        labels = le.fit_transform(labels)
        if len(le.classes_) <= 1:
            # special case for when only one label is available (and a classifier can't be trained)
            label = 'None' if not le.classes_ else le.classes_[0]
            return {turn_id: label for turn_id in target_turns}

        classifier.fit(X=train_embeddings, y=labels)

        # predict labels on test utterances
        test_embeddings = self._sentence_encoder.encode(utterances=test_utterances)
        predicted_label_ids = classifier.predict(X=test_embeddings)
        predicted_labels = le.inverse_transform(y=predicted_label_ids).tolist()
        return {turn_id: label for label, turn_id in zip(predicted_labels, turn_ids)}


def evaluate_schema(
    schema: List[Intent],
    test_data: List[Dialogue],
    classifier_evaluator: ClassifierEvaluator,
    base_output_dir: Union[str, bytes, os.PathLike],
    seeds: List[int] = None,
    ignored_labels: List[str] = None,
    overwrite: bool = False,
) -> Dict[str, List[float]]:
    """
    Evaluate an induced intent schema against a list of test utterances.

    :param schema: intent schema
    :param test_data: test utterances
    :param classifier_evaluator: classifier evaluator used to predict cluster labels on test data
    :param base_output_dir: output directory to save results
    :param seeds: optional list of random seeds ([42] by default)
    :param ignored_labels: reference labels to ignore during evaluation
    :param overwrite: whether to overwrite output directory if already existing, or return pre-existing results
    :return: metric dictionary (metric keys to a list of scores for each random seed)
    """
    if not seeds:
        seeds = [42]

    # collect intent utterances from test data
    intent_labels = {}
    turn_to_utterance = {}
    for dialog in test_data:
        for turn in dialog.turns:
            if turn.intents:
                intent_labels[turn.turn_id] = turn.intents[0]
                turn_to_utterance[turn.turn_id] = turn.utterance

    results = defaultdict(list)
    for seed in seeds:
        output_dir = Path(base_output_dir) / f'seed-{seed}'
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
        if (output_dir / OutputPaths.TURN_PREDICTIONS).exists() and not overwrite:
            turn_predictions = read_turn_predictions(output_dir / OutputPaths.TURN_PREDICTIONS)
        else:
            # train classifier and make predictions for this seed
            predictions = classifier_evaluator.predict_labels(
                intent_schema=schema,
                test_dialogues=test_data,
                target_turns=set(intent_labels),
                output_dir=output_dir / OutputPaths.MODEL_OUTPUT_DIR,
                seed=seed)
            # collect predictions
            ground_truth_labels = []
            predicted_labels = []
            turn_predictions = []
            for turn_id, predicted_label in predictions.items():
                predicted_labels.append(predicted_label)
                ground_truth_labels.append(intent_labels[turn_id])
                turn_predictions.append(TurnPrediction(
                    turn_id=turn_id,
                    predicted_label=predicted_label,
                    reference_label=intent_labels[turn_id],
                    utterance=turn_to_utterance[turn_id])
                )
            # write predictions
            write_turn_predictions(turn_predictions, output_dir / OutputPaths.TURN_PREDICTIONS)
        # write metrics
        metrics = compute_metrics_from_turn_predictions(turn_predictions, ignore_labels=ignored_labels)
        (output_dir / OutputPaths.METRICS).write_text(json.dumps(metrics, indent=True, sort_keys=True))
        for key, val in metrics.items():
            results[key].append(val)

    return results
