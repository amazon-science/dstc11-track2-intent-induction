"""
Experiments wrap execution of intent clustering and intent induction into a configurable class, allowing for
execution over multiple datasets and configurations from a single execution.
"""
import json
import logging
import os
import random
from collections import defaultdict
from numbers import Number
from pathlib import Path
from typing import List, Optional
from typing import Union, Dict

import numpy as np
import pandas as pd
import torch
from allennlp.common import Registrable

from sitod.constants import OutputPaths, MetadataFields, MetricNames
from sitod.data import (
    get_intents_by_turn_id, get_utterances_by_turn_id, TurnPrediction,
    write_turn_predictions, read_intents, read_turn_predictions, DialogueDataset,
)
from sitod.data import write_intents
from sitod.evaluate import evaluate_schema, ClassifierEvaluator, LogisticRegressionClassifierEvaluator
from sitod.intent_clustering import (
    IntentClusteringModel, IntentClusteringContext,
)
from sitod.metric import compute_metrics_from_turn_predictions, schema_metrics
from sitod.metric import format_mean_and_stdev, compute_mean_and_stdev
from sitod.open_intent_induction import IntentInductionModel
from sitod.reader import DialogueReader, IntentTestDataReader

logger = logging.getLogger(__name__)


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Experiment(Registrable):

    def __init__(self, run_id, metadata: Dict[str, str] = None) -> None:
        """
        Initialize an experiment.
        :param run_id: unique identifier for this experiment
        :param metadata: dictionary containing metadata associated with experiment
        """
        super().__init__()
        self.run_id = run_id
        if not metadata:
            metadata = {}
        self.metadata = metadata

    def _run_experiment(
        self,
        data_root_dir: Path,
        experiment_dir: Path,
    ):
        raise NotImplementedError

    def run_experiment(
        self,
        data_root_dir: Union[str, bytes, os.PathLike],
        experiment_root_dir: Union[str, bytes, os.PathLike],
    ):
        """
        Run an experiment in a given experiment root directory.

        :param data_root_dir: dataset root directory (define dataset paths relative to this)
        :param experiment_root_dir: experiment root directory to save experiment results
        """
        experiment_root_dir = Path(experiment_root_dir) / self.run_id
        experiment_root_dir.mkdir(parents=True, exist_ok=True)

        # write metadata
        metadata = {**self.metadata}
        # add default fields to metadata
        if MetadataFields.RUN_ID not in metadata:
            metadata[MetadataFields.RUN_ID] = self.run_id
        if MetadataFields.DATASET not in metadata:
            metadata[MetadataFields.DATASET] = str(data_root_dir)
        if MetadataFields.EXPERIMENT_DIR not in metadata:
            metadata[MetadataFields.EXPERIMENT_DIR] = str(experiment_root_dir)
        (experiment_root_dir / OutputPaths.METADATA).write_text(json.dumps(metadata, indent=True))

        self._run_experiment(Path(data_root_dir), experiment_root_dir)


@Experiment.register('intent_clustering_experiment')
class IntentClusteringExperiment(Experiment):
    def __init__(
        self,
        run_id: str,
        dialogue_reader: DialogueReader,
        dialogues_path: Union[str, bytes, os.PathLike],
        intent_clustering_model: IntentClusteringModel,
        metadata: Dict[str, str] = None,
        ignored_labels: List[str] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Initialize an intent clustering experiment.
        :param dialogue_reader: reader for input dialogues
        :param dialogues_path: relative path to dialogues from dataset root
        :param intent_clustering_model: intent clustering model toe valuate
        :param run_id: unique identifier for this experiment
        :param metadata: dictionary containing metadata associated with experiment
        :param ignored_labels: reference labels to exclude from evaluation
        :param overwrite: whether to overwrite any pre-existing results
        """
        super().__init__(run_id, metadata)
        self._dialog_reader = dialogue_reader
        self._dialogues_path = dialogues_path
        self._intent_clustering_model = intent_clustering_model
        if MetadataFields.DIALOGUES_PATH not in self.metadata:
            self.metadata[MetadataFields.DIALOGUES_PATH] = dialogues_path
        if not ignored_labels:
            ignored_labels = []
        self._ignored_labels = list(ignored_labels)
        self._overwrite = overwrite

    def _run_experiment(
        self,
        data_root_dir: Path,
        experiment_dir: Path,
    ):
        # read dialogues and collect intent turns
        dialogues = self._dialog_reader.read_dialogues(data_root_dir / self._dialogues_path)
        intents_by_turn_id = get_intents_by_turn_id(dialogues)
        logger.info(f'Read {len(dialogues)} dialogues and {len(intents_by_turn_id)} '
                    f'intent turns from {data_root_dir / self._dialogues_path}')

        # run clustering
        if not (experiment_dir / OutputPaths.TURN_PREDICTIONS).exists() or self._overwrite:
            utterances_by_turn_id = get_utterances_by_turn_id(dialogues)
            label_assignments = self._intent_clustering_model.cluster_intents(
                IntentClusteringContext(
                    DialogueDataset(data_root_dir.name, dialogues),
                    set(intents_by_turn_id),
                    output_dir=experiment_dir
                )
            )

            # write label assignments, evaluate
            turn_predictions = []
            for turn_id, reference_label in intents_by_turn_id.items():
                turn_predictions.append(TurnPrediction(
                    predicted_label=label_assignments[turn_id],
                    reference_label=reference_label,
                    utterance=utterances_by_turn_id[turn_id],
                    turn_id=turn_id
                ))

            # write predictions and metrics
            write_turn_predictions(turn_predictions, experiment_dir / OutputPaths.TURN_PREDICTIONS)
        else:
            logger.info(f'Reading pre-computed predictions at {experiment_dir / OutputPaths.TURN_PREDICTIONS}')
        turn_predictions = read_turn_predictions(experiment_dir / OutputPaths.TURN_PREDICTIONS)
        for prediction in turn_predictions:
            prediction.reference_label = intents_by_turn_id[prediction.turn_id]

        # write metrics JSON
        metrics = compute_metrics_from_turn_predictions(turn_predictions, ignore_labels=self._ignored_labels)
        logger.info(f'{data_root_dir / self._dialogues_path}')
        logger.info(f'{json.dumps(metrics, indent=True)}')
        (experiment_dir / OutputPaths.METRICS).write_text(json.dumps(metrics, indent=True))


@Experiment.register('open_intent_induction_experiment')
class OpenIntentInductionExperiment(Experiment):
    def __init__(
        self,
        dialogue_reader: DialogueReader,
        dialogues_path: Union[str, bytes, os.PathLike],
        test_utterances_path: Union[str, bytes, os.PathLike],
        open_intent_induction_model: IntentInductionModel,
        test_dialogue_reader: Optional[DialogueReader] = None,
        classifier_evaluator: Optional[ClassifierEvaluator] = None,
        run_id: Optional[str] = None,
        metadata: Optional[Dict[str, str]] = None,
        eval_seeds: Optional[List[int]] = None,
        ignored_labels: List[str] = None,
        overwrite: bool = False,
    ) -> None:
        """
        Initialize an open intent induction experiment.

        :param dialogue_reader: reader for input dialogues
        :param dialogues_path: relative path to dialogues from dataset root
        :param run_id: unique identifier for this experiment
        :param metadata: dictionary containing metadata associated with experiment
        :param ignored_labels: reference labels to exclude from evaluation
        :param overwrite: whether to overwrite any pre-existing results
        :param test_utterances_path: relative path to test utterances file from dataset root
        :param open_intent_induction_model: model used for open intent induction
        :param test_dialogue_reader: dialogue reader used for test data
        :param classifier_evaluator: evaluator used for prediction-based evaluation of induced intents
        :param eval_seeds: random seeds used for prediction-based evaluation of induced intents
        """
        super().__init__(run_id, metadata)
        self._dialog_reader = dialogue_reader
        self._test_dialog_reader = test_dialogue_reader if test_dialogue_reader else IntentTestDataReader()
        self._dialogues_path = Path(dialogues_path)
        self._test_utterances_path = Path(test_utterances_path)
        self._open_intent_induction_model = open_intent_induction_model
        if not classifier_evaluator:
            classifier_evaluator = LogisticRegressionClassifierEvaluator()
        self._classifier_evaluator = classifier_evaluator
        self._eval_seeds = eval_seeds if eval_seeds else [42]
        if not ignored_labels:
            ignored_labels = []
        self._ignored_labels = list(ignored_labels)
        self._overwrite = overwrite

    def _run_experiment(
        self,
        data_root_dir: Path,
        experiment_dir: Path
    ):
        # induce schema
        if not (experiment_dir / OutputPaths.INTENTS).exists() or self._overwrite:
            dialogues = self._dialog_reader.read_dialogues(data_root_dir / self._dialogues_path)
            logger.info(f'Read {len(dialogues)} dialogues')
            dataset = DialogueDataset(data_root_dir.name, dialogues)
            intent_schema = self._open_intent_induction_model.induce_intents(dataset,
                                                                             output_path=experiment_dir)
            # output intents
            write_intents(intent_schema, experiment_dir / OutputPaths.INTENTS)
        else:
            logger.info(f'Reading pre-existing intent schema at {experiment_dir / OutputPaths.INTENTS}')
        intent_schema = read_intents(experiment_dir / OutputPaths.INTENTS)

        # read input dialogues and test dialogues
        test_dialogues = self._test_dialog_reader.read_dialogues(data_root_dir / self._test_utterances_path)
        # evaluate schema
        intent_induction_metrics = evaluate_schema(
            schema=intent_schema,
            test_data=test_dialogues,
            classifier_evaluator=self._classifier_evaluator,
            base_output_dir=experiment_dir,
            seeds=self._eval_seeds,
            ignored_labels=self._ignored_labels,
            overwrite=self._overwrite
        )
        # output results
        skip_keys = [key for key, vals in intent_induction_metrics.items() if not isinstance(vals[0], Number)]
        mean, stdev = compute_mean_and_stdev(intent_induction_metrics, skip_keys=skip_keys)
        mean.update(schema_metrics(intent_schema))
        (experiment_dir / OutputPaths.METRICS).write_text(json.dumps(mean, indent=True))
        formatted_metrics = format_mean_and_stdev(mean, stdev)
        (experiment_dir / OutputPaths.SUMMARY).write_text(json.dumps(formatted_metrics, indent=True))
        logger.info(json.dumps(formatted_metrics, indent=True))


@Experiment.register('meta_experiment')
class MetaExperiment(Experiment):
    def __init__(
        self,
        run_id,
        experiments: List[Experiment],
        metadata: Dict[str, str] = None,
        datasets: List[str] = None,
        initial_seed: int = 0,
        sort_by_fields: List[str] = None,
        ascending: List[bool] = None,
        skip_metrics: List[str] = None,
    ) -> None:
        """
        Initialize a meta experiment, which runs multiple experiments across a list of datasets and aggregates results.
        :param run_id: unique identifier for this experiment
        :param experiments: list of experiments to run on each dataset
        :param metadata: dictionary containing metadata associated with experiment
        :param datasets: list of datasets
        :param initial_seed: random seed set before each experiment
        :param sort_by_fields: fields to sort rows in resulting metrics tables
        :param ascending: boolean list indicating sort direction for each field
        :param skip_metrics: metrics to drop from summary
        """
        super().__init__(run_id, metadata)
        self._datasets = datasets
        self._experiments = experiments
        self._initial_seed = initial_seed
        self._sort_by_fields = sort_by_fields if sort_by_fields else ['ACC', MetadataFields.RUN_ID]
        self._ascending = ascending if ascending else [False, True]
        if not skip_metrics:
            skip_metrics = MetricNames.JSON_METRICS
        self._skip_metrics = skip_metrics
        assert len(self._ascending) == len(self._sort_by_fields)

    def _run_experiment(self, data_root_dir: Path, experiment_dir: Path):
        for experiment in self._experiments:
            for dataset in self._datasets:
                logger.info(f'Running {experiment.run_id} on {dataset}')
                _set_seed(self._initial_seed)
                experiment.run_experiment(data_root_dir / dataset, experiment_dir / dataset)

        # collect metrics / metadata
        fields_by_config = defaultdict(lambda: defaultdict(list))
        fields = defaultdict(list)
        for metrics_path in (experiment_dir.glob(f'*/*/{OutputPaths.METRICS}')):
            metadata_path = metrics_path.parent / OutputPaths.METADATA
            metrics = json.loads(metrics_path.read_bytes())
            metadata = json.loads(metadata_path.read_bytes())
            config = metadata[MetadataFields.RUN_ID]
            for k, val in metadata.items():
                fields_by_config[config][k].append(val)
                fields[k].append(val)
            for k, val in metrics.items():
                if k in self._skip_metrics:
                    continue
                fields_by_config[config][k].append(val)
                fields[k].append(val)

        df = pd.DataFrame(fields).sort_values(by=[MetadataFields.DATASET] + self._sort_by_fields,
                                              ascending=[True] + self._ascending)
        df.to_csv((experiment_dir / OutputPaths.EXPERIMENTS), sep='\t', index=False, float_format='%.4f')
        logger.info('\n' + df.to_csv(sep='\t', index=False, float_format='%.2f'))

        # summarize with per-config metric averages
        summary_fields = defaultdict(list)
        for config, dataset_fields in fields_by_config.items():
            # compute metrics averaged over datasets
            means, stdevs = compute_mean_and_stdev(
                dataset_fields,
                skip_keys=[key for key, vals in dataset_fields.items() if not isinstance(vals[0], Number)]
            )
            summary_fields[MetadataFields.RUN_ID].append(config)
            for key, average in means.items():
                summary_fields[key].append(average)

        df = pd.DataFrame(summary_fields).sort_values(by=self._sort_by_fields, ascending=self._ascending)
        df.to_csv((experiment_dir / OutputPaths.SUMMARY), sep='\t', index=False, float_format='%.4f')
        logger.info('\n' + df.to_csv(sep='\t', index=False, float_format='%.1f'))
