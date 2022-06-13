"""
Intent induction interface and baseline implementation.
"""
from collections import defaultdict, Counter
from pathlib import Path
from typing import List, Optional

from allennlp.common import Registrable

from sitod.constants import DialogueAct
from sitod.data import Intent, DialogueDataset
from sitod.intent_clustering import (
    IntentClusteringContext,
    IntentClusteringModel,
)


class IntentInductionModel(Registrable):
    def induce_intents(self, dataset: DialogueDataset, output_path: Optional[Path] = None) -> List[Intent]:
        """
        Induce an intent schema (intents with corresponding sample utterances) from a list of dialogues.
        :param dataset: list of dialogues
        :param output_path: optional output path for logging results
        :return: intents with sample utterances
        """
        raise NotImplementedError


@IntentInductionModel.register('baseline_intent_induction_model')
class BaselineIntentInductionModel(IntentInductionModel):
    """
    Intent induction baseline using clustering of InformIntent dialogue act predictions followed by
    cosine similarity-based sample utterance selection.
    """

    def __init__(
        self,
        intent_clustering_model: IntentClusteringModel,
    ) -> None:
        """
        Initialize intent induction baseline model from an intent clustering model.
        :param intent_clustering_model: intent clustering model
        """
        super().__init__()
        self._intent_clustering_model = intent_clustering_model

    def induce_intents(self, dataset: DialogueDataset, output_path: Optional[Path] = None) -> List[Intent]:
        # collect InformIntent dialogue act turns
        inform_intent_turn_ids = set()
        utterance_by_id = {}
        for dialogue in dataset.dialogues:
            for turn in dialogue.turns:
                if DialogueAct.INFORM_INTENT in turn.dialogue_acts:
                    inform_intent_turn_ids.add(turn.turn_id)
                    utterance_by_id[turn.turn_id] = turn.utterance

        # initialize clustering context
        clustering_context = IntentClusteringContext(
            dataset=dataset,
            intent_turn_ids=inform_intent_turn_ids,
            output_dir=output_path
        )
        # perform clustering
        cluster_assignments = self._intent_clustering_model.cluster_intents(clustering_context)

        # collect utterances
        utterances_by_cluster_id = defaultdict(list)
        for turn_id, cluster_label in cluster_assignments.items():
            utterances_by_cluster_id[cluster_label].append(utterance_by_id[turn_id])

        intents = []
        for cluster_label, utterances in utterances_by_cluster_id.items():
            sorted_utterances = [
                utt for utt, count in sorted(Counter(utterances).items(),
                                             key=lambda item: (-item[1],  # more frequent first
                                                               len(item[0]),  # shorter utterances first
                                                               item[0]))  # alphanumeric
            ]
            intent = Intent(cluster_label, sorted_utterances)
            intents.append(intent)
        return intents
