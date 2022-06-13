import json
import os
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Union


@dataclass
class Turn:
    """
    Dialogue turn, consisting of a unique ID, speaker role, utterance, list of dialogue acts and intents.
    """
    turn_id: str
    speaker_role: str
    utterance: str
    dialogue_acts: List[str] = field(default_factory=list)
    intents: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, json_dict: Dict):
        return Turn(
            turn_id=json_dict['turn_id'],
            speaker_role=json_dict['speaker_role'],
            utterance=json_dict['utterance'],
            dialogue_acts=json_dict['dialogue_acts'],
            intents=json_dict['intents']
        )


@dataclass
class Dialogue:
    """
    Dialogue, consisting of a unique ID and list of turns.
    """
    dialogue_id: str
    turns: List[Turn]

    @classmethod
    def from_dict(cls, json_dict: Dict):
        return Dialogue(
            dialogue_id=json_dict['dialogue_id'],
            turns=[Turn.from_dict(turn) for turn in json_dict['turns']]
        )


@dataclass
class DialogueDataset:
    """
    Dialogue dataset, a wrapper for a list of dialogues with the original dataset path the dialogues were read from.
    """
    dataset_path: str
    dialogues: List[Dialogue]


def get_intents_by_turn_id(dialogues: List[Dialogue]) -> Dict[str, str]:
    """
    Return a mapping from turn IDs in dialogues to ground truth intent labels.
    :param dialogues: input dialogues
    :return: mapping from turn IDs to ground truth intent labels
    """
    intent_turns = {}
    for dialog in dialogues:
        for turn in dialog.turns:
            if turn.intents:
                intent_turns[turn.turn_id] = turn.intents[0]
    return intent_turns


def get_utterances_by_turn_id(dialogues: List[Dialogue]) -> Dict[str, str]:
    """
    Return a direct mapping from turn IDs in dialogues to corresponding utterances.
    :param dialogues: input dialogues
    :return: mapping from turn IDs to to utterances
    """
    utterances = {}
    for dialog in dialogues:
        for turn in dialog.turns:
            utterances[turn.turn_id] = turn.utterance
    return utterances


@dataclass
class Intent:
    """
    Data class containing an induced intent (intent ID and corresponding utterances).
    """
    intent_id: str
    utterances: List[str]

    @classmethod
    def from_dict(cls, json_dict: Dict):
        return Intent(
            intent_id=json_dict['intent_id'],
            utterances=json_dict['utterances'],
        )


@dataclass
class TurnPrediction:
    """
    Data class used to hold turn-level predictions (and any relevant metadata).
    """
    predicted_label: str
    reference_label: str
    utterance: str
    turn_id: str

    @classmethod
    def from_dict(cls, json_dict: Dict):
        return TurnPrediction(
            predicted_label=str(json_dict['predicted_label']),
            reference_label=str(json_dict.get('reference_label', '')),
            utterance=json_dict.get('utterance', ''),
            turn_id=json_dict['turn_id'],
        )


def write_list_jsonl(data: List[Union[TurnPrediction, Intent, Dialogue]], output_path: Union[str, bytes, os.PathLike]):
    """
    Write a list of dataclasses to a JSONL-formatted file using `asdict` for serialization.
    :param data: dataclasses to write
    :param output_path: output path to write JSONL file
    """
    with open(output_path, mode='w') as out:
        for prediction in data:
            out.write(json.dumps(asdict(prediction)) + '\n')


def _read_list_jsonl(initializer, path: Union[str, bytes, os.PathLike]):
    predictions = []
    with open(path, mode='r') as lines:
        for line in lines:
            if not line:
                continue
            predictions.append(initializer.from_dict(json.loads(line)))
    return predictions


def write_turn_predictions(predictions: List[TurnPrediction], path: Union[str, bytes, os.PathLike]):
    """
    Write a list of turn-level predictions to a file (JSONL format).
    :param predictions: turn-level predictions
    :param path: output path
    """
    write_list_jsonl(predictions, path)


def read_turn_predictions(path: Union[str, bytes, os.PathLike]) -> List[TurnPrediction]:
    """
    Read a list of turn-level predictions from a file (JSONL format).
    :param path: path to read predictions from
    :return: list of turn-level predictions
    """
    return _read_list_jsonl(TurnPrediction, path)


def write_intents(intents: List[Intent], path: Union[str, bytes, os.PathLike]):
    """
    Write a list of turn-level predictions to a file (JSONL format).
    :param intents: intent schemas
    :param path: output path
    """
    write_list_jsonl(intents, path)


def read_intents(path: Union[str, bytes, os.PathLike]) -> List[Intent]:
    """
    Read a list of intents from a given path.
    :param path: path to read intents from
    :return: list of intents
    """
    return _read_list_jsonl(Intent, path)
