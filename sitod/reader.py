"""
Readers for intent induction datasets.
"""
import json
import os
from pathlib import Path
from typing import Union, List, Dict, Iterable, Tuple

import pandas as pd
from allennlp.common import Registrable

from sitod.constants import Speaker, DialogueAct
from sitod.data import Dialogue, Turn


class DialogueReader(Registrable):
    default_implementation = 'default_dialogue_reader'

    def read_dialogues(self, path: Union[str, bytes, os.PathLike]) -> List[Dialogue]:
        """
        Read a list of dialogues from a given path.
        :param path: dialogues path
        :return: list of dialogues
        """
        raise NotImplementedError


@DialogueReader.register('default_dialogue_reader')
class DefaultDialogueReader(DialogueReader):
    """
    `DialogueReader` expecting a JSONL input file consisting of a dialogue JSON per line.
    ```
    {"dialogue_id": "000", "turns": [{"turn_id": "000_000", "speaker_role": "Agent", "utterance": "...}
    {"dialogue_id": "001", "turns": [{"turn_id": "001_000", "speaker_role": "Customer", "utterance": "...}
    ...
    ```
    """

    def read_dialogues(self, path: Union[str, bytes, os.PathLike]) -> List[Dialogue]:
        dialogues = []
        with Path(path).open() as lines:
            for line in lines:
                if not line:
                    continue
                dialogues.append(Dialogue.from_dict(json.loads(line)))
        return dialogues


class IntentUtteranceReader(DialogueReader):
    FORMAT_TO_READER = {
        'csv': pd.read_csv,
        'excel': pd.read_excel,
        'jsonl': pd.read_json,
    }

    def __init__(
        self,
        utterance_field: str,
        intent_label_field: str,
        utterance_id_field: str = None,
        file_format: str = 'csv',
        **reader_kwargs
    ) -> None:
        super().__init__()
        self._utterance_field = utterance_field
        self._intent_label_field = intent_label_field
        self._utterance_id_field = utterance_id_field
        if file_format not in IntentUtteranceReader.FORMAT_TO_READER:
            raise ValueError(f'Format "{file_format}" not supported. '
                             f'Select from: {", ".join(IntentUtteranceReader.FORMAT_TO_READER.keys())}')
        self._reader = IntentUtteranceReader.FORMAT_TO_READER[file_format]
        self._reader_params = reader_kwargs

    def _read_utterances(self, path: Union[str, bytes, os.PathLike]) -> Iterable[Tuple[int, Dict]]:
        df = self._reader(path, **self._reader_params)
        required_fields = [self._utterance_field, self._intent_label_field]
        if self._utterance_id_field:
            required_fields.append(self._utterance_id_field)
        df.dropna(subset=required_fields, inplace=True)
        return df.iterrows()

    def _convert_utterance_to_dialogue(self, uid: str, line: Dict) -> Dialogue:
        return Dialogue(dialogue_id=uid, turns=[Turn(
            turn_id=uid,
            speaker_role=Speaker.CUSTOMER,
            utterance=line[self._utterance_field],
            dialogue_acts=[DialogueAct.INFORM_INTENT],
            intents=[line[self._intent_label_field]]
        )])

    def read_dialogues(self, path: Union[str, bytes, os.PathLike]) -> List[Dialogue]:
        path_id = Path(path).with_suffix('')
        name = path_id.name if path_id.parent == path else f'{path_id.parent.name}/{path_id.name}'
        lines = []
        for i, line in self._read_utterances(path):
            uid = line[self._utterance_id_field] if self._utterance_id_field else f'{name}_{i}'
            lines.append(self._convert_utterance_to_dialogue(uid, line))
        return lines


@DialogueReader.register('intent_test_data_reader')
class IntentTestDataReader(IntentUtteranceReader):
    """
    `DialogueReader` expecting a JSONL file consisting of a turn per line:
    ```
    {"utterance": "I need a quote for my car", "utterance_id": "00", "intent": "GetQuote"}
    {"utterance": "Yeah, I'm looking to get automobile insurance", "utterance_id": "01", "intent": "GetQuote"}
    ...
    ```
    """

    def __init__(
        self,
        utterance_field: str = 'utterance',
        intent_label_field: str = 'intent',
        utterance_id_field: str = 'utterance_id',
    ) -> None:
        super().__init__(
            utterance_field=utterance_field,
            intent_label_field=intent_label_field,
            utterance_id_field=utterance_id_field,
            file_format='jsonl',
            lines=True
        )
