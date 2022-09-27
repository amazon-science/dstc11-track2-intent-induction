"""
Script for validating task entries for DSTC 11 Track 2.

Example usage:

# task 1 validation
python validate_task_inputs.py predictions.json --task 1 --dialogues test-banking/dialogues.jsonl

# task 2 validation
python validate_task_inputs.py induced-intents.json --task 2

"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set


def read_turn_ids(dialogues_path: Path) -> Set[str]:
    turn_ids = set()
    with dialogues_path.open(encoding='utf-8') as lines:
        for line in lines:
            if not line:
                continue
            dialogue = json.loads(line)
            turns = dialogue['turns']
            for turn in turns:
                if turn['intents']:
                    turn_ids.add(turn['turn_id'])
    return turn_ids


def read_task_2_intent_schema(input_path: Path) -> Dict[str, List[str]]:
    utterances_by_label = {}
    with input_path.open(encoding='utf-8') as lines:
        for line in lines:
            if not line:
                continue
            intent = json.loads(line)
            intent_id = intent['intent_id']
            if not isinstance(intent_id, str):
                raise ValueError(f'"intent_id" field should be a String, got {intent_id}')
            if intent_id in utterances_by_label:
                raise ValueError(f'Intent with intent_id "{intent_id}" appears multiple times')
            utterances = intent['utterances']
            if not utterances:
                raise ValueError(f'"utterances" field for intent_id "{intent_id}" are empty')
            for i, utterance in enumerate(utterances):
                if not isinstance(utterance, str):
                    raise ValueError(f'"utterances" should be a list of Strings')
                if not utterance.strip():
                    raise ValueError(f'Utterance {i} for intent_id "{intent_id}" is blank')
            utterances_by_label[intent_id] = utterances
    if len(utterances_by_label) < 2:
        raise ValueError(f'Expecting at least 2 intents, got {len(utterances_by_label)}')
    return utterances_by_label


def read_task_1_predictions(input_path: Path, dialogues_path: Path = None) -> Dict[str, List[str]]:
    turn_ids_by_cluster_label = defaultdict(list)
    seen_ids = set()
    with input_path.open(encoding='utf-8') as lines:
        for line in lines:
            if not line:
                continue
            prediction = json.loads(line)
            turn_id = prediction['turn_id']
            if not isinstance(turn_id, str):
                raise ValueError(f'"turn_id" field should be a String, got {turn_id}')
            if turn_id in seen_ids:
                print(f'Warning: Turn {turn_id} has multiple cluster predictions')
            seen_ids.add(turn_id)
            predicted_label = prediction['predicted_label']
            if not isinstance(predicted_label, str):
                raise ValueError(f'"predicted_label" field should be a String, got {predicted_label}')
            turn_ids_by_cluster_label[predicted_label].append(turn_id)
    if len(turn_ids_by_cluster_label) < 2:
        raise ValueError(f'Expecting at least 2 clusters, got {len(turn_ids_by_cluster_label)}')

    if dialogues_path:
        dialogues_turn_ids = read_turn_ids(dialogues_path)
        clustered_turn_uids = set()
        for cluster_label, turn_ids in turn_ids_by_cluster_label.items():
            for turn_id in turn_ids:
                if turn_id in clustered_turn_uids:
                    print(f'Warning: Turn {turn_id} assigned multiple cluster labels')
            clustered_turn_uids.update(turn_ids)
        cluster_uids_not_in_dialogues = clustered_turn_uids.difference(dialogues_turn_ids)
        if cluster_uids_not_in_dialogues:
            raise ValueError(f'Clustered turns not found in inputs: {cluster_uids_not_in_dialogues}')
        dialogue_uids_not_clustered = dialogues_turn_ids.difference(clustered_turn_uids)
        if dialogue_uids_not_clustered:
            raise ValueError(f'Input turns not assigned cluster labels: {dialogue_uids_not_clustered}')

    return turn_ids_by_cluster_label


def validate(output_file: str, task: str, dialogues_file: str = None) -> bool:
    output_file_path = Path(output_file)
    if not output_file_path.exists():
        print(f'File "{output_file_path}" not found.')
        return False

    if task == '1':
        if not dialogues_file:
            print(f'For Task 1 validation, dialogues path is required '
                  f'(e.g. --dialogues test-banking/dialogues.jsonl)')
            return False
        dialogues_path = Path(dialogues_file)
        if not dialogues_path.exists():
            print(f'File "{dialogues_path}" not found (expecting something like '
                  f'"test-banking/dialogues.jsonl").')
            return False
        clusters = read_task_1_predictions(output_file_path, dialogues_path)
        print(f'Successfully read cluster predictions for {len(clusters)} clusters and '
              f'{sum(len(utts) for utts in clusters.values())} turns.')
    elif task == '2':
        intent_schema = read_task_2_intent_schema(output_file_path)
        print(f'Successfully read {len(intent_schema)} intents with '
              f'{sum(len(utts) for utts in intent_schema.values())} total utterances.')
    else:
        raise ValueError(f'Unknown Task ID {task}')

    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output_file', type=str, help='Path to task output file to validate')
    parser.add_argument('--task', type=str, required=True, choices=['1', '2'], default='1',
                        help="Task ID, either 1 or 2")
    parser.add_argument('--dialogues', type=str,
                        help='Path to conversations, e.g. "test-banking/dialogues.jsonl"')
    args = parser.parse_args()
    validation_result = validate(args.output_file, args.task, args.dialogues)
    print(f'\nValidation of {args.output_file} for Task {args.task} '
          f'{"successful" if validation_result else "failed"}!')
    if not validation_result:
        exit(1)
