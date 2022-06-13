class OutputPaths:
    TURN_PREDICTIONS = 'predictions.json'
    INTENTS = 'induced-intents.json'
    METRICS = 'metrics.json'
    METADATA = 'metadata.json'
    MODEL_OUTPUT_DIR = 'model'
    SUMMARY = 'summary.tsv'
    EXPERIMENTS = 'experiments.tsv'


class DialogueAct:
    """
    Task-oriented dialogue act labels.
    """
    INFORM_INTENT = 'InformIntent'


class Speaker:
    """
    Customer and agent speaker roles.
    """
    CUSTOMER = 'Customer'
    AGENT = 'Agent'


class MetadataFields:
    DATASET = 'Dataset'
    DIALOGUES_PATH = 'Dialogues Path'
    EXPERIMENT_DIR = 'Experiments Path'
    RUN_ID = 'RunID'
