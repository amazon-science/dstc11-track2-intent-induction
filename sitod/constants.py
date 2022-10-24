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


class MetricNames:
    CLASSIFICATION_MANY_TO_1 = 'Classification Many:1'
    CLASSIFICATION_1_TO_1 = 'Classification 1:1'
    ALIGNMENT_1_TO_1 = '1:1 Alignment'
    ALIGNMENT_CLUSTER_REF_MANY_TO_1 = 'Cluster:Ref Many:1 Alignment'
    ALIGNMENT_REF_CLUSTER_MANY_TO_1 = 'Ref:Cluster Many:1 Alignment'
    JSON_METRICS = [
        CLASSIFICATION_MANY_TO_1,
        CLASSIFICATION_1_TO_1,
        ALIGNMENT_1_TO_1,
        ALIGNMENT_CLUSTER_REF_MANY_TO_1,
        ALIGNMENT_REF_CLUSTER_MANY_TO_1
    ]
