local logistic_regression_classifier_evaluator = import 'classifier-evaluator/logistic-regression-classifier-evaluator.jsonnet';

local exp(run_id) = {
    type: 'intent_clustering_experiment',
    run_id: run_id,
    dialogue_reader: 'default_dialogue_reader',
    dialogues_path: 'dialogues.jsonl',
    intent_clustering_model: {type: 'external_predictions_intent_clustering_model'}
};

local experiments = [
    exp(team)
    for team in [
        'T00',
        'T01',
        'T02',
        'T03',
        'T04',
        'T05',
        'T06',
        'T07',
        'T08',
        'T09',
        'T10',
        'T11',
        'T13',
        'T14',
        'T15',
        'T16',
        'T17',
        'T18',
        'T19',
        'T20',
        'T21',
        'T22',
        'T23',
        'T24',
        'T25',
        'T26',
        'T27',
        'T28',
        'T29',
        'T30',
        'T31',
        'T32',
        'T33',
        'T34',
        'T35',
        'T36',
        'T37',
        'Baseline',
    ]
];

{
    type: 'meta_experiment',
    run_id: '1',
    datasets: [
        'test-banking',
        'test-finance',
    ],
    experiments: experiments,
}