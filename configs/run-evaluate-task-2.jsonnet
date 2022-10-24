local logistic_regression_classifier_evaluator = import 'classifier-evaluator/logistic-regression-classifier-evaluator.jsonnet';

local exp(
        run_id,
        schema_file_name = 'induced-intents.json',
        drop_noise_cluster = false,
        embedding_model_name = 'sentence-transformers/all-mpnet-base-v2',
) = {
    type: 'open_intent_induction_experiment',
    run_id: run_id,
    dialogue_reader: 'default_dialogue_reader',
    dialogues_path: 'dialogues.jsonl',
    test_utterances_path: 'test-utterances.jsonl',
    open_intent_induction_model: {
        type: 'external_schema_intent_induction_model',
        drop_noise_cluster: drop_noise_cluster,
        schema_file_name: schema_file_name,
    },
    classifier_evaluator: logistic_regression_classifier_evaluator(
        embedding_model = {
            type: 'sentence_transformers_model',
            model_name_or_path: embedding_model_name,
        }
    )
};

local experiments = [
    exp(run_id=team,
        schema_file_name=std.format('../%s/induced-intents.json', team))
    for team in [
        'T00',
        'T02',
        'T03',
        'T04',
        'T05',
        'T08',
        'T09',
        'T13',
        'T14',
        'T15',
        'T17',
        'T19',
        'T20',
        'T23',
        'T24',
        'T26',
        'T27',
        'T30',
        'T31',
        'T34',
        'T36',
        'Baseline',
    ]
];

{
    type: 'meta_experiment',
    run_id: '2',
    datasets: [
        'test-banking',
        'test-finance',
    ],
    experiments: experiments,
}