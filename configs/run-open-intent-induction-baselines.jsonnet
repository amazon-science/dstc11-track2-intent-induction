local baseline_intent_clustering_model_fn = import 'intent-clustering/baseline-intent-clustering-model.jsonnet';
local baseline_open_intent_induction_model_fn = import 'open-intent-induction/baseline-open-intent-induction-model.jsonnet';
local logistic_regression_classifier_evaluator = import 'classifier-evaluator/logistic-regression-classifier-evaluator.jsonnet';
local clustering_baselines = import 'clustering-baselines.jsonnet';

local glove = {key: 'glove-840b-300d', model: import 'embedding-model/average_word_embeddings_glove.840B.300d.jsonnet'};
local all_mpnet = {key: 'all-mpnet-base-v2', model: import 'embedding-model/all-mpnet-base-v2.jsonnet'};

local baseline_open_intent_induction_experiment(
    run_id,
    intent_clustering_model,
    embedding_model = all_mpnet,
    classifier_evaluator = logistic_regression_classifier_evaluator(),
) = {
    type: 'open_intent_induction_experiment',
    run_id: run_id + '_' + embedding_model.key,
    dialogue_reader: 'default_dialogue_reader',
    dialogues_path: 'dialogues.jsonl',
    test_utterances_path: 'test-utterances.jsonl',
    open_intent_induction_model: baseline_open_intent_induction_model_fn(
        intent_clustering_model = baseline_intent_clustering_model_fn(
            clustering_algorithm = intent_clustering_model,
            embedding_model = embedding_model.model,
        ),
    ),
    classifier_evaluator: classifier_evaluator,
};

local exp(clustering_model, embedding_model) = baseline_open_intent_induction_experiment(clustering_model.name,
    clustering_model.model, embedding_model);

{
    type: 'meta_experiment',
    run_id: 'open-intent-induction-baselines',
    datasets: ['development'],
    experiments:  [
        exp(clustering_baselines.kmeans, glove),
        exp(clustering_baselines.kmeans, all_mpnet),
    ],
}