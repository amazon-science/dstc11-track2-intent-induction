local baseline_intent_clustering_model_fn = import 'intent-clustering/baseline-intent-clustering-model.jsonnet';
local clustering_baselines = import 'clustering-baselines.jsonnet';
local glove = {key: 'glove-840b-300d', model: import 'embedding-model/average_word_embeddings_glove.840B.300d.jsonnet'};
local all_mpnet = {key: 'all-mpnet-base-v2', model: import 'embedding-model/all-mpnet-base-v2.jsonnet'};

local intent_clustering_experiment(run_id, intent_clustering_model) = {
    type: 'intent_clustering_experiment',
    run_id: run_id,
    dialogue_reader: 'default_dialogue_reader',
    dialogues_path: 'dialogues.jsonl',
    intent_clustering_model: intent_clustering_model
};

local ic_exp(baseline, embedding_model) = intent_clustering_experiment(
    baseline.name + '_' + embedding_model.key,
    baseline_intent_clustering_model_fn(
        clustering_algorithm = baseline.model,
        embedding_model = embedding_model.model,
    )
);

{
    type: 'meta_experiment',
    run_id: 'intent-clustering-baselines',
    datasets: ['development'],
    experiments:  [
        ic_exp(clustering_baselines.kmeans, glove),
        ic_exp(clustering_baselines.kmeans, all_mpnet),
    ],
}