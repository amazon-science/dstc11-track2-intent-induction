local intent_clustering_model(
    clustering_algorithm,
    embedding_model,
) = {
    type: 'baseline_intent_clustering_model',
    clustering_algorithm: clustering_algorithm,
    embedding_model: embedding_model
};

intent_clustering_model