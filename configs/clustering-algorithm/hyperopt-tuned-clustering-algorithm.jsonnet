local silhouette_score = {
    type: 'sklearn_clustering_metric',
    metric_name: 'silhouette_score',
    metric_params: {
        metric: 'cosine'
    }
};

local tuned(
    clustering_algorithm,
    metric = silhouette_score,
    parameter_search_space,
    max_evals = 100,
    patience = 25,
    trials_per_eval = 1,
    min_clusters = 5,
    max_clusters = 50,
    tpe_startup_jobs = 10,
) = {
    type: 'hyperopt_tuned_clustering_algorithm',
    clustering_algorithm: clustering_algorithm,
    metric: metric,
    parameter_search_space: parameter_search_space,
    max_evals: max_evals,
    patience: patience,
    trials_per_eval: trials_per_eval,
    min_clusters: min_clusters,
    max_clusters: max_clusters,
    tpe_startup_jobs: tpe_startup_jobs,
};

tuned
