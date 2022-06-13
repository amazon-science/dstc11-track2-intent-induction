
local precomputed_distances_wrapper(
    clustering_algorithm,
    metric = 'euclidean',
    normalized = false,
    distance_kwargs = {}
) = {
    type: 'precomputed_similarities_clustering_wrapper',
    clustering_algorithm: clustering_algorithm,
    metric: metric,
    normalized: normalized,
    distance_kwargs: distance_kwargs
};

precomputed_distances_wrapper