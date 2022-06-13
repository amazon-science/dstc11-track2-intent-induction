local tuned = import 'clustering-algorithm/hyperopt-tuned-clustering-algorithm.jsonnet';
local label_propagated = import 'clustering-algorithm/label-propagated-clustering-algorithm.jsonnet';
local precomputed = import 'clustering-algorithm/precomputed-distances-wrapper.jsonnet';

local kmeans = import 'clustering-algorithm/kmeans.jsonnet';

local tuned_kmeans = tuned(
    clustering_algorithm = kmeans(),
    parameter_search_space = {
        n_clusters: ['quniform', 5, 50, 1]
    },
    // k-means results may differ slightly by seed, so take average score over 3 trials
    trials_per_eval = 3,
    // number of trials without improvement before early sotpping
    patience = 25,
);

{
    kmeans: {name: 'kmeans', model: tuned_kmeans},
}