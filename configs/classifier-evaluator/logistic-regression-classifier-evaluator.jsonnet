local default_settings = {
    class_weight: 'balanced'
};

local default_embedding_model = import '../embedding-model/all-mpnet-base-v2.jsonnet';
local logistic_regression_classifier_evaluator(
    settings = default_settings,
    embedding_model = default_embedding_model,
) = {
    type: 'logistic_regression_classifier_evaluator',
    sentence_encoder: embedding_model,
    classifier_settings: settings
};

logistic_regression_classifier_evaluator