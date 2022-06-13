local default_settings = {
    class_weight: 'balanced'
};

local logistic_regression_classifier_evaluator(settings = default_settings) = {
    type: 'logistic_regression_classifier_evaluator',
    sentence_encoder: import '../embedding-model/all-mpnet-base-v2.jsonnet',
    classifier_settings: settings
};

logistic_regression_classifier_evaluator