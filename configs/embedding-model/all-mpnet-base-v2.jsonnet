{
    type: 'caching_sentence_embedding_model',
    sentence_embedding_model: {
        type: 'sentence_transformers_model',
        model_name_or_path: 'sentence-transformers/all-mpnet-base-v2',
    },
    cache_path: 'cache',
    prefix: 'all-mpnet-base-v2',
}
