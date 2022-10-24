from typing import List

import numpy as np
from allennlp.common import Registrable
from sentence_transformers import SentenceTransformer


class SentenceEmbeddingModel(Registrable):
    def encode(self, utterances: List[str]) -> np.ndarray:
        """
        Encode a list of utterances as an array of real-valued vectors.
        :param utterances: original utterances
        :return: output encoding
        """
        raise NotImplementedError


@SentenceEmbeddingModel.register('sentence_transformers_model')
class SentenceTransformersModel(SentenceEmbeddingModel):

    def __init__(self, model_name_or_path: str) -> None:
        """
        Initialize SentenceTransformers model for a given path or model name.
        :param model_name_or_path: model name or path for SentenceTransformers sentence encoder
        """
        super().__init__()
        self._sentence_transformer = model_name_or_path

    def encode(self, utterances: List[str]) -> np.ndarray:
        encoder = SentenceTransformer(self._sentence_transformer)
        return encoder.encode(utterances)
