import os

import hdbscan
import joblib
import numpy


class Lightopic:
    def __init__(self) -> None:
        pass

    def load(self, model_directory: str) -> None:
        self.umap_model = joblib.load(
            os.path.join(model_directory, "umap_model.joblib")
        )
        self.hdbscan_model = joblib.load(
            os.path.join(model_directory, "hdbscan_model.joblib")
        )
        full_raw_data_shape = self.umap_model._raw_data.shape
        self.len_full_embedding = full_raw_data_shape[1]
        self.len_umap_embedding = self.umap_model.n_components

    def reduce_embeddings(self, embeddings: numpy.ndarray) -> numpy.ndarray:
        input_dimension = embeddings.shape[1]
        if input_dimension != self.len_full_embedding:
            raise ValueError(
                "Dimension mismatch: embeddings have "
                f"{input_dimension} dimensions, expected "
                f"{self.len_full_embedding} dimensions"
            )
        return self.umap_model.transform(embeddings)

    def transform(
        self, embeddings: numpy.ndarray, calculate_probabilities: bool = True
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        input_dimension = embeddings.shape[1]
        if input_dimension != self.len_full_embedding:
            raise ValueError(
                "Dimension mismatch: embeddings have "
                f"{input_dimension} dimensions, expected "
                f"{self.len_full_embedding} dimensions"
            )
        umap_embeddings = self.reduce_embeddings(embeddings)
        predictions, probabilities = hdbscan.approximate_predict(
            self.hdbscan_model, umap_embeddings
        )
        if calculate_probabilities:
            probabilities = hdbscan.membership_vector(
                self.hdbscan_model, umap_embeddings
            )
        return predictions, probabilities
