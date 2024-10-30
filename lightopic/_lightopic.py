import os

import hdbscan
import joblib
import numpy


class Lightopic:
    """The Lightopic class is to be used with a trained topic model
    that has been serialised appropriately (see docs for
    `lightopic.LightBERTopic.save_lightopic`). Methods will transform
    new data by reducing the embeddings or putting them into topics.
    """

    def __init__(self) -> None:
        pass

    def load(self, model_directory: str) -> None:
        self.umap_model = joblib.load(
            os.path.join(model_directory, "umap_model.joblib")
        )
        self.reduced_umap_model = joblib.load(
            os.path.join(model_directory, "reduced_umap_model.loblib")
        )
        self.hdbscan_model = joblib.load(
            os.path.join(model_directory, "hdbscan_model.joblib")
        )
        full_raw_data_shape = self.umap_model._raw_data.shape
        self.len_full_embedding = full_raw_data_shape[1]
        self.len_umap_embedding = self.umap_model.n_components

    def reduce_embeddings(
        self, embeddings: numpy.ndarray, output_type: str = "reduced"
    ) -> numpy.ndarray:
        """Reduce the dimensions of an input array of embeddings.

        Args:
            embeddings (numpy.ndarray): An array of embeddings.
            output_type (str, optional): Set the dimensions of the
                output. Defaults to "reduced", which will reduce to the
                dimensions used for clustering. (If you used the
                BERTopic defaults this will be 5.) Setting this to "2d"
                will output 2-dimensional vectors that you can use for
                plotting.

        Raises:
            ValueError: if `output_type` is not either `"reduced"` or
                `"2d"`.
            ValueError: the dimension of the `embeddings` must be
                compatible with the Lightopic model, i.e. it must match
                either the full dimension of the original embeddings
                (`self.len_full_embedding`) or that of the reduced
                embeddings (`self.len_umap_embedding`).

        Returns:
            numpy.ndarray: The reduced embeddings.
        """
        legal_output_types = ["reduced", "2d"]
        if output_type not in legal_output_types:
            raise ValueError(
                "Illegal value for output_type, must be one of "
                f"{legal_output_types}"
            )
        input_dimension = embeddings.shape[1]
        if input_dimension == self.len_full_embedding:
            if output_type == "reduced":
                return self.umap_model.transform(embeddings)
            elif output_type == "2d":
                return self.reduced_umap_model.transform(embeddings)
        elif input_dimension == self.len_umap_embedding:
            if output_type == "reduced":
                return self.reduced_umap_model.transform(
                    self.umap_model.transform(embeddings)
                )
            elif output_type == "2d":
                return embeddings
        else:
            raise ValueError(
                "Dimension mismatch: embeddings have "
                f"{input_dimension} dimensions. Expected "
                f"{self.len_full_embedding} or "
                f"{self.len_umap_embedding} dimensions"
            )

    def transform(
        self, embeddings: numpy.ndarray, calculate_probabilities: bool = True
    ) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Transform new documents with the model.

        Args:
            embeddings (numpy.ndarray): Array of embeddings
            calculate_probabilities (bool, optional): Return
                probabilities for all topics? Defaults to True.

        Raises:
            ValueError: the dimension of the `embeddings` must be
                compatible with the Lightopic model, i.e. it must match
                either the full dimension of the original embeddings
                (`self.len_full_embedding`) or that of the reduced
                embeddings (`self.len_umap_embedding`).

        Returns:
            tuple[numpy.ndarray, numpy.ndarray]: The topic assignments
                and probabilities.
        """
        input_dimension = embeddings.shape[1]
        if input_dimension == self.len_full_embedding:
            umap_embeddings = self.reduce_embeddings(embeddings)
        elif input_dimension == self.len_umap_embedding:
            umap_embeddings = embeddings
        else:
            raise ValueError(
                "Dimension mismatch: embeddings have "
                f"{input_dimension} dimensions. Expected "
                f"{self.len_full_embedding} or "
                f"{self.len_umap_embedding} dimensions"
            )
        predictions, probabilities = hdbscan.approximate_predict(
            self.hdbscan_model, umap_embeddings
        )
        if calculate_probabilities:
            probabilities = hdbscan.membership_vector(
                self.hdbscan_model, umap_embeddings
            )
        return predictions, probabilities
