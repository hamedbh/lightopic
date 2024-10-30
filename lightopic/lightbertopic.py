"""A simple child class of bertopic.BERTopic to enable using Lightopic.

The only addition to the base class is a method to `save_lightopic`:
this will serialise a model in a format suitable for instantiating a
Lightopic model with Lightopic.load. As part of this process it will
train a reduced UMAP model, suitable for projecting down to two
dimensions for plotting.

Typical usage example:
```python
from lightopic import LightBERTopic
from sklearn.datasets import fetch_20newsgroups

docs = fetch_20newsgroups(subset="all")["data"]
topic_model = LightBERTopic()
topics, probabilities = topic_model.fit_transform(docs)
topic_model.save_lightopic("lightopic_model")
```
"""

import logging
import os
from pathlib import Path

from bertopic import BERTopic
from joblib import dump
from umap import UMAP


class LightBERTopic(BERTopic):
    """Child class of bertopic.BERTopic to enable serialising models in
    the Lightopic format.

    All methods from bertopic.BERTopic are left unchanged; we add only
    the save_lightopic method.
    """

    def save_lightopic(self, output_path: str) -> None:
        """Serialise a trained model for use as a Lightopic model.

        Args:
            output_path (str): Directory path to serialise the model,
                will be created if it does not exist.
        """
        logger = logging.getLogger(__name__)
        save_directory = Path(output_path)
        save_directory.mkdir(exist_ok=True, parents=True)
        logger.info("Training reduced UMAP model")
        # TODO: handle other methods of training the reduced umap model.
        reduced_umap = UMAP(
            n_neighbors=10,
            n_components=2,
            min_dist=0.0,
            metric="cosine",
        ).fit(self.umap_model.embedding_)
        logger.info("Finished training reduced UMAP model")
        logger.info(f"Serialising LightBERTopic to {save_directory}")
        dump(
            self.umap_model,
            os.path.join(save_directory, "umap_model.joblib"),
        )
        dump(
            reduced_umap,
            os.path.join(save_directory, "reduced_umap_model.joblib"),
        )
        dump(
            self.hdbscan_model,
            os.path.join(save_directory, "hdbscan_model.joblib"),
        )
        logger.info(
            f"Finished serialising LightBERTopic model to {save_directory}"
        )
