# Lightopic

This package addresses a specific use case:

1. You have trained a [BERTopic](https://maartengr.github.io/BERTopic/index.html) model.
2. You want to use that trained model for transforming new data, e.g. via an API.

This came up for me because I wanted to deploy such a model API but wanted to make the deployment smaller and faster. The BERTopic package is broad, which brings with it a load of dependencies (e.g. torch, a bunch of cuda libraries). So I wrote this as a way to do the `transform` step only, with a virtual environment that's about 95% smaller than one with the actual BERTopic package.

The main external dependency is that you need to have trained a BERTopic model separately and have serialised it in a way that's compatible with `lightopic`. There is guidance on how to do that below. From that point it should be easy enough to instantiate a `Lightopic` object and use its `transform` method on new data.

## Serialising your `BERTopic` model

This package uses only the `umap_model` and `hdbscan_model` attributes of the `BERTopic` model, so that's what you need to save. The code below extends the basic topic model example from the `BERTopic` docs with a function that will serialise your model in the required format. **NOTE**: this package is still under development, so this required format may (and probably will) change!

```python
from bertopic import BERTopic
from sklearn.datasets import fetch_20newsgroups
from pathlib import Path
from joblib import dump
from umap import UMAP
import os
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

topic_model = BERTopic()
topics, probs = topic_model.fit_transform(docs)
def save_lightopic(topic_model: BERTopic, save_directory: str) -> None:
    save_directory = Path(save_directory)
    save_directory.mkdir(exist_ok=True, parents=True)
    dump(topic_model.umap_model, os.path.join(save_directory, "umap_model.joblib"))
    reduced_umap = UMAP(
        n_neighbors=10,
        n_components=2,
        min_dist=0.0,
        metric="cosine",
    ).fit(topic_model.umap_model.embedding_)
    dump(topic_model.hdbscan_model, os.path.join(save_directory, "hdbscan_model.joblib"))

save_lightopic(topic_model, "model_directory")
```
