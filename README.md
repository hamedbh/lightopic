# Lightopic

This package addresses the specific use case of deploying a [BERTopic](https://maartengr.github.io/BERTopic/index.html) model that you've trained, and now want to use for transforming new data, e.g. via an API.

This came up for me because I wanted to deploy such a model API but wanted to make the deployment smaller and faster. The BERTopic package is broad, which brings with it a load of dependencies (e.g. torch, a bunch of cuda libraries). So I wrote this as a way to do the `transform` step only, with a virtual environment that's about 95% smaller than one with the actual BERTopic package.

The main prerequisite is that you need to have trained a BERTopic model separately and have serialised it in a way that's compatible with `lightopic`. The `lightopic` package also offers you a way to do that: guidance on how is below. From that point you can instantiate a `Lightopic` object and use its `transform` method on new data.

## Training and serialising your `LightBERTopic` model

This is a necessary step: you can't instantiate a `Lightopic` object without first having trained and serialised your model. To make this part easier the `LightBERTopic` class is available: this is a child class of `bertopic.BERTopic`, only with a method added to `save_lightopic`.
```python
from lightopic.lightbertopic import LightBERTopic
docs = fetch_20newsgroups(subset='all',  remove=('headers', 'footers', 'quotes'))['data']

topic_model = LightBERTopic()
topics, probs = topic_model.fit_transform(docs)
topic_model.save_lightopic("model_directory")
```

NB. for this to work you must have `bertopic` installed, which you can do with `pip install lightopic[bertopic]`.

**NOTE**: this package is still under development, so this required format may (and probably will) change!

## Using a `Lightopic` model

Now the serialised model is ready to use.

```python
from lightopic import Lightopic
topic_model = Lightopic()
topic_model.load("model_directory")
topic_model.transform(embeddings)
```

This transform step does not rely on BERTopic at all, so it can use the smaller installation you get from `pip install lightopic`.
