# Lightopic

This package addresses a specific use case:

1. You have trained a [BERTopic](https://maartengr.github.io/BERTopic/index.html) model.
2. You want to use that trained model for transforming new data, e.g. via an API.

This came up for me because I wanted to deploy such a model API but wanted to make the deployment smaller and faster. The BERTopic package is broad, which brings with it a load of dependencies (e.g. torch, a bunch of cuda libraries). So I wrote this as a way to do the `transform` step only, with a virtual environment that's about 95% smaller than one with the actual BERTopic package.

The main external dependency is that you need to have trained a BERTopic model separately and have serialised it in a way that's compatible with `lightopic`. There is guidance on how to do that below. From that point it should be easy enough to instantiate a `Lightopic` object and use its `transform` method on new data.
