"""The lightopic package helps make BERTopic model deployments smaller
and lighter. When installed without any of the optional extras,
lightopic provides methods for using a trained and serialised model
to `transform` new documents (i.e. label them with a topic).

You can also install with the bertopic package to train your model.
"""

from lightopic._lightopic import Lightopic

__all__ = [
    "Lightopic",
]
