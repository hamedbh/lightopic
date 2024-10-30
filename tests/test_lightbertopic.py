from lightopic.lightbertopic import LightBERTopic


def test_lightbertopic_model_exists(base_lightbertopic_model):
    assert isinstance(base_lightbertopic_model, LightBERTopic)
