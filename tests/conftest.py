import pytest
from sentence_transformers import SentenceTransformer
from sklearn.datasets import fetch_20newsgroups

from lightopic.lightbertopic import LightBERTopic


@pytest.fixture(scope="session")
def embedding_model():
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model


@pytest.fixture(scope="session")
def document_embeddings(documents, embedding_model):
    embeddings = embedding_model.encode(documents)
    return embeddings


@pytest.fixture(scope="session")
def documents():
    newsgroup_docs = fetch_20newsgroups(
        subset="all", remove=("headers", "footers", "quotes")
    )["data"][:1000]
    return newsgroup_docs


@pytest.fixture(scope="session")
def base_lightbertopic_model(documents, document_embeddings, embedding_model):
    model = LightBERTopic(
        embedding_model=embedding_model, calculate_probabilities=True
    )
    model.umap_model.random_state = 42
    model.hdbscan_model.min_cluster_size = 3
    model.fit(documents, document_embeddings)
    return model
