from haystack.document_stores import ElasticsearchDocumentStore, \
  OpenSearchDocumentStore
from haystack.nodes import EmbeddingRetriever

document_store_1 = ElasticsearchDocumentStore(
    embedding_dim=128,
    index="store_1",
    similarity="dot_product",
    host="localhost",
    port=9200
)
retriever_1 = EmbeddingRetriever(
    document_store_1, embedding_model="yjernite/retribert-base-uncased",
    model_format="retribert")
dicts_1 = [
  {
    'content': "Document text one alpha beta gamma",
  },
  {
    'content': "Document text two alpha beta gamma",
  }
]
document_store_1.delete_documents()
document_store_1.write_documents(dicts_1)
document_store_1.update_embeddings(retriever_1, batch_size=10)

document_store_2 = ElasticsearchDocumentStore(
    embedding_dim=128,
    index="store_2",
    similarity="dot_product",
    host="localhost",
    port=9200
)
retriever_2 = EmbeddingRetriever(
    document_store_2, embedding_model="yjernite/retribert-base-uncased",
    model_format="retribert")
dicts_2 = [
  {
    'content': "Document text one alpha beta gamma",
  },
  {
    'content': "Document text two alpha beta gamma",
  }
]
document_store_2.delete_documents()
document_store_2.write_documents(dicts_2)
document_store_2.update_embeddings(retriever_2, batch_size=10)

documents = retriever_1.retrieve(query="Alpha Beta Gamma Delta")
for doc in documents:
  print(doc, doc.score)
documents = retriever_2.retrieve(query="Alpha Beta Gamma Delta")
for doc in documents:
  print(doc, doc.score)
