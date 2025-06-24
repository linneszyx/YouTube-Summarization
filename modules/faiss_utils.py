from langchain_community.vectorstores import FAISS

def create_faiss_index(chunks, embedding_model):
    return FAISS.from_texts(chunks, embedding_model)

def perform_similarity_search(faiss_index, query, k=3):
    return faiss_index.similarity_search(query, k=k)

def retrieve(query, faiss_index, k=7):
    return faiss_index.similarity_search(query, k=k)
