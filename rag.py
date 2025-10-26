import numpy as np
import pandas as pd
import faiss
import ollama

# Load the Faiss index
def load_faiss_index(index_path="anaimal_facts.index"):
    try:
        index = faiss.read_index(index_path)
        return index
    except Exception as e:
        print(f"Failed to load Faiss index: {e}")
        return None
    
# Generate embedding for a query
def generate_query_embedding(query):
    try:
        response = ollama.embeddings(model="embeddinggemma", prompt=query)
        return np.array(response['embedding']).astype('float32').reshape(1, -1)
    except Exception as e:
        print(f"Failed to generate query embedding: {e}")
        return None
    
# Retrieve relevant documents
def retrieve_documents(index, query_embedding, k=15):
    try:
        D, I = index.search(query_embedding, k)
        return I[0], D[0]
    except Exception as e:
        print(f"Failed to retrieve documents: {e}")
        return None, None
    

    # Generate response using the generator model
def generate_response(query, retrieved_documents, df):
    try:
        #print(retrieved_documents)
        len(retrieved_documents)
        context = ""
        for doc_id in retrieved_documents:
            # print("Fact retrieved: ", get_document_text(doc_id, df))
            context += get_document_text(doc_id, df) + "\n"
        # print("\n")
        prompt = f"{context}\n Using only the context before, answer the following query! {query}"
        # print("Prompt: ", prompt)
        # print("\n")
        response = ollama.generate(model="smollm:360m", prompt=prompt)
        return response['response']
    except Exception as e:
        print(f"Failed to generate response: {e}")
        return None
    

# Main function
def rag(query, index_path, df):
    index = load_faiss_index()
    if index is None:
        return
    
    query_embedding = generate_query_embedding(query)
    if query_embedding is None:
        return
    
    retrieved_documents, distances = retrieve_documents(index, query_embedding)
    if retrieved_documents is None:
        return
    
    response = generate_response(query, retrieved_documents, df)
    return response

def get_document_text(doc_id, df):
    return df.iloc[doc_id]['animal_fact']







if __name__ == "__main__":
    # Load your data and start an input loop
    CSV_PATH = "AnimalFacts.csv"        # change if needed
    INDEX_PATH = "animal_facts.index"   # change if needed

    try:
        df = pd.read_csv(CSV_PATH)
        if "animal_fact" not in df.columns:
            raise ValueError("Expected column 'animal_fact' not found in CSV.")
        df = df[["animal_fact"]]
        print(f"Loaded CSV: {CSV_PATH} (rows: {len(df)})")
    except Exception as e:
        print(f"Failed to load CSV: {e}")
        raise SystemExit(1)

    print("RAG ready. Ask about animals! Type 'stop' to exit.\n")

    while True:
        q = input("Q> ").strip()
        if q.lower() in {"stop", "exit", "quit"}:
            print("Bye!")
            break
        if not q:
            continue

        answer = rag(q, INDEX_PATH, df)
        if answer is None:
            print("Sorry, something went wrong answering that.\n")
        else:
            print("\nA>", answer, "\n")
