import ollama
import pandas as pd
import numpy as np
import faiss

import time

print("Test")

def embed_text(text, model="embeddinggemma"):
    response = ollama.embeddings(model=model, prompt=text)
    return response['embedding']


def main():
    df = pd.read_csv("AnimalFacts.csv")
    print("Data loaded")
    #Embedding
    embeddings = []
    start_time = time.time()
    for text in df["animal_fact"]:
        embeddings.append(embed_text(text))
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")


    embedding_array = np.array(embeddings).astype('float32')


    #Create and save Faiss Vector DB
    index = faiss.IndexFlatL2(embedding_array.shape[1])
    index.add(embedding_array)
    faiss.write_index(index, "anaimal_facts.index")


    # print(np.shape(embedding_array))
    # print(np.shape(embedding_array[0]))
    # print(embedding_array[0])

main()