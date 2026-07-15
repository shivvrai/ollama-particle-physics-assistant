import ollama

def cosine_similarity(a, b):
    """Computes the cosine similarity between two vectors."""
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x ** 2 for x in a]) ** 0.5
    norm_b = sum([x ** 2 for x in b]) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)

## Using ollama's embedding API to create vector representations of the text chunks 
EMBEDDING_MODEL = 'nomic-embed-text'
LANGUAGE_MODEL = 'llama3.2:1b'
DATASET_PATH = 'physics-facts.txt'

# Each element in the VECTOR_DB will be a tuple (chunk, embedding)
VECTOR_DB = []

def load_and_embed():
    """Loads the physics facts dataset and computes embeddings for each chunk."""
    try:
        with open(DATASET_PATH, 'r', encoding="utf-8") as file:
            dataset = file.readlines()
            
        print(f"Loading {len(dataset)} facts from {DATASET_PATH}...")
        for i, chunk in enumerate(dataset):
            chunk = chunk.strip()
            if not chunk:
                continue
            
            print(f"Generating embedding for chunk {i+1}/{len(dataset)}...")
            # ollama.embed returns a dict like {'embeddings': [[...]]}
            response = ollama.embed(model=EMBEDDING_MODEL, input=chunk)
            embedding = response['embeddings'][0]
            
            VECTOR_DB.append((chunk, embedding))
        print("Vector Database initialization complete!")
    except FileNotFoundError:
        print(f"Error: Could not find {DATASET_PATH}. Please make sure it exists.")
        exit(1)


def retrieve(query, top_k=3):
    """Finds the top_k most relevant chunks for a given query."""
    # Embed the user query
    query_response = ollama.embed(model=EMBEDDING_MODEL, input=query)
    query_embedding = query_response['embeddings'][0]
    
    similarities = []
    # Compare query embedding with all stored chunk embeddings
    for chunk, embedding in VECTOR_DB:
        similarity = cosine_similarity(query_embedding, embedding)
        similarities.append((chunk, similarity))
        
    # Sort by similarity score in descending order
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]

def main():
    load_and_embed()
    
    print("\n" + "="*50)
    print("Welcome to the Particle Physics RAG Application!")
    print("Type your question below. Type 'exit' or 'quit' to close.")
    print("="*50 + "\n")
    
    while True:
        prompt = input("Ask a question: ")
        if prompt.lower() in ['exit', 'quit']:
            break
            
        # 1. Retrieve relevant facts
        results = retrieve(prompt)
        print("\n--- Retrieved Facts ---")
        for chunk, score in results:
            print(f"[{score:.3f}] {chunk}")
            
        # Combine the retrieved chunks into a single context string
        context = '\n'.join([chunk for chunk, score in results])
        
        # 2. Build the system prompt
        system_prompt = f"""You are a helpful assistant that answers questions about particle physics using ONLY the context below.
If the context does not contain the answer, say "I don't have enough information to answer that."

Context:
{context}"""
        
        # 3. Generate response using the local LLM
        print("\n--- Generating Answer ---")
        try:
            response = ollama.chat(
                model=LANGUAGE_MODEL,
                messages=[
                    {'role': 'system', 'content': system_prompt},
                    {'role': 'user', 'content': prompt},
                ],
                options={'num_ctx': 2048, 'num_thread': 4}
            )
            print(f"AI: {response['message']['content']}\n")
        except Exception as e:
            print(f"Error generating response: {e}\n")

if __name__ == "__main__":
    main()
