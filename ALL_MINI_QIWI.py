import pandas as pd
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModel, pipeline
import torch

# Load the CSV file
csv_file = "C:\\Users\\lakmg\\tutorial\\K6_Scripts.csv"  # Replace with your CSV file path
df = pd.read_csv(csv_file)

# Ensure columns INST and RESP are present
if not {'INST', 'RESP'}.issubset(df.columns):
    raise ValueError("The CSV file must contain 'INST' and 'RESP' columns.")

# Load embedding model
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"  # Example embedding model
tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
embedding_model = AutoModel.from_pretrained(embedding_model_name)

def get_embeddings(texts):
    """Generate embeddings for a list of texts."""
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = embedding_model(**inputs).last_hidden_state.mean(dim=1).numpy()
    return embeddings

# Generate embeddings for the INST column
inst_texts = df['INST'].tolist()
embeddings = get_embeddings(inst_texts)

# Initialize FAISS index and add embeddings
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)

# Map FAISS index to RESP for retrieval
responses = df['RESP'].tolist()

# Load Qwen model
llm_model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
llm_pipeline = pipeline("text-generation", model=llm_model_name, tokenizer=llm_model_name)

def query_faiss_and_generate_response(user_input):
    """Query FAISS index and generate response using LLM."""
    # Embed user input
    user_embedding = get_embeddings([user_input])[0].reshape(1, -1)
    
    # Search FAISS index
    _, indices = faiss_index.search(user_embedding, 1)
    closest_response = responses[indices[0][0]]
    
    # Generate response with LLM
    prompt = f"User input: {user_input}\nClosest response from data: {closest_response}\nGenerate a refined response:"
    llm_response = llm_pipeline(prompt, max_new_tokens=200, num_return_sequences=1)
    
    return llm_response[0]['generated_text']

# Example usage
while True:
    user_input = input("Enter your query(or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    response = query_faiss_and_generate_response(user_input)
    print("Generated Response:", response)
