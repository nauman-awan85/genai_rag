import ollama


model_name = "llama3"
def generate_response(prompt: str, context: str = "") -> str:
    full_prompt = f"Context:\n{context}\n\nQuestion:\n{prompt}\n\nAnswer:"

    response = ollama.chat(
        model=model_name,
        messages=[{"role": "user", "content": full_prompt}],
    )

    return response["message"]["content"].strip()
