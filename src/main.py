from src.input_processing import speech_to_text, image_to_text, handle_text_input
from src.rag_retriever import retrieve_context
from src.generator import generate_response
from src.memory_manager import retrieve_memory_context, save_to_memory


def multimodal_pipeline(input_type, input_data, return_context: bool = False):
# handle text input
    if input_type == "text":
        query = handle_text_input(input_data)

# handle audio input
    elif input_type == "audio":
        query, success = speech_to_text(input_data)
        if not success:
            return "audio processing failed.", "", ""

# handle image input
    elif input_type == "image":
        query = image_to_text(input_data).strip()
        query_lower = query.lower()

        real_world_terms = ["person", "animal", "nature", "landscape"]
        technical_terms = ["diagram", "chart", "graph", "code", "flowchart", "figure", "table","equation", 
        "block diagram", "box", "label", "variable", "arrow","data flow", "schema", "structure"]

# detect real-world images
        if any(term in query_lower for term in real_world_terms):
            answer = f"{query_lower} is likely a real-world image, not a technical or textbook diagram."
            if return_context:
                return query, answer, "no relevant context for this image."
            save_to_memory(query, answer)
            return query, answer, "no relevant context for this image."

# detect technical images and allow RAG
        elif any(term in query_lower for term in technical_terms) or "diagram" in query_lower:
            print("technical image detected → retrieving context")
            # continue below for retrieval and response

        else:

            answer = f"{query_lower} appears to be an unclear"
            if return_context:
                return query, answer, "no relevant context for this image."
            save_to_memory(query, answer)
            return query, answer, "no relevant context for this image."

    else:
        raise ValueError(f"invalid input")

    memory_context = retrieve_memory_context(query)
    rag_context = retrieve_context(query)
    combined_context = f"{memory_context}\n{rag_context}".strip()


    if return_context:
        return query, "", rag_context

    answer = generate_response(query, combined_context)

    save_to_memory(query, answer)

    return query, answer, rag_context
