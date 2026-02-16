
from langchain_core.prompts import ChatPromptTemplate
from retriever import retriever

def format_context(retrieved_chunks):
    formatted_chunks = []

    for i, chunk in enumerate(retrieved_chunks):
        formatted = (
            f"[Document {i+1} | Source: {chunk.get('source', 'Unknown')}]\n"
            f"{chunk.get('content', '')}"
        )
        formatted_chunks.append(formatted)

    return "\n\n".join(formatted_chunks)


rag_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a professional medical assistant AI.

Use ONLY the provided context to answer the question.
If the answer is not in the context, say: "I could not find the answer in the provided medical documents." except for the greetings

Always keep the answer clear, concise, and medically accurate.
If possible, mention the document source."""
    ),
    (
        "user",
        """Context:
{context}

Question:
{question}

Answer:"""
    )
])