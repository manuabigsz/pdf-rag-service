import os
from typing import List
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def generate_answer(question: str, context_chunks: List[str]) -> str:
    context = "\n\n".join(
        [f"[Context {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)]
    )

    prompt = f"""You are a helpful assistant that answers questions based on the provided context.

    Context:
    {context}

    Question: {question}

    Instructions:
    - Answer the question based ONLY on the information provided in the context above
    - If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question based on the provided documents."
    - Be concise and specific
    - Quote relevant parts from the context when applicable

    Answer:"""

    return _generate_with_openai(prompt)


def _generate_with_openai(prompt: str) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)

    response = client.chat.completions.create(
        model="gpt-4-turbo-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1024,
        temperature=0.3,
    )

    return response.choices[0].message.content
