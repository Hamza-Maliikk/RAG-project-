import os
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from huggingface_hub import InferenceClient

load_dotenv()

HF_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")

# Notebook wala hi model (chunks isi se embed hue thay, isliye same hona zaroori hai)
EMBED_MODEL = "sentence-transformers/all-mpnet-base-v2"
# Jawab likhne wala model, chahein to badal sakte hain
LLM_MODEL = "moonshotai/Kimi-K2-Instruct"
COLLECTION = "Document"

# ---- setup: sirf ek baar, server start hone par ----
embeddings = HuggingFaceEndpointEmbeddings(
    model=EMBED_MODEL,
    task="feature-extraction",
    huggingfacehub_api_token=HF_TOKEN,
)

weaviate_client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
)
collection = weaviate_client.collections.get(COLLECTION)

llm = InferenceClient(api_key=HF_TOKEN)


def retrieve(question: str, k: int = 5) -> list[str]:
    query_vector = embeddings.embed_query(question)
    results = collection.query.near_vector(near_vector=query_vector, limit=k)
    return [obj.properties["content"] for obj in results.objects]


def rag_answer(question: str) -> str:
    chunks = retrieve(question)
    if not chunks:
        return "Mujhe documents mein is sawal ka jawab nahi mila."

    context = "\n\n---\n\n".join(chunks)

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Answer the question using ONLY "
                "the provided context. If the answer is not in the context, "
                "say you don't know. Reply in the same language as the question."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        },
    ]

    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=512,
        temperature=0.2,
    )
    return response.choices[0].message.content