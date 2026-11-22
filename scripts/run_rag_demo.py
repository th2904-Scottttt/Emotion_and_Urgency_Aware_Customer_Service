"""Run one retrieval-augmented customer-service query."""
import argparse
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def conversations(paths):
    texts = []
    for path in paths:
        groups = {}
        with path.open(encoding="utf-8-sig", newline="") as stream:
            rows = csv.DictReader(stream)
            required = {"conversationId", "authorRole", "utterance"}
            if not required.issubset(rows.fieldnames or []):
                raise ValueError(f"{path} needs columns {sorted(required)}")
            for row in rows:
                message = (row["utterance"] or "").strip()
                if message and row["conversationId"]:
                    role = "Customer" if row["authorRole"] == "customer" else "Agent"
                    groups.setdefault(row["conversationId"], []).append(f"{role}: {message}")
        texts.extend("\n".join(groups[key]) for key in sorted(groups)[:200])
    if not texts:
        raise ValueError("The supplied CSV files contain no conversation text.")
    return texts


def retrieve(question, texts):
    import chromadb
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(EMBEDDING_MODEL)
    splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=50)
    chunks = [chunk for text in texts for chunk in splitter.split_text(text)]
    collection = chromadb.EphemeralClient().create_collection("support_conversations")
    vectors = encoder.encode(chunks)
    collection.add(ids=[str(i) for i in range(len(chunks))], documents=chunks,
                   embeddings=vectors.tolist())
    query = encoder.encode([question])[0]
    documents = collection.query(query_embeddings=[query.tolist()],
                                 n_results=min(3, len(chunks)))["documents"][0]
    # Match the notebook's cosine grading over the first 400 characters.
    snippets = encoder.encode([doc[:400] for doc in documents], normalize_embeddings=True)
    from numpy.linalg import norm
    score = float((snippets @ (query / norm(query))).max())
    return documents, score


def needs_web(question, score):
    keywords = ("weather", "today", "now", "price", "stock")
    return score < 0.4 or any(word in question.lower() for word in keywords)


TONES = {
    ("high", "angry"): "Be extremely empathetic and apologetic. Acknowledge the issue immediately and provide concrete solutions with urgency.",
    ("high", "frustrated"): "Be understanding and solution-focused. Act quickly to resolve the issue and show you're prioritizing their concern.",
    ("high", "anxious"): "Be reassuring and calm while acting swiftly. Provide clear steps and timeline to reduce their anxiety.",
    ("high", "neutral"): "Be direct, efficient, and action-oriented. Provide immediate solutions without unnecessary details.",
    ("high", "satisfied"): "Be appreciative and maintain the positive experience. Act quickly while being warm and professional.",
    ("high", "happy"): "Be enthusiastic and efficient. Match their positive energy while resolving their urgent request quickly.",
    ("medium", "angry"): "Be patient and understanding. Acknowledge their frustration and work through the solution methodically.",
    ("medium", "frustrated"): "Be empathetic and helpful. Show you understand their concern and provide clear, actionable solutions.",
    ("medium", "anxious"): "Be supportive and reassuring. Provide clear guidance and communicate each step to ease their worry.",
    ("medium", "neutral"): "Be friendly, helpful, and professional. Provide thorough assistance with a balanced approach.",
    ("medium", "satisfied"): "Be warm and attentive. Maintain their satisfaction while helping them with their request.",
    ("medium", "happy"): "Be friendly and engaging. Match their positive mood while being helpful and thorough.",
    ("low", "angry"): "Be calm, patient, and thorough. Take time to fully explain and address their concerns.",
    ("low", "frustrated"): "Be understanding and patient. Provide detailed explanations and show genuine care for resolving their issue.",
    ("low", "anxious"): "Be gentle, reassuring, and thorough. Take time to address their concerns and provide comprehensive support.",
    ("low", "neutral"): "Be professional, informative, and helpful. Provide clear and complete information.",
    ("low", "satisfied"): "Be friendly and informative. Maintain the positive relationship while helping them effectively.",
    ("low", "happy"): "Be warm, friendly, and conversational. Enjoy the positive interaction while being helpful.",
}


def generate(args, documents):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )
    model = PeftModel.from_pretrained(base, str(args.model_path))
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tone = TONES[args.urgency, args.emotion]
    context = "\n\n".join(doc[:500] for doc in documents)
    messages = [
        {"role": "system", "content": (
            f"You are a general helpful assistant. Urgency: {args.urgency}. "
            f"Emotion: {args.emotion}. {tone} Answer in 2-3 sentences using only "
            "the supplied context. If it lacks the answer, say so. Avoid company names and websites."
        )},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {args.question}"},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer)
    return generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.5,
                     repetition_penalty=1.2, return_full_text=False)[0]["generated_text"].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--question", required=True)
    parser.add_argument("--urgency", required=True, choices=["low", "medium", "high"])
    parser.add_argument("--emotion", required=True, choices=sorted({e for _, e in TONES}))
    parser.add_argument("--model-path", type=Path, default=ROOT / "CAI_Pipline/qwen2.5-cai-dpo-final")
    parser.add_argument("--csv", nargs="+", type=Path, default=[ROOT / "data/examples/support.csv"])
    parser.add_argument("--web-search", action="store_true",
                        help="Allow the fallback to send the question to Tavily.")
    args = parser.parse_args()
    if not args.question.strip():
        parser.error("Provide a nonempty question.")
    if not (args.model_path / "adapter_config.json").is_file():
        parser.error(f"Missing adapter_config.json in {args.model_path}")
    if args.web_search and not os.getenv("TAVILY_API_KEY"):
        parser.error("--web-search requires TAVILY_API_KEY")
    try:
        texts = conversations(args.csv)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    documents, score = retrieve(args.question, texts)
    if args.web_search and needs_web(args.question, score):
        from tavily import TavilyClient
        results = TavilyClient().search(query=args.question, max_results=3)
        documents.append("\n".join(row["content"] for row in results["results"]))
    print(generate(args, documents))


if __name__ == "__main__":
    main()
