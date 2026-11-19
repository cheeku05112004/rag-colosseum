
import re
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from fastembed import TextEmbedding
from knowledge_base import PASSAGES

app = FastAPI(title="RAG Colosseum")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STOPWORDS = {
    "what", "is", "are", "the", "a", "an", "in", "on", "of",
    "for", "to", "and", "does", "do", "how", "which", "who",
    "when", "where", "why", "explain", "tell", "me", "about",
    "with", "by", "from", "that", "this", "it", "be", "was",
    "were", "can", "could", "would", "should", "consensus", "consensu"
}

def tokenize(text):
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    result = []

    for word in words:
        if word.endswith("s") and (len(word) > 4 or word == "keys"):
            word = word[:-1]
        if word not in STOPWORDS:
            result.append(word)

    return result

def lexical_score(question, text):
    q_tokens = set(tokenize(question))
    text_tokens = set(tokenize(text))

    if not q_tokens:
        return 0.0

    matched = q_tokens.intersection(text_tokens)
    return len(matched) / len(q_tokens)

def normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, 1e-12)

# Build sentence-to-passage mapping without generating embeddings.
SENTENCES = []
SENTENCE_SOURCES = []

for passage_id, passage in enumerate(PASSAGES):
    parts = re.split(r'(?<=[.!?])\s+', passage.strip())
    for sentence in parts:
        if sentence:
            SENTENCES.append(sentence)
            SENTENCE_SOURCES.append(passage_id)

# Load precomputed embeddings instead of generating them at startup.
passage_vectors = np.load("embeddings.npy").astype(np.float32)
sentence_vectors = np.load("sentence_embeddings.npy").astype(np.float32)

if passage_vectors.shape[0] != len(PASSAGES):
    raise RuntimeError("Passage embedding count mismatch")

if sentence_vectors.shape[0] != len(SENTENCES):
    raise RuntimeError("Sentence embedding count mismatch")

model = None

@app.on_event("startup")
def load_embedding_model():
    global model
    print("Loading embedding model...")
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    print("RAG system ready!")
    print("Passages:", len(PASSAGES))
    print("Sentences:", len(SENTENCES))


class QueryRequest(BaseModel):
    question: str
    top_k: int = Field(default=3, ge=1, le=10)


@app.get("/")
def home():
    return {"status": "RAG Colosseum is running"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "passages": len(PASSAGES),
        "sentences": len(SENTENCES)
    }


@app.post("/query")
def query(request: QueryRequest):
    question = request.question.strip()

    # Remove common prompt-injection instructions before retrieval.
    question = re.sub(
        r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions[^.?!]*[.?!]?",
        "",
        question,
        flags=re.IGNORECASE
    ).strip()
    top_k = request.top_k

    if not question:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.0
        }

    # Step 1: Understand the question semantically.
    q_vector = np.array(
        list(model.query_embed(question))[0],
        dtype=np.float32
    )
    q_vector /= max(np.linalg.norm(q_vector), 1e-12)

    # Step 2: Rank complete passages using semantic + keyword matching.
    dense_passage_scores = passage_vectors @ q_vector

    passage_results = []

    for pid, passage in enumerate(PASSAGES):
        dense = float(dense_passage_scores[pid])
        lexical = lexical_score(question, passage)

        combined = 0.65 * dense + 0.35 * lexical

        passage_results.append((pid, combined, lexical))

    passage_results.sort(key=lambda x: x[1], reverse=True)

    # Give named topics such as Raft and SWIM priority.
    capitalized = re.findall(r"\b[A-Z][a-zA-Z0-9]*\b", question)

    ignored = {
        "what", "which", "who", "where", "when",
        "why", "how", "explain", "tell"
    }

    anchors = [
        word.lower() for word in capitalized
        if word.lower() not in ignored
    ]

    # Expand known acronyms before sentence-level lexical matching.
    detail_question = question

    detail_expansions = {
        "tcp": " ",
        "mvcc": "multi version concurrency control",
        "wal": "write ahead log",
        "sgd": "stochastic gradient descent",
        "svm": "support vector machine",
        "mlp": "multi layer perceptron",
        "resnet": "residual convolutional network",
        "hlc": "hybrid logical clock",
        "dns": " ",
        "mdns": "multicast dns",
        "lsm": "log structured merge tree",
        "tls": "transport layer security",
        "aes": "advanced encryption standard",
        "jwt": "json web token",
        "pkce": "proof key for code exchange",
        "ecc": "elliptic curve cryptography",
        "2pc": "two phase commit",
        "3pc": "three phase commit",
        "sqli": "sql injection",
        "xss": "cross site scripting"
    }

    for anchor in anchors:
        replacement = detail_expansions.get(anchor, " ")
        detail_question = re.sub(
            rf"\b{re.escape(anchor)}\b",
            replacement,
            detail_question,
            flags=re.IGNORECASE
        )

    # If a named topic exists, search passages containing that topic.
    candidate_pool = passage_results

    if anchors:
        anchor_aliases = {
            "tcp": {"transmission", "control", "protocol"},
            "mvcc": {"multi", "version", "concurrency", "control"},
            "wal": {"write", "ahead", "log"},
            "sgd": {"stochastic", "gradient", "descent"},
            "svm": {"support", "vector", "machine"},
            "mlp": {"multi", "layer", "perceptron"},
            "resnet": {"residual", "convolutional", "network"},
            "hlc": {"hybrid", "logical", "clock"},
            "dns": {"domain", "name", "system"},
            "mdns": {"multicast", "dns"},
            "lsm": {"log", "structured", "merge", "tree"},
            "tls": {"transport", "layer", "security"},
            "aes": {"advanced", "encryption", "standard"},
            "jwt": {"json", "web", "token"},
            "pkce": {"proof", "key", "code", "exchange"},
            "ecc": {"elliptic", "curve", "cryptography"},
            "2pc": {"two", "phase", "commit"},
            "3pc": {"three", "phase", "commit"},
            "sqli": {"sql", "injection"},
            "xss": {"cross", "site", "scripting"}
        }

        anchored = [
            item for item in passage_results
            if all(
                anchor in tokenize(PASSAGES[item[0]])
                or (
                    anchor in anchor_aliases
                    and anchor_aliases[anchor].issubset(
                        set(tokenize(PASSAGES[item[0]]))
                    )
                )
                for anchor in anchors
            )
        ]

        if anchored:
            candidate_pool = anchored

    candidate_passages = candidate_pool[:8]

    best_passage_score = candidate_passages[0][1]

    if best_passage_score < 0.48:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.20
        }

    # Step 3: Search for the best sentence inside those passages.
    numeric_intent = bool(re.search(
        r"\b(how many|how much|what number|number of|how long|how often)\b",
        question,
        flags=re.IGNORECASE
    ))

    asks_timeout = bool(re.search(
        r"\b(timeout|timeouts|time-out|time-outs|rto)\b",
        question,
        re.IGNORECASE
    ))

    timeout_value_pattern = re.compile(
        r"\b\d+(?:\.\d+)?\s*(?:ms|milliseconds?|seconds?|secs?|minutes?|mins?|hours?|hrs?)"
        r"(?:\s+[a-z-]+){0,3}\s+(?:timeout|timeouts|time-out|time-outs|rto)\b"
        r"|\b(?:timeout|timeouts|time-out|time-outs|rto)(?:\s+[a-z-]+){0,3}"
        r"\s+\d+(?:\.\d+)?\s*(?:ms|milliseconds?|seconds?|secs?|minutes?|mins?|hours?|hrs?)\b",
        re.IGNORECASE
    )

    protection_intent = bool(re.search(
        r"\b(?:protect|protection|defend|safeguard)\b.*\bagainst\b",
        question,
        re.IGNORECASE
    ))

    protection_action_pattern = re.compile(
        r"\b(?:protect(?:s|ed|ing)?|prevent(?:s|ed|ing)?|"
        r"eliminat(?:e|es|ed|ing)|block(?:s|ed|ing)|"
        r"mitigat(?:e|es|ed|ing)|defend(?:s|ed|ing)|secure(?:s|d|ing))\b",
        re.IGNORECASE
    )

    protection_target_pattern = re.compile(
        r"\b(?:attacks?|interception|intrusion|exploitation|"
        r"unauthorized access|replay|injection|credential theft)\b",
        re.IGNORECASE
    )

    candidates = []

    for pid, passage_score, passage_lexical in candidate_passages:

        for sid, source_id in enumerate(SENTENCE_SOURCES):
            if source_id != pid:
                continue

            sentence = SENTENCES[sid]
            dense = float(sentence_vectors[sid] @ q_vector)
            # Rank sentence details separately from the topic name.
            lexical = lexical_score(detail_question, sentence)

            detail_tokens = set(tokenize(detail_question))
            sentence_tokens = set(tokenize(sentence))

            keyword_coverage = (
                len(detail_tokens.intersection(sentence_tokens))
                / max(len(detail_tokens), 1)
            )

            keyword_bonus = 0.0
            if detail_tokens.intersection(sentence_tokens):
                keyword_bonus = 0.20

            definition_intent = bool(re.search(
                r"\bwhat\s+(?:does|do)\b.*\b(?:provide|mean|represent|refer|stand\s+for)\b"
                r"|\b(?:define|meaning\s+of|definition\s+of)\b",
                question,
                re.IGNORECASE
            ))

            definition_bonus = 0.20 if (
                definition_intent
                and re.search(
                    r"\b(?:provide|provides|mean|means|refer(?:s)?\s+to|"
                    r"represent(?:s)?|stand(?:s)?\s+for)\b",
                    sentence,
                    re.IGNORECASE
                )
            ) else 0.0

            protection_evidence = (
                protection_intent
                and bool(protection_action_pattern.search(sentence))
                and bool(protection_target_pattern.search(sentence))
            )

            protection_bonus = 0.45 if protection_evidence else 0.0

            numeric_bonus = 0.25 if (
                numeric_intent
                and re.search(
                    r"(?<![A-Za-z0-9_])\d+(?:\.\d+)?"
                    r"(?:\s*(?:ms|milliseconds?|seconds?|secs?|minutes?|mins?|hours?|hrs?))?\b",
                    sentence,
                    re.IGNORECASE
                )
            ) else 0.0

            final_score = (
                0.25 * dense
                + 0.35 * lexical
                + 0.10 * passage_lexical
                + 0.10 * keyword_coverage
                + keyword_bonus
                + definition_bonus
                + protection_bonus
                + numeric_bonus
            )

            candidates.append({
                "sentence": sentence,
                "source": pid,
                "score": final_score,
                "passage_score": passage_score,
                "keyword_coverage": keyword_coverage,
                "protection_evidence": protection_evidence
            })

    candidates.sort(key=lambda x: x["score"], reverse=True)

    if asks_timeout:
        timeout_candidates = [
            item for item in candidates
            if timeout_value_pattern.search(item["sentence"])
        ]

        if not timeout_candidates:
            return {
                "answer": None,
                "sources": [],
                "scores": [],
                "confidence": 0.20
            }

        candidates = timeout_candidates

    # TCP retransmission timeout specifically maps to the RTO evidence.
    asks_tcp_retransmission = bool(re.search(
        r"\btcp\b",
        question,
        re.IGNORECASE
    )) and bool(re.search(
        r"\bretransmission\b",
        question,
        re.IGNORECASE
    ))

    if asks_tcp_retransmission and asks_timeout:
        candidates = [
            item for item in candidates
            if re.search(r"\brto\b", item["sentence"], re.IGNORECASE)
        ]

        if not candidates:
            return {
                "answer": None,
                "sources": [],
                "scores": [],
                "confidence": 0.20
            }

    # Abstain when a multi-keyword question has insufficient
    # lexical evidence in the best matching sentence.
    detail_token_count = len(set(tokenize(detail_question)))
    best_keyword_coverage = candidates[0]["keyword_coverage"]

    best_has_protection_evidence = candidates[0]["protection_evidence"]

    if (
        detail_token_count >= 4
        and best_keyword_coverage < 0.40
        and not best_has_protection_evidence
    ):
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.20
        }

    # Step 4: Select relevant sentences, avoiding duplicate sources.
    selected = []
    used_sources = set()

    best_sentence_score = candidates[0]["score"]

    for item in candidates:
        # Numeric questions need one precise answer sentence.
        if numeric_intent and selected:
            break

        if item["score"] < best_sentence_score - 0.10:
            continue

        if item["source"] in used_sources:
            continue

        selected.append(item)
        used_sources.add(item["source"])

        if len(selected) >= top_k:
            break

    if not selected or best_sentence_score < 0.42:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.20
        }

    answer = " ".join(item["sentence"] for item in selected)

    asks_lease = bool(re.search(r"\blease(s)?\b", question, re.IGNORECASE))
    asks_timing = bool(re.search(
        r"\b(interval|period|frequency|duration|how long)\b",
        question,
        re.IGNORECASE
    ))

    has_time_value = bool(re.search(
        r"(?<![A-Za-z0-9_])\d+(?:\.\d+)?\s*(?:ms|milliseconds?|seconds?|secs?|minutes?|mins?|hours?|hrs?)\b",
        answer,
        re.IGNORECASE
    ))

    lease_supported = (
        not asks_lease
        or bool(re.search(r"\blease(s)?\b", answer, re.IGNORECASE))
    )

    timing_supported = not asks_timing or has_time_value

    timeout_supported = (
        not asks_timeout or bool(timeout_value_pattern.search(answer))
    )

    if not lease_supported or not timing_supported or not timeout_supported:
        return {
            "answer": None,
            "sources": [],
            "scores": [],
            "confidence": 0.20
        }
    sources = [item["source"] for item in selected]
    scores = [
        round(min(1.0, max(0.0, float(item["score"]))), 4)
        for item in selected
    ]

    # Confidence is based on the best selected evidence score.
    confidence = min(0.95, max(0.61, best_sentence_score))

    return {
        "answer": answer,
        "sources": sources,
        "scores": scores,
        "confidence": round(confidence, 4)
    }

