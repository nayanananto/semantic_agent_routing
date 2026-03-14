# evaluate.py
from __future__ import annotations

from typing import Dict, List, Tuple
import pickle

import numpy as np

from agents import get_agents, id_to_name_map
from bandit import LinUCB
from dataset import generate_dataset_300, load_dataset_csv
from embedder import Embedder, EmbedderConfig
from routers import KNNRouter, MLRouter, PairwiseMLRouter
from vector_store import InMemoryVectorStore, VectorItem


def _camel_to_words(name: str) -> List[str]:
    out = []
    buf = ""
    for ch in name:
        if ch.isupper() and buf:
            out.append(buf)
            buf = ch
        else:
            buf += ch
    if buf:
        out.append(buf)
    return out


def _agent_profile_texts(agents) -> List[str]:
    texts: List[str] = []
    for agent in agents:
        name_words = _camel_to_words(agent.name.replace("Agent", ""))
        keywords = " ".join(word.lower() for word in name_words)
        name_hint = " ".join(name_words)
        texts.append(
            " | ".join(
                [
                    agent.description,
                    f"Agent name: {agent.name}. Capabilities: {agent.description}",
                    f"Use this agent when user asks about: {agent.name.replace('Agent', '').lower()} tasks.",
                    f"Keywords: {keywords}",
                    f"Task intent: {name_hint}",
                ]
            )
        )
    return texts


def build_vector_store(agents, embedder: Embedder) -> Tuple[InMemoryVectorStore, Dict[str, str]]:
    store = InMemoryVectorStore()
    items: List[VectorItem] = []
    for agent in agents:
        name_words = _camel_to_words(agent.name.replace("Agent", ""))
        keywords = " ".join(word.lower() for word in name_words)
        name_hint = " ".join(name_words)
        chunks = [
            agent.description,
            f"Agent name: {agent.name}. Capabilities: {agent.description}",
            f"Use this agent when user asks about: {agent.name.replace('Agent', '').lower()} tasks.",
            f"Keywords: {keywords}",
            f"Task intent: {name_hint}",
        ]
        embs = embedder.encode(chunks)
        for idx, (text, emb) in enumerate(zip(chunks, embs)):
            items.append(
                VectorItem(
                    item_id=f"{agent.agent_id}_c{idx}",
                    agent_id=agent.agent_id,
                    text=text,
                    embedding=emb,
                    meta={"chunk_type": f"c{idx}", "agent_name": agent.name},
                )
            )
    store.add_items(items)
    store.build_index()
    return store, id_to_name_map(agents)


def choose_pred_set(candidates: List[Tuple[str, float]], threshold: float) -> List[str]:
    if not candidates:
        return []
    chosen = [name for name, score in candidates if float(score) >= threshold]
    if not chosen:
        chosen = [candidates[0][0]]
    return chosen


def set_metrics(pred_names: List[str], gold_set: List[str]) -> Tuple[float, float, float, float, float]:
    pred = set(pred_names)
    gold = set(gold_set)
    tp = len(pred & gold)
    precision = tp / max(1, len(pred))
    recall = tp / max(1, len(gold))
    f1 = 0.0 if (precision + recall) == 0 else (2.0 * precision * recall / (precision + recall))
    jacc = tp / max(1, len(pred | gold))
    exact = 1.0 if pred == gold else 0.0
    return precision, recall, f1, jacc, exact


def evaluate(
    dataset_path: str | None,
    ml_model_path: str = "ml_router.pkl",
    alpha: float = 1.0,
    cost_lambda: float = 0.02,
    set_threshold: float = 0.5,
    dataset_size: int = 1000,
):
    agents = get_agents()
    data = load_dataset_csv(dataset_path) if dataset_path else generate_dataset_300(n_samples=dataset_size)

    with open(ml_model_path, "rb") as f:
        payload = pickle.load(f)

    test_ids = set(payload.get("test_prompt_ids", []))
    if test_ids:
        data = [ex for ex in data if ex.prompt_id in test_ids]
    else:
        print("[warn] No test_prompt_ids found in ML payload; evaluating on full dataset.")

    embedder_cfg = payload["embedder_config"] if "embedder_config" in payload else EmbedderConfig()
    embedder = Embedder(embedder_cfg)
    if payload["embedder_backend"] == "tfidf":
        embedder.backend = ("tfidf", None)
        embedder._tfidf = payload["tfidf_vectorizer"]

    print(f"[eval] ML payload embedder backend: {payload.get('embedder_backend', 'unknown')}")
    print(f"[eval] ML payload embedder model: {getattr(embedder_cfg, 'model_name', 'unknown')}")
    print(f"[eval] ML payload model type: {type(payload.get('model')).__name__}")
    if payload.get("embedder_backend") == "tfidf" and getattr(embedder_cfg, "model_name", None):
        print("[warn] ML was trained with TF-IDF (not SBERT). Retrain after installing sentence-transformers to use SBERT.")

    store, id_to_name = build_vector_store(agents, embedder)
    router_top_k = len(agents)
    knn_router = KNNRouter(agents, store, id_to_name, top_k=router_top_k)
    if payload.get("ml_mode") == "pairwise":
        profile_texts = payload.get("agent_profile_texts") or _agent_profile_texts(agents)
        ml_router = PairwiseMLRouter(payload["model"], payload["class_names"], profile_texts, embedder, top_k=router_top_k)
    else:
        ml_router = MLRouter(payload["model"], payload["class_names"], top_k=router_top_k)

    dim = embedder.encode([data[0].prompt]).shape[1]
    bandit = LinUCB(actions=[agent.name for agent in agents], dim=dim, alpha=alpha)

    stats = {
        "knn_set_prec_sum": 0.0,
        "knn_set_rec_sum": 0.0,
        "knn_set_f1_sum": 0.0,
        "knn_set_jacc_sum": 0.0,
        "knn_set_exact_sum": 0.0,
        "knn_set_size_sum": 0.0,
        "ml_set_prec_sum": 0.0,
        "ml_set_rec_sum": 0.0,
        "ml_set_f1_sum": 0.0,
        "ml_set_jacc_sum": 0.0,
        "ml_set_exact_sum": 0.0,
        "ml_set_size_sum": 0.0,
        "war_set_prec_sum": 0.0,
        "war_set_rec_sum": 0.0,
        "war_set_f1_sum": 0.0,
        "war_set_jacc_sum": 0.0,
        "war_set_exact_sum": 0.0,
        "war_set_size_sum": 0.0,
        "maj_set_prec_sum": 0.0,
        "maj_set_rec_sum": 0.0,
        "maj_set_f1_sum": 0.0,
        "maj_set_jacc_sum": 0.0,
        "maj_set_exact_sum": 0.0,
        "maj_set_size_sum": 0.0,
        "selected_cost_sum": 0.0,
        "selected_count": 0,
    }

    majority_counts: Dict[str, int] = {agent.name: 0 for agent in agents}
    for ex in data:
        for gold in ex.gold_agents:
            if gold in majority_counts:
                majority_counts[gold] += 1
    majority_agent = max(majority_counts.items(), key=lambda x: x[1])[0] if majority_counts else None

    for ex in data:
        query_emb = embedder.encode([ex.prompt])[0]
        gold = ex.gold_agents

        r_knn = knn_router.route(query_emb)
        knn_pred_set = choose_pred_set(r_knn.candidates, set_threshold)
        p_s, r_s, f_s, j_s, e_s = set_metrics(knn_pred_set, gold)
        stats["knn_set_prec_sum"] += p_s
        stats["knn_set_rec_sum"] += r_s
        stats["knn_set_f1_sum"] += f_s
        stats["knn_set_jacc_sum"] += j_s
        stats["knn_set_exact_sum"] += e_s
        stats["knn_set_size_sum"] += len(knn_pred_set)

        r_ml = ml_router.route(query_emb)
        ml_pred_set = choose_pred_set(r_ml.candidates, set_threshold)
        p_s, r_s, f_s, j_s, e_s = set_metrics(ml_pred_set, gold)
        stats["ml_set_prec_sum"] += p_s
        stats["ml_set_rec_sum"] += r_s
        stats["ml_set_f1_sum"] += f_s
        stats["ml_set_jacc_sum"] += j_s
        stats["ml_set_exact_sum"] += e_s
        stats["ml_set_size_sum"] += len(ml_pred_set)

        cand_names = [name for name, _ in r_ml.candidates]
        chosen_war = bandit.choose(query_emb, candidate_actions=cand_names)
        war_pred_set = choose_pred_set(r_ml.candidates, set_threshold)
        if chosen_war not in war_pred_set:
            war_pred_set.append(chosen_war)
        p_s, r_s, f_s, j_s, e_s = set_metrics(war_pred_set, gold)
        stats["war_set_prec_sum"] += p_s
        stats["war_set_rec_sum"] += r_s
        stats["war_set_f1_sum"] += f_s
        stats["war_set_jacc_sum"] += j_s
        stats["war_set_exact_sum"] += e_s
        stats["war_set_size_sum"] += len(set(war_pred_set))

        if majority_agent is not None:
            p_s, r_s, f_s, j_s, e_s = set_metrics([majority_agent], gold)
            stats["maj_set_prec_sum"] += p_s
            stats["maj_set_rec_sum"] += r_s
            stats["maj_set_f1_sum"] += f_s
            stats["maj_set_jacc_sum"] += j_s
            stats["maj_set_exact_sum"] += e_s
            stats["maj_set_size_sum"] += 1

        success = 1.0 if chosen_war in gold else 0.0
        cost = 0.0
        for agent in agents:
            if agent.name == chosen_war:
                cost = agent.cost
                break
        reward = success - cost_lambda * cost
        stats["selected_cost_sum"] += cost
        stats["selected_count"] += 1
        bandit.update(chosen_war, query_emb, reward)

    n = len(data)
    avg_selected_cost = (stats["selected_cost_sum"] / stats["selected_count"]) if stats["selected_count"] else 0.0

    print("\n=== Routing Benchmark Results ===")
    print(f"Samples: {n}")
    print(f"[eval] set-eval mode: threshold (threshold={set_threshold})")
    print(f"[eval] WAR cost_lambda: {cost_lambda:.4f} | avg_selected_cost: {avg_selected_cost:.4f}")
    print("=== Variable-Set Metrics (model-decided K) ===")
    print(f"KNN      Prec: {100.0 * stats['knn_set_prec_sum'] / n:6.2f}% | Rec: {100.0 * stats['knn_set_rec_sum'] / n:6.2f}% | F1: {100.0 * stats['knn_set_f1_sum'] / n:6.2f}% | Jacc: {100.0 * stats['knn_set_jacc_sum'] / n:6.2f}% | Exact: {100.0 * stats['knn_set_exact_sum'] / n:6.2f}% | Avg|P|: {stats['knn_set_size_sum'] / n:5.2f}")
    print(f"ML       Prec: {100.0 * stats['ml_set_prec_sum'] / n:6.2f}% | Rec: {100.0 * stats['ml_set_rec_sum'] / n:6.2f}% | F1: {100.0 * stats['ml_set_f1_sum'] / n:6.2f}% | Jacc: {100.0 * stats['ml_set_jacc_sum'] / n:6.2f}% | Exact: {100.0 * stats['ml_set_exact_sum'] / n:6.2f}% | Avg|P|: {stats['ml_set_size_sum'] / n:5.2f}")
    print(f"WAR      Prec: {100.0 * stats['war_set_prec_sum'] / n:6.2f}% | Rec: {100.0 * stats['war_set_rec_sum'] / n:6.2f}% | F1: {100.0 * stats['war_set_f1_sum'] / n:6.2f}% | Jacc: {100.0 * stats['war_set_jacc_sum'] / n:6.2f}% | Exact: {100.0 * stats['war_set_exact_sum'] / n:6.2f}% | Avg|P|: {stats['war_set_size_sum'] / n:5.2f}")
    if majority_agent is not None:
        print(f"Majority Prec: {100.0 * stats['maj_set_prec_sum'] / n:6.2f}% | Rec: {100.0 * stats['maj_set_rec_sum'] / n:6.2f}% | F1: {100.0 * stats['maj_set_f1_sum'] / n:6.2f}% | Jacc: {100.0 * stats['maj_set_jacc_sum'] / n:6.2f}% | Exact: {100.0 * stats['maj_set_exact_sum'] / n:6.2f}% | Avg|P|: {stats['maj_set_size_sum'] / n:5.2f}")
    print("\nNote: Methods output variable-size predicted sets and are scored with set metrics.")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=None, help="Path to benchmark CSV")
    ap.add_argument("--ml", type=str, default="ml_router.pkl")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--cost_lambda", type=float, default=0.02)
    ap.add_argument("--set_threshold", type=float, default=0.5)
    ap.add_argument("--dataset_size", type=int, default=1000, help="Samples to generate when no CSV is provided")
    args = ap.parse_args()
    evaluate(
        args.dataset,
        args.ml,
        alpha=args.alpha,
        cost_lambda=args.cost_lambda,
        set_threshold=args.set_threshold,
        dataset_size=args.dataset_size,
    )
