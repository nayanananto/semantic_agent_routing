# train_ml.py
from __future__ import annotations

from typing import List
import pickle

import numpy as np

from agents import get_agents
from dataset import generate_dataset_300, load_dataset_csv
from embedder import Embedder, EmbedderConfig


def train_ml(
    dataset_path: str | None,
    output_path: str = "ml_router.pkl",
    split_by_template: bool = False,
    ml_model: str = "svm",
    embedder_model: str | None = None,
    dataset_size: int = 1000,
    ml_mode: str = "multilabel",
    min_test_per_agent: int = 1,
    max_negatives_per_prompt: int = 5,
    use_profile_text: bool = True,
    split_seed: int = 42,
) -> None:
    agents = get_agents()
    agent_names = [agent.name for agent in agents]
    name_to_idx = {name: i for i, name in enumerate(agent_names)}

    priority = [
        "SQLQueryAgent",
        "TimeSeriesQueryAgent",
        "APIDataFetchAgent",
        "LogRetrievalAgent",
        "MetadataLookupAgent",
        "ForecastAgent",
        "AnomalyDetectionAgent",
        "TrendAnalysisAgent",
        "StatisticalAnalysisAgent",
        "PlotGenerationAgent",
        "SummaryAgent",
        "ReportWriterAgent",
    ]
    priority_rank = {name: i for i, name in enumerate(priority)}

    def pick_label(gold_list: List[str]) -> str:
        # Used only to build a stable stratification label for train/test splitting.
        return sorted(gold_list, key=lambda x: priority_rank.get(x, 10**9))[0]

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

    def _agent_desc_texts():
        texts = []
        labels = []
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
            for text in chunks:
                texts.append(text)
                labels.append(agent.name)
        return texts, labels

    def _agent_profile_texts():
        texts = []
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

    if ml_model != "svm":
        raise ValueError("Only the SVM configuration is kept in the cleaned training pipeline.")
    if ml_mode not in {"multilabel", "pairwise"}:
        raise ValueError("Only multilabel and pairwise modes are kept in the cleaned training pipeline.")

    data = load_dataset_csv(dataset_path) if dataset_path else generate_dataset_300(n_samples=dataset_size)

    primary_y = np.array([name_to_idx[pick_label(ex.gold_agents)] for ex in data], dtype=int)
    y_multi = np.zeros((len(data), len(agent_names)), dtype=int)
    for i, ex in enumerate(data):
        for gold in ex.gold_agents:
            if gold in name_to_idx:
                y_multi[i, name_to_idx[gold]] = 1

    from sklearn.model_selection import train_test_split

    prompts = np.array([ex.prompt for ex in data], dtype=object)
    prompt_ids = np.array([ex.prompt_id for ex in data], dtype=int)
    template_ids = np.array([ex.template_id for ex in data], dtype=object)
    idx = np.arange(len(data))

    if split_by_template and any(t is not None for t in template_ids):
        from sklearn.model_selection import GroupShuffleSplit

        groups = template_ids
        splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=split_seed)
        train_idx, test_idx = next(splitter.split(idx, primary_y, groups=groups))
    else:
        if split_by_template:
            print("[warn] split_by_template requested but no template_id found; falling back to random split.")
        train_idx, test_idx = train_test_split(
            idx,
            test_size=0.2,
            random_state=split_seed,
            stratify=primary_y,
        )

    train_set = set(train_idx.tolist())
    test_set = set(test_idx.tolist())
    test_counts = {agent: 0 for agent in agent_names}
    for i in test_set:
        for gold in data[i].gold_agents:
            if gold in test_counts:
                test_counts[gold] += 1

    for agent in agent_names:
        needed = max(0, min_test_per_agent - test_counts.get(agent, 0))
        if needed == 0:
            continue
        moved = 0
        for i in list(train_set):
            if agent in data[i].gold_agents:
                train_set.remove(i)
                test_set.add(i)
                for gold in data[i].gold_agents:
                    if gold in test_counts:
                        test_counts[gold] += 1
                moved += 1
                if moved >= needed:
                    break
        if moved < needed:
            print(f"[warn] Could not reach min_test_per_agent={min_test_per_agent} for {agent} (moved {moved}).")

    train_idx = np.array(sorted(train_set), dtype=int)
    test_idx = np.array(sorted(test_set), dtype=int)
    prompts_train, prompts_test = prompts[train_idx], prompts[test_idx]
    y_multi_train, y_multi_test = y_multi[train_idx], y_multi[test_idx]
    id_test = prompt_ids[test_idx]

    profile_texts = _agent_profile_texts() if use_profile_text else [agent.name for agent in agents]

    embedder_cfg = EmbedderConfig()
    if embedder_model:
        embedder_cfg.model_name = embedder_model
    embedder = Embedder(embedder_cfg)
    print(f"[train_ml] embedder backend: {embedder.backend[0]}")
    if embedder.backend[0] == "sbert":
        print(f"[train_ml] embedder model: {embedder.config.model_name}")
    if embedder.backend[0] == "tfidf":
        fit_texts = list(prompts_train) + [agent.description for agent in agents] + profile_texts
        embedder.fit(fit_texts)

    X_train = embedder.encode(list(prompts_train))
    X_test = embedder.encode(list(prompts_test))
    profile_embs = embedder.encode(profile_texts) if profile_texts else None

    def _pair_features(prompt_emb: np.ndarray, agent_embs: np.ndarray) -> np.ndarray:
        p = np.repeat(prompt_emb.reshape(1, -1), agent_embs.shape[0], axis=0)
        a = agent_embs
        return np.concatenate([p, a, np.abs(p - a), p * a], axis=1)

    if ml_mode == "pairwise":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.svm import LinearSVC

        X_pairs = []
        y_pairs = []
        rng = np.random.default_rng(split_seed)
        for i in train_idx:
            prompt_emb = embedder.encode([prompts[i]])[0]
            gold = set(data[i].gold_agents)
            pos_idx = [j for j in range(len(agent_names)) if agent_names[j] in gold]
            neg_idx = [j for j in range(len(agent_names)) if agent_names[j] not in gold]
            if max_negatives_per_prompt > 0 and len(neg_idx) > max_negatives_per_prompt:
                neg_idx = rng.choice(neg_idx, size=max_negatives_per_prompt, replace=False).tolist()
            sel_idx = pos_idx + neg_idx
            feats = _pair_features(prompt_emb, profile_embs)[sel_idx]
            labels = np.array([1 if j in pos_idx else 0 for j in sel_idx], dtype=int)
            X_pairs.append(feats)
            y_pairs.append(labels)

        X_train_pw = np.vstack(X_pairs)
        y_train_pw = np.concatenate(y_pairs)
        base = LinearSVC(class_weight="balanced", max_iter=5000)
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        clf.fit(X_train_pw, y_train_pw)

        correct = 0
        for i in test_idx:
            prompt_emb = embedder.encode([prompts[i]])[0]
            feats = _pair_features(prompt_emb, profile_embs)
            probs = clf.predict_proba(feats)
            classes = list(clf.classes_)
            pos_idx = classes.index(1) if 1 in classes else -1
            scores = probs[:, pos_idx]
            chosen_idx = int(np.argmax(scores))
            if agent_names[chosen_idx] in data[i].gold_agents:
                correct += 1
        holdout_score = correct / len(test_idx) if len(test_idx) else 0.0
        class_names = agent_names
    else:
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.svm import LinearSVC

        if use_profile_text:
            desc_texts, desc_labels = _agent_desc_texts()
            if desc_texts:
                X_desc = embedder.encode(desc_texts)
                y_desc = np.zeros((len(desc_labels), len(agent_names)), dtype=int)
                for i, name in enumerate(desc_labels):
                    y_desc[i, name_to_idx[name]] = 1
                X_train = np.vstack([X_train, X_desc])
                y_multi_train = np.vstack([y_multi_train, y_desc])

        base = LinearSVC(class_weight="balanced", max_iter=5000)
        clf = OneVsRestClassifier(base)
        clf.fit(X_train, y_multi_train)

        if hasattr(clf, "predict_proba"):
            scores = clf.predict_proba(X_test)
        else:
            scores = clf.decision_function(X_test)
            scores = 1.0 / (1.0 + np.exp(-scores))
        top1_idx = np.argmax(scores, axis=1)
        holdout_score = float(np.mean([y_multi_test[i, j] == 1 for i, j in enumerate(top1_idx)])) if len(top1_idx) else 0.0
        class_names = agent_names

    payload = {
        "model": clf,
        "class_names": class_names,
        "embedder_backend": embedder.backend[0],
        "embedder_config": embedder.config,
        "tfidf_vectorizer": embedder._tfidf,
        "test_prompt_ids": id_test.tolist(),
        "ml_mode": ml_mode,
        "agent_profile_texts": profile_texts,
        "use_profile_text": use_profile_text,
        "label_mode": ml_mode,
    }

    with open(output_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"[train_ml] saved model to {output_path}")
    print(f"[train_ml] holdout top-1 hit: {holdout_score:.4f}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, default=None, help="Path to benchmark CSV")
    ap.add_argument("--out", type=str, default="ml_router.pkl")
    ap.add_argument("--split_by_template", action="store_true", help="Use template-family holdout split when available")
    ap.add_argument("--ml_model", choices=["svm"], default="svm")
    ap.add_argument("--embedder_model", type=str, default=None, help="Sentence-transformer model name")
    ap.add_argument("--dataset_size", type=int, default=1000, help="Samples to generate when no CSV is provided")
    ap.add_argument("--ml_mode", choices=["pairwise", "multilabel"], default="multilabel")
    ap.add_argument("--min_test_per_agent", type=int, default=10, help="Minimum test examples per agent")
    ap.add_argument("--max_negatives_per_prompt", type=int, default=5, help="Max negatives per prompt for pairwise training")
    ap.add_argument("--use_profile_text", action="store_true", help="Include agent profile text in training")
    ap.add_argument("--no_profile_text", action="store_true", help="Disable agent profile text in training")
    ap.add_argument("--split_seed", type=int, default=42, help="Random seed for train/test split and sampling")
    args = ap.parse_args()

    use_profile_text = True
    if args.no_profile_text:
        use_profile_text = False
    if args.use_profile_text:
        use_profile_text = True

    train_ml(
        args.dataset,
        args.out,
        split_by_template=args.split_by_template,
        ml_model=args.ml_model,
        embedder_model=args.embedder_model,
        dataset_size=args.dataset_size,
        ml_mode=args.ml_mode,
        min_test_per_agent=args.min_test_per_agent,
        max_negatives_per_prompt=args.max_negatives_per_prompt,
        use_profile_text=use_profile_text,
        split_seed=args.split_seed,
    )
