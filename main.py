# main.py
import argparse

from train_ml import train_ml
from evaluate import evaluate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["train", "eval"], required=True)
    ap.add_argument("--dataset", type=str, default=None, help="Path to benchmark CSV")
    ap.add_argument("--ml_out", type=str, default="ml_router.pkl")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--cost_lambda", type=float, default=0.02, help="Cost penalty coefficient for WAR/LinUCB reward")
    ap.add_argument("--set_threshold", type=float, default=0.5, help="Absolute score threshold for set prediction")
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

    if args.mode == "train":
        use_profile_text = True
        if args.no_profile_text:
            use_profile_text = False
        if args.use_profile_text:
            use_profile_text = True
        train_ml(
            args.dataset,
            args.ml_out,
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
    else:
        evaluate(
            args.dataset,
            args.ml_out,
            alpha=args.alpha,
            cost_lambda=args.cost_lambda,
            set_threshold=args.set_threshold,
            dataset_size=args.dataset_size,
        )


if __name__ == "__main__":
    main()
