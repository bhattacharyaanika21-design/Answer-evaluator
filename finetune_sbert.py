"""Optional: fine-tune Sentence-BERT on your own teacher-marked answers.

Each training pair is (reference answer, student answer) labelled with the
normalised teacher score (0..1). CosineSimilarityLoss pushes the cosine of the two
embeddings towards that label, so the model learns what "a good answer" looks like
in your subject.

    python finetune_sbert.py --epochs 4
    python train.py --backend sbert --sbert-model models/sbert-finetuned   (see README)

Keep one question out for testing so the reported scores stay honest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--holdout-question", default="Q6", help="question kept out of training")
    ap.add_argument("--out", default=str(ROOT / "models" / "sbert-finetuned"))
    args = ap.parse_args()

    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    questions = {q["id"]: q for q in json.loads((ROOT / "data" / "questions.json").read_text())}
    answers = pd.read_csv(ROOT / "data" / "student_answers.csv")
    train = answers[answers.question_id != args.holdout_question]

    examples = [InputExample(texts=[questions[r.question_id]["reference_answer"], r.student_answer],
                             label=float(r.human_score) / questions[r.question_id]["max_marks"])
                for r in train.itertuples()]
    print(f"Training pairs: {len(examples)} (held out {args.holdout_question})")

    model = SentenceTransformer(args.base_model)
    loader = DataLoader(examples, shuffle=True, batch_size=args.batch_size)
    loss = losses.CosineSimilarityLoss(model)
    model.fit(train_objectives=[(loader, loss)], epochs=args.epochs,
              warmup_steps=max(1, len(loader) // 10), show_progress_bar=True)
    model.save(args.out)

    # Quick check on the held-out question
    test = answers[answers.question_id == args.holdout_question]
    if len(test):
        from scipy.stats import pearsonr

        ref = questions[args.holdout_question]["reference_answer"]
        emb = model.encode([ref] + test.student_answer.tolist(), normalize_embeddings=True)
        cos = emb[1:] @ emb[0]
        print(f"Held-out Pearson(cosine, teacher score) = {pearsonr(cos, test.human_score)[0]:.3f}")
    print(f"Saved fine-tuned model to {args.out}")


if __name__ == "__main__":
    main()
