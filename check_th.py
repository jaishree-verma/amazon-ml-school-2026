import numpy as np
from collections import defaultdict
from evaluate_ab_blocking import enhanced_blocking, model, val_s1_ids, s1_records, candidates_pool, s1_gt, compute_macro_f05, compute_pairwise_features

val_candidates = {}
val_pair_feats = []
val_pair_keys = []
for sid in val_s1_ids:
    s1_rec = s1_records[sid]
    cand_set = enhanced_blocking.get_candidates(s1_rec, candidates_pool, top_k=30)
    val_candidates[sid] = cand_set
    for cid in cand_set:
        cand_rec = candidates_pool[cid]
        val_pair_feats.append(compute_pairwise_features(s1_rec, cand_rec))
        val_pair_keys.append((sid, cid))

X_val = np.array(val_pair_feats, dtype=np.float32)
val_probs = model.predict_proba(X_val)[:, 1]
val_s1_scores = defaultdict(list)
for (sid, cid), p in zip(val_pair_keys, val_probs):
    val_s1_scores[sid].append((cid, float(p)))

val_gt = {sid: s1_gt[sid] for sid in val_s1_ids}
for th in [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]:
    preds = {sid: {cid for cid, p in val_s1_scores[sid] if p >= th} for sid in val_s1_ids}
    m = compute_macro_f05(val_gt, preds)
    tp = sum(len(val_gt[sid] & preds[sid]) for sid in val_s1_ids)
    fp = sum(len(preds[sid] - val_gt[sid]) for sid in val_s1_ids)
    fn = sum(len(val_gt[sid] - preds[sid]) for sid in val_s1_ids)
    prec = tp / (tp + fp) if (tp+fp)>0 else 0
    rec = tp / (tp + fn) if (tp+fn)>0 else 0
    correct = sum(1 for sid in val_s1_ids if val_gt[sid] == preds[sid])
    print(f"Th={th:.2f} | Macro F0.5={m['macro_f05']:.4f} | Prec={prec:.4f} | Rec={rec:.4f} | Exact Acc={correct/2500*100:.2f}% | Singleton FPs={m['singleton_false_positives']}")
