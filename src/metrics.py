import numpy as np

def compute_macro_f05(ground_truth, predictions):
    """
    Calculates the official Challenge Macro F_0.5 score, Precision, Recall, and Singleton statistics.
    
    Args:
        ground_truth: dict of {s1_id: set of true matching s2/s3 ids}
        predictions: dict of {s1_id: set of predicted matching s2/s3 ids}
        
    Returns:
        dict containing:
            - macro_f05
            - avg_precision
            - avg_recall
            - singleton_accuracy
            - total_entities
            - total_singletons
            - singleton_false_positives
    """
    total_f05 = 0.0
    total_prec = 0.0
    total_rec = 0.0
    
    total_entities = len(ground_truth)
    total_singletons = 0
    singleton_correct = 0
    singleton_fps = 0
    
    non_singleton_count = 0
    
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        
        is_singleton = (len(true_set) == 0)
        
        if is_singleton:
            total_singletons += 1
            if len(pred_set) == 0:
                f05 = 1.0
                prec = 1.0
                rec = 1.0
                singleton_correct += 1
            else:
                f05 = 0.0
                prec = 0.0
                rec = 1.0
                singleton_fps += 1
        else:
            non_singleton_count += 1
            if len(pred_set) == 0:
                f05 = 0.0
                prec = 0.0
                rec = 0.0
            else:
                tp = len(true_set & pred_set)
                fp = len(pred_set - true_set)
                fn = len(true_set - pred_set)
                
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                
                if (0.25 * prec + rec) > 0:
                    f05 = (1.25 * prec * rec) / (0.25 * prec + rec)
                else:
                    f05 = 0.0
                    
        total_f05 += f05
        total_prec += prec
        total_rec += rec
        
    macro_f05 = total_f05 / total_entities if total_entities > 0 else 0.0
    avg_precision = total_prec / total_entities if total_entities > 0 else 0.0
    avg_recall = total_rec / total_entities if total_entities > 0 else 0.0
    singleton_acc = singleton_correct / total_singletons if total_singletons > 0 else 1.0
    
    return {
        "macro_f05": round(macro_f05, 4),
        "avg_precision": round(avg_precision, 4),
        "avg_recall": round(avg_recall, 4),
        "singleton_accuracy": round(singleton_acc, 4),
        "total_entities": total_entities,
        "total_singletons": total_singletons,
        "singleton_false_positives": singleton_fps
    }
