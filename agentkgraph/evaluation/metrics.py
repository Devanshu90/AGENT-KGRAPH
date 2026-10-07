import re
from collections import Counter


def _normalize_text(text):
    text=str(text or "").lower()
    text=re.sub(r"\s+", " ", text).strip()
    return text


def _tokens(text):
    return _normalize_text(text).split()


def exact_match(prediction, references):
    if isinstance(references,str):
        references=[references]
    prediction=_normalize_text(prediction)
    return float(any(prediction==_normalize_text(reference) for reference in references))


def token_f1(prediction, references):
    if isinstance(references,str):
        references=[references]

    pred=_tokens(prediction)

    best=0.0

    for reference in references:
        ref=_tokens(reference)

        if not pred and not ref:
            score=1.0
        elif not pred or not ref:
            score=0.0
        else:
            common=Counter(pred)&Counter(ref)
            overlap=sum(common.values())

            if overlap==0:
                score=0.0
            else:
                precision=overlap/len(pred)
                recall=overlap/len(ref)
                score=2*precision*recall/(precision+recall)

        best=max(best,score)

    return best


def hits_at_1(predictions,references):
    if isinstance(references,str):
        references=[references]

    if not predictions:
        return 0.0

    first=_normalize_text(predictions[0])

    return float(any(first==_normalize_text(reference) for reference in references))


def precision_at_k(predictions,references,k):
    if isinstance(references,str):
        references=[references]

    if k<=0:
        return 0.0

    selected=predictions[:k]

    if not selected:
        return 0.0

    refs={_normalize_text(reference) for reference in references}

    hits=sum(
        1
        for prediction in selected
        if _normalize_text(prediction) in refs
    )

    return hits/len(selected)


def provenance_completeness(provenance,required):
    if required<=0:
        return 1.0

    if provenance is None:
        return 0.0

    if isinstance(provenance,dict):
        count=len(provenance)
    elif isinstance(provenance,(list,tuple,set)):
        count=len(provenance)
    else:
        count=1

    return min(count/required,1.0)


def routing_accuracy(predicted_routes,expected_routes):
    if len(predicted_routes)!=len(expected_routes):
        raise ValueError("Prediction and expected route lengths must match.")

    if not predicted_routes:
        return 0.0

    correct=sum(
        1
        for predicted,expected in zip(predicted_routes,expected_routes)
        if str(predicted).lower()==str(expected).lower()
    )

    return correct/len(predicted_routes)