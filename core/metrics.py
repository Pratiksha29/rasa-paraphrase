from typing import List, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizer

from core.paraphraser import ParaphraseModel


def embeddings(model: PreTrainedModel, tokens: List[int]):
    """extract embeddings from tokens"""
    return model(torch.tensor(tokens), labels=torch.tensor(tokens))[:2][1]


def pad(txt1: str, txt2: str, tokenizer: PreTrainedTokenizer, char: str = ">"):
    """normalize token size before encoding"""
    txt1 = txt1.strip()
    txt2 = txt2.strip()
    lens = [
        (txt1, len(tokenizer.tokenize(txt1))),
        (txt2, len(tokenizer.tokenize(txt2))),
    ]
    pad_num = max([l for _, l in lens])

    padded = []
    for t, ln in lens:
        if ln < pad_num:
            t += f" {char}" * (pad_num - ln)
        padded.append(t)
    return tuple(padded)


def distance(txt1: str, txt2: str, model: ParaphraseModel) -> tuple:
    """
    calc distance between phrases
    --
    pairwise distance
    cosine similarity
    euclidean-norm
    """
    pd_txt1, pd_txt2 = pad(txt1, txt2, model.tokenizer)
    tkns1 = model.tokenizer.encode(pd_txt1)
    tkns2 = model.tokenizer.encode(pd_txt2)

    emb1 = embeddings(model.model, tkns1)
    emb2 = embeddings(model.model, tkns2)

    pdist = torch.nn.PairwiseDistance(p=2)
    pw_dist = pdist(emb1, emb2)
    min_pw_dist = float(torch.min(pw_dist))

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    cosd = cos(emb1, emb2)
    min_cosd = float(torch.min(cosd))

    norm = torch.dist(emb1, emb2, p=2)

    return (min_pw_dist, min_cosd, norm)
