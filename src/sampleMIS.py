import itertools, logging, copy
from typing import Any, Dict, List, Sequence, Tuple, Set
from itertools import combinations

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from src.defense import *

logger = logging.getLogger('RRAG-main')

# ────────────────────────── MIS helpers ──────────────────────────
def _is_independent(subset: Sequence[int],
                    graph: Dict[int, Set[int]]) -> bool:
    return all(u not in graph[v] for v in subset for u in subset if u != v)


def _lex_key(rank_tuple: Sequence[int]) -> Tuple[int]:
    # rank_tuple is already sorted, but keep this helper for completeness
    return tuple(rank_tuple)


def _max_independent_set_lex(
    graph: Dict[int, Set[int]],
    rank_tuples: List[Tuple[int, ...]],
) -> List[int]:
    """Maximum independent set; among ties pick lexicographically best."""
    vertices = list(graph.keys())
    best_set, best_key = [], None
    for r in range(len(vertices), 0, -1):            # largest → smallest
        for subset in combinations(vertices, r):
            if _is_independent(subset, graph):
                key = sorted(_lex_key(rank_tuples[i]) for i in subset)
                if best_key is None or key < best_key:
                    best_set, best_key = list(subset), key
        if best_set:
            break
    return best_set


# ───────────────────── Sample‑MIS‑RAG class ──────────────────────
class SampleMISRRAG(RRAG):
    """
    γ‑weighted sampling  →  MIS contradiction filtering  →  final query
    on the union of docs in the MIS (lower‑ranked docs first).
    """

    def __init__(
        self,
        llm,
        sample_size: int = 1,
        num_samples: int = 10,
        gamma: float = 1.0,
        nli_model_path: str = (
            "/scratch/gpfs/zs7353/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
        ),
        contradiction_threshold: float = 0.5,
    ):
        super().__init__(llm)
        self.sample_size = sample_size
        self.num_samples = num_samples
        self.gamma       = gamma
        self.thres       = contradiction_threshold

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.nli_tok = AutoTokenizer.from_pretrained(nli_model_path)
        self.nli     = AutoModelForSequenceClassification.from_pretrained(
            nli_model_path
        ).to(device)

    # ───────────────────────── main entry ─────────────────────────
    def query(self, data_item: Dict[str, Any]) -> str:
        docs = data_item["topk_content"]              # original ranking
        K    = len(docs)

        if K == 0:
            return "I don't know."

        weights = np.array([self.gamma**i for i in range(K)], dtype=float)
        weights /= weights.sum()

        sample_sets: List[List[str]]    = []
        ranks:        List[List[int]]   = []          # 1‑based indices
        rank_tuples:  List[Tuple[int]]  = []          # for tie‑break
        prompts:      List[str]         = []

        for _ in range(self.num_samples):
            idxs = np.random.choice(
                K, size=min(self.sample_size, K), replace=True, p=weights
            )
            print(idxs)

            idxs_sorted = sorted(idxs)
            subset_docs = list(reversed([docs[i] for i in idxs_sorted]))
            # subset_docs = [docs[i] for i in idxs_sorted]
            sample_sets.append(subset_docs)
            ranks.append([i + 1 for i in idxs_sorted])            # 1‑based
            rank_tuples.append(tuple(i + 1 for i in idxs_sorted)) # tie‑break

            subset_item = copy.deepcopy(data_item)
            subset_item["topk_content"] = subset_docs
            prompts.append(
                self.llm.wrap_prompt(
                    subset_item,
                    as_multi_choice=False,
                    seperate=False
                )
            )

        answers: List[str] = self.llm.batch_query(prompts)
        print(answers)

        graph = {i: set() for i in range(self.num_samples)}
        premises, hypotheses, pairs = [], [], []
        for i, j in combinations(range(self.num_samples), 2):
            premise = (
                f"The answer to the question: {data_item['question']}\n"
                f"is {answers[i]}."
            )
            hypothesis = (
                f"The answer to the question: {data_item['question']}\n"
                f"is {answers[j]}."
            )
            premises.append(premise)
            hypotheses.append(hypothesis)
            pairs.append((i, j))

        if premises:
            inputs = self.nli_tok(
                premises, hypotheses,
                return_tensors="pt", truncation=True, padding=True
            ).to(self.nli.device)
            with torch.no_grad():
                logits = self.nli(**inputs).logits
            probs = torch.softmax(logits, dim=1)[:, 2]   # CONTRADICTION
            for p, (i, j) in zip(probs.tolist(), pairs):
                if p >= self.thres and "I don't know" not in answers[i] and "I don't know" not in answers[j]:
                    graph[i].add(j)
                    graph[j].add(i)

        mis_set_idx = _max_independent_set_lex(graph, rank_tuples)
        logger.info(f"MIS set document indices: {mis_set_idx}")
        
        mis_doc_idxs = [idx - 1 for s in mis_set_idx for idx in ranks[s] if "I don't know" not in answers[s]]
        mis_docs = [docs[i] for i in mis_doc_idxs]
        logger.info(f"MIS document indices: {mis_doc_idxs}")

        final_item = copy.deepcopy(data_item)
        final_item["topk_content"] = mis_docs
        query_prompt = self.llm.wrap_prompt(
            final_item,
            as_multi_choice=False,
            seperate=False
        )
        return self.llm.query(query_prompt)