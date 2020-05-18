import os
import random
from typing import List, Tuple

import pandas as pd
import torch
from tabulate import tabulate
from tqdm import tqdm

from core.md_parser import extract_nlu, nlu2md
from core.paraphraser import ModelParams, ParaphraseModel, set_seed
from core.metrics import distance
from core.util import file_ext, filename, open_file

DEFAULT_PARAMS: List[Tuple] = [
    ("model_type", "gpt2"),
    ("model_name_or_path", ""),
    ("prompt", ""),
    ("padding_text", ""),
    ("xlm_lang", ""),
    ("length", 20),
    ("num_samples", 3),
    ("temperature", 1.0),
    ("repetition_penalty", 1.0),
    ("top_k", 20),
    ("top_p", 0.0),
    ("no_cuda", False),
    ("seed", 42),
    ("stop_token", None),
]


def expand_nlu(
    md: str, params: ModelParams, model: ParaphraseModel, verbose=False
) -> str:
    utterance_groups: list = extract_nlu(md)
    max_samples: int = max([len(u[1]) for u in utterance_groups])

    expanded_utterances: list = []
    h2s: list = []
    for h2, utterances in tqdm(utterance_groups):
        h2s.append(h2)
        num_samples = max_samples - len(utterances)
        if utterances:
            # TODO: how do we select the sample? k-cluster?
            sample_utterance: str = random.choice(utterances)
            # TODO: similarity filter may need to occur during generation (as a penalty)
            gen: list = gen_paraphrases(model, sample_utterance, num_samples)
            if gen not in utterances:
                utterances.extend(gen)
        expanded_utterances.append(utterances)

    if verbose:
        print(
            f"\nAdded {len(expanded_utterances) - len(utterance_groups[1])} paraphrased utterances"
        )
    return nlu2md(list(zip(h2s, expanded_utterances)))


def gen_paraphrases(model: ParaphraseModel, input: str, num_samples: int) -> list:
    """Generate paraphrases from input"""
    return model.get_paraphrases(input, num_samples, ";")


def init_model(params: ModelParams, model_path: str) -> ParaphraseModel:
    """Initialize model with static and dynamic params"""
    model: ParaphraseModel = ParaphraseModel(model_path, params)
    set_seed(params)
    model.load_model(model_path)
    return model


if __name__ == "__main__":
    import argparse
    from time import perf_counter

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Sentence as input")
    parser.add_argument("--nlu", help="RASA nlu markdown file as input")
    parser.add_argument(
        "--similarity", help="Displays text with similarity as tabular output"
    )
    parser.add_argument("--csv", help="CSV output file path")
    parser.add_argument("--verbose", default=False)

    # TODO: consider invoke cli interface

    for k, v in DEFAULT_PARAMS:
        # TODO descriptions
        parser.add_argument(f"--{k}", default=v)
    args = parser.parse_args()

    # Start the clock
    start_perf = perf_counter()

    # Define all parameters
    params: ModelParams = ModelParams(
        args.model_type,
        args.model_name_or_path,
        args.prompt,
        args.padding_text,
        args.xlm_lang,
        int(args.length),
        int(args.num_samples),
        float(args.temperature),
        float(args.repetition_penalty),
        int(args.top_k),
        float(args.top_p),
        args.no_cuda,
        int(args.seed),
        args.stop_token,
    )

    # Initialize Model
    model = init_model(params, os.getenv("MODEL_PATH"))

    if args.input:
        sims = []
        paraphrases = gen_paraphrases(model, args.input, params.num_samples)
        for para in paraphrases:
            _, cos, _ = distance(args.input, para, model)
            sims.append(cos)
            if not args.similarity:
                print(para)

        outputs_with_sim: List[Tuple] = sorted(
            list(zip(paraphrases, sims)), key=lambda x: x[1], reverse=True
        )
        if args.similarity:
            similar: List[List] = [
                [t, s] for t, s in outputs_with_sim if s >= float(args.similarity)
            ]
            less_similar: List[List] = [
                [t, s] for t, s in outputs_with_sim if s < float(args.similarity)
            ]

            print("Similar: ")
            print(tabulate(similar))

            print("\nLess Similar: ")
            print(tabulate(less_similar))

        if args.csv:
            # csv will always include similarity
            df = pd.DataFrame(outputs_with_sim, columns=["text", "similarity"])
            df.to_csv(args.csv)

    if args.nlu:
        expanded_md = expand_nlu(
            open_file(args.nlu), params, model=model, verbose=args.verbose
        )
        print(expanded_md)

    # Stop the clock
    stop_perf = perf_counter()

    if args.verbose:
        print("Elapsed time:", abs(stop_perf - start_perf))
