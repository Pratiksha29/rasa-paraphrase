import os
from typing import List

from core.paraphraser import (
    ModelParams,
    ParaphraseModel,
    set_seed,
)

# from core.md_parser import extract_nlu
from core.util import filename, file_ext
from zipfile import ZipFile

DEFAULT_PARAMS: List[tuple] = [
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

def gen_paraphrases(input: str, params: ModelParams) -> list:
    # Seed parameters
    set_seed(params)

    # Initialize Model
    model_path: str = os.getenv("MODEL_PATH")
    model: ParaphraseModel = ParaphraseModel(model_path, params)

    # Get paraphrases from input
    return model.get_paraphrases(input, int(params.num_samples), ";")

if __name__ == "__main__":
    import argparse
    from time import perf_counter

    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('--input', help="A sentence to paraphrase")
    parser.add_argument('--nlu', help="A rasa nlu markdown file")
    parser.add_argument('--verbose', default=True)

    # TODO: move to invoke cli interface

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
        args.length,
        args.num_samples,
        args.temperature,
        args.repetition_penalty,
        args.top_k,
        args.top_p,
        args.no_cuda,
        args.seed,
        args.stop_token,
    )

    if args.input:
        print(gen_paraphrases(args.input, params))

    if args.nlu:
        convert(open_file('examples/nlu_no_entities.md'))

    # Stop the clock
    stop_perf = perf_counter()

    if args.verbose:
        print("Elapsed time:", abs(stop_perf - start_perf))
