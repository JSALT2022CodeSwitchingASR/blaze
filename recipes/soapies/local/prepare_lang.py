import argparse
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lang-dir",
        type=str,
        help="""Input and output directory.
        It should contain a file lexicon.txt.
        """,
    )

    return parser.parse_args()


def write_mapping(filename: str, sym2id: Dict[str, int]) -> None:
    with open(filename, "w", encoding="utf-8") as f:
        for sym, i in sym2id.items():
            f.write(f"{sym} {i}\n")

def generate_id_map(symbols: List[str]) -> Dict[str, int]:
    return {sym: i for i, sym in enumerate(symbols)}


def read_lexicon(path):
    lexicon = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            tokens = line.strip().split()
            lexicon[tokens[0]].append(tokens[1:])
    return lexicon


def main():
    args = get_args()
    lang_dir = Path(args.lang_dir)
    lexicon_filename = lang_dir / "lexicon.txt"

    lexicon = read_lexicon(lexicon_filename)


    assert "<s>" not in lexicon.keys()
    assert "</s>" not in lexicon.keys()

    tokens = set()
    for word in lexicon:
        for pronun in lexicon[word]:
            for token in pronun:
                tokens.add(token)


    token2id = generate_id_map(sorted(tokens))
    word2id = generate_id_map(sorted(lexicon.keys()))

    write_mapping(lang_dir / "tokens.txt", token2id)
    write_mapping(lang_dir / "words.txt", word2id)

if __name__ == "__main__":
    main()
