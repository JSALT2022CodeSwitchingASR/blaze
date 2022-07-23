#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
# 	       2022  Xiaomi Crop.        (authors: Mingshuang Luo)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Compute fbank features for a language of the soapies corpus.

It looks for manifests in the directory data/manifests.

The generated fbank features are saved in data/fbank.
"""

import argparse
import logging
import os
from pathlib import Path

import torch
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter
from lhotse.recipes.utils import read_manifests_if_cached

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
# Do this outside of main() in case it needs to take effect
# even when we are not invoking the main (e.g. when spawning subprocesses).
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def compute_fbank_soapies(lang):
    src_dir = Path("data/manifests") / lang
    output_dir = Path("data/fbank") / lang
    num_jobs = min(15, os.cpu_count())
    num_mel_bins = 80

    dataset_parts = (
        "train",
        "dev",
        "test",
    )

    prefix = f"soapies-{lang}"
    suffix = "jsonl.gz"
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
    )
    assert manifests is not None

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins))

    for partition, m in manifests.items():
        if (output_dir / f"{prefix}_cuts_{partition}.{suffix}").is_file():
            logging.info(f"{partition} already exists - skipping.")
            continue
        logging.info(f"Processing {partition}")
        cut_set = CutSet.from_manifests(
            recordings=m["recordings"],
            supervisions=m["supervisions"],
        )

        if "train" in partition:
            cut_set = (
                cut_set
                + cut_set.perturb_speed(0.9)
                + cut_set.perturb_speed(1.1)
            )
        cur_num_jobs = min(num_jobs, len(cut_set))

        cut_set = cut_set.compute_and_store_features(
            extractor=extractor,
            storage_path=f"{output_dir}/{prefix}_feats_{partition}",
            # when an executor is specified, make more partitions
            num_jobs=cur_num_jobs,
            storage_type=LilcomChunkyWriter,
        )
        # Split long cuts into many short and un-overlapping cuts
        cut_set = cut_set.trim_to_supervisions(keep_overlapping=False)
        cut_set.to_file(output_dir / f"{prefix}_cuts_{partition}.{suffix}")


if __name__ == "__main__":
    formatter = (
        "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
    )

    logging.basicConfig(format=formatter, level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lang",
        help="language for which to extract the features")
    args = parser.parse_args()

    compute_fbank_soapies(args.lang)

