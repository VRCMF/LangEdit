import json
from pathlib import Path

import torch
from transformers import AutoTokenizer
import ipdb

from util.globals import *

REMOTE_URL = f"{REMOTE_ROOT_URL}/data/dsets/zsre_mend_eval.json"


class BIZSREQADataset:
    """
    Dataset of factual knowledge based on zsRE.
    Specifically selected from the QA validation slice from Mitchell et al.
    Project page: http://nlp.cs.washington.edu/zeroshot/
    """

    def __init__(self, data_dir: str, tok: AutoTokenizer, size=None, lang_s='en', *args, **kwargs):
        data_dir = Path(data_dir)
        # zsre_loc = data_dir / "mulilingual_mzsRE_en_de_du_es_fr.json"
        # # ================
        if lang_s == 'en':
            zsre_loc = data_dir / "bizsre/bizsre_random_en.json"
        elif lang_s == 'zh':
            zsre_loc = data_dir / "bizsre/bizsre_random_zh.json"
        elif lang_s == 'mt':
            zsre_loc = data_dir / "bizsre/bizsre_1_mix_en_zh_random.json"
        elif lang_s == 'delta':
            zsre_loc = data_dir / "bizsre/bizsre_100_delta_en_zh_random.json"
        # # ================
        # if lang_s == 'en':
        #     zsre_loc = data_dir / "bizsre/bizsre_en.json"
        # elif lang_s == 'zh':
        #     zsre_loc = data_dir / "bizsre/bizsre_zh.json"
        # elif lang_s == 'mt':
        #     zsre_loc = data_dir / "bizsre/bizsre_1_mix_en_zh.json"
        # elif lang_s == 'delta':
        #     zsre_loc = data_dir / "bizsre/bizsre_100_delta_en_zh.json"
        # # ================
        if not zsre_loc.exists():
            print(f"{zsre_loc} does not exist. Downloading from {REMOTE_URL}")
            data_dir.mkdir(exist_ok=True, parents=True)
            torch.hub.download_url_to_file(REMOTE_URL, zsre_loc)

        with open(zsre_loc, "r") as f:
            raw = json.load(f)

        data = []
        for i, record in enumerate(raw):
            if "nq question: " not in record["loc"]:
                record["loc"] = "nq question: " + record["loc"]
            assert (
                "nq question: " in record["loc"]
            ), f"Neighborhood prompt missing `nq question:`. Check for errors?"
            ans_toks = tok(" " + record["loc_ans"])["input_ids"]
            data.append(
                {
                    "case_id": i,
                    "requested_rewrite": {
                        "prompt": record["src"].replace(record["subject"], "{}"),
                        "subject": record["subject"],
                        # "target_new": {"str": record["answers"][0]},
                        "target_new": {"str": record["alt"]},
                        "target_true": {"str": "<|endoftext|>"},
                    },
                    "paraphrase_prompts": [record["rephrase"]],
                    "neighborhood_prompts": [
                        {
                            "prompt": record["loc"] + "?" + tok.decode(ans_toks[:i]),
                            "target": tok.decode(ans_toks[i]),
                        }
                        for i in range(len(ans_toks))
                    ],
                    "attribute_prompts": [],
                    "generation_prompts": [],
                }
            )

        self._data = data[:size]

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)
