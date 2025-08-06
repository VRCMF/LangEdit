import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import csv
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome.layer_stats import layer_stats
from util import nethook
from util.generate import generate_fast
from util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .AlphaEdit_hparams import AlphaEditHyperParams
import ipdb
# Cache variable(s)
# CONTEXT_TEMPLATES_CACHE = None
CONTEXT_TEMPLATES_CACHE_en = None
CONTEXT_TEMPLATES_CACHE_fr = None
CONTEXT_TEMPLATES_CACHE_es = None
CONTEXT_TEMPLATES_CACHE_de = None
CONTEXT_TEMPLATES_CACHE_nl = None
COV_CACHE = {}

def apply_AlphaEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: AlphaEditHyperParams,
    lang_s = 'en',
    cnt = 0,
    cov_list = None,
    cache_template: Optional[str] = None,
    cache_c = None,
    P = None,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            # Space required for correct tokenization
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]
    for request in requests[:10]:
        print(
            f"MEMIT request sample: "
            f"[{request['prompt'].format(request['subject'])}] -> [{request['target_new']['str']}]"
        )

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # reclaim
    mt_list = ['en', 'de', 'nl', 'es', 'fr']
    if lang_s == 'delta':
        lang_s = mt_list[cnt % 5]
        # context_templates = get_context_templates(model, tok, mt_list[cnt % 5])
        # ipdb.set_trace()
    # Compute z for final layer
    if lang_s in mt_list:
        context_templates = get_context_templates(model, tok, lang_s)
    else:
        context_templates_list = []
        for cur_lang in mt_list:
            context_templates_list.append(get_context_templates(model, tok, cur_lang))
    z_layer = hparams.layers[-1]
    z_list = []

    # for request in requests:
    for request_idx, request in enumerate(requests):
        if lang_s == 'mt':
            position_in_cycle = request_idx % 5
            context_templates = context_templates_list[position_in_cycle]
        # Retrieve k/v pair if already stored in cache
        cache_fname = (
            Path(
                str(cache_template).format(
                    z_layer, hparams.clamp_norm_factor, request["case_id"]
                )
            )
            if cache_template is not None
            else None
        )
        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                z_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            cur_z = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates,
            )

            z_list.append(cur_z)

            if cache_fname is not None:
                cache_fname.parent.mkdir(exist_ok=True, parents=True)
                np.savez(
                    cache_fname,
                    **{
                        "v_star": cur_z.detach().cpu().numpy(),
                    },
                )
                print(f"Cached k/v pair at {cache_fname}")
    zs = torch.stack(z_list, dim=1)

    for i, layer in enumerate(hparams.layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

        # Compute residual error
        cur_zs = get_module_input_output_at_words(
            model,
            tok,
            z_layer,
            context_templates=[request["prompt"] for request in requests],
            words=[request["subject"] for request in requests],
            module_template=hparams.layer_module_tmp,
            fact_token_strategy=hparams.fact_token,
        )[1].T
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())

        repeat_factor = (layer_ks.size(1) // targets.size(1))
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers

        # recalculate the P 
        if cnt == 0:
            P[i,:,:], cov = get_project(model,tok,layer,hparams, None, None, cnt)
            cov_list.append(cov)
        else:
            P[i,:,:], cov = get_project(model,tok,layer,hparams, cov_list[i], cache_c[i,:,:], cnt)
            cov_list[i] = cov

        upd_matrix = torch.linalg.solve(
                P[i,:,:].cuda() @ (layer_ks @ layer_ks.T + cache_c[i,:,:].cuda()) + hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device="cuda"), P[i,:,:].cuda() @ layer_ks @ resid.T
        )
        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))
        with torch.no_grad():
            weights[weight_name][...] = weights[weight_name] + upd_matrix
        # Clear GPU memory
        #del U,S,cov
        for x in [layer_ks, cur_zs, targets, upd_matrix]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
    for i, layer in enumerate(hparams.layers):
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        cache_c[i,:,:] += layer_ks.cpu() @ layer_ks.cpu().T

    print(f"Deltas successfully computed for {list(weights.keys())}")
    return model, cache_c


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )

def get_project(model, tok, layer, hparams, cov, new_cov, cnt):
    force_recompute = False
    # 把这个重写一下，变成更便捷的load
    if cov == None:
        cov = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            force_recompute=force_recompute,
        ).cpu()
    # 对cov累加
    if new_cov != None:
        # ipdb.set_trace()
        cov = (100000/(100000 + cnt*100))*cov + (100/(100000 + cnt*100))*new_cov
    # 加cuda的svd快5倍数
    U, S, _ = torch.linalg.svd(cov.cuda(), full_matrices=False)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (S < threshold).nonzero(as_tuple=True)[0]
    # print(len(small_singular_indices))
    return (U[:, small_singular_indices] @ U[:, small_singular_indices].T).cpu(), cov.cpu()

def get_context_templates(model, tok, lang_s): # 这一步相当于，tokens作为第一个
    global CONTEXT_TEMPLATES_CACHE_en
    global CONTEXT_TEMPLATES_CACHE_fr
    global CONTEXT_TEMPLATES_CACHE_es
    global CONTEXT_TEMPLATES_CACHE_de
    global CONTEXT_TEMPLATES_CACHE_nl

    # ============== en ==========
    if lang_s == 'en':
        if CONTEXT_TEMPLATES_CACHE_en is None:
            CONTEXT_TEMPLATES_CACHE_en = [["{}"]] + [
                [
                    f.replace("{", " ").replace("}", " ") + ". {}"
                    for f in generate_fast(
                        model,
                        tok,
                        ["The", "Therefore", "Because", "I", "You"],
                        n_gen_per_prompt=n_gen // 5,
                        max_out_len=length,
                    )
                ]
                for length, n_gen in [(10, 5)]  # Be careful about changing this.
            ]
            print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE_en}")
        return CONTEXT_TEMPLATES_CACHE_en
    # ============== de ==========
    elif lang_s == 'de':
        if CONTEXT_TEMPLATES_CACHE_de is None:
            CONTEXT_TEMPLATES_CACHE_de = [["{}"]] + [
                [
                    f.replace("{", " ").replace("}", " ") + ". {}"
                    for f in generate_fast(
                        model,
                        tok,
                        ["Heute", "Kommst", "Weil", "Den", "Der"],
                        n_gen_per_prompt=n_gen // 5,
                        max_out_len=length,
                    )
                ]
                for length, n_gen in [(10, 5)]  # Be careful about changing this.
            ]
            print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE_de}")
        return CONTEXT_TEMPLATES_CACHE_de
    # ============== nl ==========
    elif lang_s == 'nl':
        if CONTEXT_TEMPLATES_CACHE_nl is None:
            CONTEXT_TEMPLATES_CACHE_nl = [["{}"]] + [
                [
                    f.replace("{", " ").replace("}", " ") + ". {}"
                    for f in generate_fast(
                        model,
                        tok,
                        ["Wij", "Daarom", "Omdat", "Het", "Jij"],
                        n_gen_per_prompt=n_gen // 5,
                        max_out_len=length,
                    )
                ]
                for length, n_gen in [(10, 5)]  # Be careful about changing this.
            ]
            print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE_nl}")
        return CONTEXT_TEMPLATES_CACHE_nl
    elif lang_s == 'es':
        if CONTEXT_TEMPLATES_CACHE_es is None:
            CONTEXT_TEMPLATES_CACHE_es = [["{}"]] + [
                [
                    f.replace("{", " ").replace("}", " ") + ". {}"
                    for f in generate_fast(
                        model,
                        tok,
                        ["Ellos", "Por lo tanto", "Porque", "Vosotras", "Tú"],
                        n_gen_per_prompt=n_gen // 5,
                        max_out_len=length,
                    )
                ]
                for length, n_gen in [(10, 5)]  # Be careful about changing this.
            ]
            print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE_es}")
        return CONTEXT_TEMPLATES_CACHE_es
    elif lang_s == 'fr':
        if CONTEXT_TEMPLATES_CACHE_fr is None:
            CONTEXT_TEMPLATES_CACHE_fr = [["{}"]] + [
                [
                    f.replace("{", " ").replace("}", " ") + ". {}"
                    for f in generate_fast(
                        model,
                        tok,
                        ["très", "Par conséquent", "Parce que", "Je suis", "Vous"],
                        n_gen_per_prompt=n_gen // 5,
                        max_out_len=length,
                    )
                ]
                for length, n_gen in [(10, 5)]  # Be careful about changing this.
            ]
            print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE_fr}")
        return CONTEXT_TEMPLATES_CACHE_fr
