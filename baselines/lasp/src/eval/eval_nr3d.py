"""Inherited LaSP Nr3D scoring with repository split, dump and tie bookkeeping."""

from copy import deepcopy
import json
from json import JSONDecodeError
import os
from pathlib import Path
from PIL import Image

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from openai import OpenAI

from grounding.resolution.evaluate import count_result, row_key as _row_key
from src.relation_encoders.compute_features import ALL_VALID_RELATIONS, rel_num
from src.dataset.datasets import Nr3DDataset
from grounding._vendor.lasp.vlm_utils import resize_image_to_GPT_size, encode_PIL_image_to_base64, user_prompt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Global Configurations & Splits ---
BENCHMARK_SPLITS_PATH = Path("benchmark") / "splits.json"
DEFAULT_PARSES_PATH = Path("data/symbolic_exp/nr3d.jsonl")

_CLIENT = None

OFFICIAL_TOTALS = {
    "easy": 3669, 
    "hard": 3816, 
    "view_dependent": 2478,
    "view_independent": 5007, 
    "overall": 7485
}

# The shared benchmark strata tables, one per evaluated split. Row indices are
# positions in that split's own csv and the ranges overlap, so the table has to
# follow --split: the test table answers a dev index with another utterance's
# flags. A split with no table scores no strata, which is the pre-existing
# fallback for probe rows.
_SPLITS_BY_SPLIT = {}
_SPLIT = "test"


def _splits_table(split):
    if split not in _SPLITS_BY_SPLIT:
        name = "splits.json" if split == "test" else f"splits_{split}.json"
        path = BENCHMARK_SPLITS_PATH.with_name(name)
        _SPLITS_BY_SPLIT[split] = (json.loads(path.read_text())
                                   if path.exists() else {})
    return _SPLITS_BY_SPLIT[split]


def _split_flags(dataset_idx):
    e = _splits_table(_SPLIT).get(str(int(dataset_idx)))
    if e is None:
        return False, False
    return e["is_hard"], e["is_view_dependent"]




def get_client():
    """Build the OpenAI client lazily."""
    global _CLIENT
    if _CLIENT is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        _CLIENT = OpenAI(api_key=api_key)
    return _CLIENT


def parse(scan_id, json_obj, all_concepts, valid_ids):
    final_concept = torch.ones((len(valid_ids),), device=DEVICE)
    
    if json_obj["category"] in ["corner", "middle", "room", "center"]:
        return final_concept
    if json_obj["category"] in ALL_VALID_RELATIONS:
        raise NotImplementedError

    appearance_concept = all_concepts[json_obj["category"]][valid_ids]
    final_concept = torch.minimum(final_concept, appearance_concept)

    if "relations" in json_obj:
        for relation_item in json_obj["relations"]:
            if "anchors" in relation_item:
                relation_item["objects"] = relation_item["anchors"] 
            
            relation_name = relation_item["relation_name"]
            if relation_name not in ALL_VALID_RELATIONS:
                continue
            
            relation_concept = all_concepts[relation_name]
            num = rel_num[relation_name]
            concept = torch.ones((len(valid_ids),), device=DEVICE)

            if num == 1:
                relation_concept = relation_concept[valid_ids, :][:, valid_ids]
                if len(relation_item["objects"]) >= 1:
                    sub_concept = parse(scan_id, relation_item["objects"][0], all_concepts, valid_ids)
                else:
                    obj = deepcopy(json_obj)
                    obj["relations"] = []
                    obj.pop("color", None)
                    obj.pop("shape", None)
                    sub_concept = parse(scan_id, obj, all_concepts, valid_ids)
                concept = concept * (relation_concept @ sub_concept)

            elif num == 0:
                concept = relation_concept[valid_ids]

            elif num == 2:
                relation_concept = relation_concept[valid_ids, :, :][:, valid_ids, :][:, :, valid_ids]
                objects = relation_item["objects"]
                if len(objects) == 0:
                    obj = deepcopy(json_obj)
                    obj["relations"] = []
                    sub_concept = parse(scan_id, obj, all_concepts, valid_ids)
                    concept = torch.einsum('ijk,j,k->i', relation_concept, sub_concept, sub_concept)
                elif len(objects) == 1:
                    sub_concept = parse(scan_id, objects[0], all_concepts, valid_ids)
                    concept = torch.einsum('ijk,j,k->i', relation_concept, sub_concept, sub_concept)
                elif len(objects) >= 2:
                    # `>= 2`, not `== 2`: a `between` carrying more than two
                    # anchors is truncated to its first two. This arithmetic is
                    # upstream and unchanged; the comment is here because the
                    # extended arm now states the SAME policy explicitly at
                    # src/grounding/resolution/resolver.py, where it used to
                    # refuse such a row instead and drop it from the comparison.
                    sub_concept_1 = parse(scan_id, objects[0], all_concepts, valid_ids)
                    sub_concept_2 = parse(scan_id, objects[1], all_concepts, valid_ids)
                    concept = torch.einsum('ijk,j,k->i', relation_concept, sub_concept_1, sub_concept_2)
            else:
                raise NotImplementedError(f"Unsupported relation num: {num}")

            concept = F.softmax(concept, dim=0)
            if relation_item.get("negative") is True:
                concept = concept.max() - concept

            final_concept = final_concept * concept

    return final_concept


def thesis_parse(norm_mode):
    """The thesis resolver over LaSP's own vocabulary and feature tensors.

    Opt-in (``--executor thesis``) for the representation ablation: the scene
    representation is LaSP's (its encoders, its predicate table, no colour
    keys, no ordinal rank), the executor is the extended tree's. Rank is
    switched off because ordinal scoring is one of the representation additions;
    colour drops out because LaSP's feature tensors carry no ``color:`` keys and
    the resolver skips a colour it has no feature for. With
    ``norm_mode="softmax"`` this reproduces ``parse`` above exactly
    (tests/test_executor_equivalence.py).
    """
    from grounding.resolution.config import DEFAULT_RESOLVER
    from grounding.resolution.resolver import parse as resolver_parse

    config = DEFAULT_RESOLVER.updated(norm_mode=norm_mode, use_rank=False)

    def scorer(scan_id, json_obj, all_concepts, valid_ids):
        return resolver_parse(scan_id, json_obj, all_concepts, valid_ids,
                              rel_num=rel_num, valid_relations=ALL_VALID_RELATIONS,
                              config=config)
    return scorer, config


def query_vlm(scan_id, caption, filtered_candidates):
    image_root = Path('data/frames')
    masks_path = Path("data/nr3d_masks") / scan_id
    if not masks_path.exists():
        return -1

    merged_indices = {}
    for obj_dir in os.listdir(masks_path):
        obj_id = int(obj_dir.split("_")[0])
        if obj_id not in filtered_candidates:
            continue
        indices_file = masks_path / obj_dir / "indices.npz"
        if not indices_file.exists():
            continue
        indices = np.load(indices_file)
        for img_name in indices.keys():
            if img_name not in merged_indices:
                merged_indices[img_name] = {}
            merged_indices[img_name][obj_id] = indices[img_name]

    if not merged_indices:
        return -1

    # Calculate aggregate areas to select top images
    merged_areas = {}
    for img_name, obj_dict in merged_indices.items():
        area = sum(
            (inds[1].max() - inds[1].min()) * (inds[0].max() - inds[0].min())
            for inds in obj_dict.values()
        )
        merged_areas[img_name] = area

    sorted_images = sorted(merged_areas.items(), key=lambda x: x[1], reverse=True)
    top8_images = sorted_images[:8]
    selected_candidates = {obj_id for img_name, _ in top8_images for obj_id in merged_indices[img_name]}

    # Fill remaining slots if all objects aren't covered
    for obj_id in filtered_candidates:
        if obj_id not in selected_candidates and len(top8_images) < 8:
            for img_name, _ in sorted_images:
                if obj_id in merged_indices[img_name] and (img_name, merged_areas[img_name]) not in top8_images:
                    top8_images.append((img_name, merged_areas[img_name]))
                    break

    sample_img_path = image_root / scan_id / "color" / top8_images[0][0]
    if not sample_img_path.exists():
        return -1

    sample_img = cv2.imread(str(sample_img_path))
    single_img_size = sample_img.shape[:2]
    
    stitched_img = np.ones((single_img_size[0] * 2, single_img_size[1] * 4, 3), dtype=np.uint8) * 255
    for idx, (img_name, _) in enumerate(top8_images[:8]):
        r, c = divmod(idx, 4)
        img_path = image_root / scan_id / "color" / img_name
        img = cv2.imread(str(img_path)) if img_path.exists() else np.zeros_like(sample_img)
        
        for obj_id, inds in merged_indices[img_name].items():
            pt = (int((inds[1].max() + inds[1].min()) / 2), int((inds[0].max() + inds[0].min()) / 2))
            cv2.putText(img, f"obj_{obj_id}", pt, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        stitched_img[r * single_img_size[0]:(r + 1) * single_img_size[0], c * single_img_size[1]:(c + 1) * single_img_size[1]] = img

    stitched_img = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2RGB)
    from skimage import img_as_ubyte  # only the VLM path needs scikit-image
    pil_image = resize_image_to_GPT_size(Image.fromarray(img_as_ubyte(stitched_img)))
    base64_frame = encode_PIL_image_to_base64(pil_image)

    messages = [
        {
            "role": "system",
            "content": "You are good at finding objects specified by a description in indoor rooms by watching the videos scanning the rooms."
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt.format(utterance=caption, candidates=str(filtered_candidates))},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_frame}", "detail": "high"}}
            ]
        }
    ]

    try:
        client = get_client()
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",
            messages=messages,
            top_p=0.3,
            temperature=0.1,
        ).choices[0].message.content

        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        return int(json.loads(response)['object id'])
    except (JSONDecodeError, KeyError, Exception):
        return -1


def eval_nr3d(args):
    global _SPLIT
    _SPLIT = getattr(args, "split", "test")
    dataset = Nr3DDataset(split=_SPLIT, label_type=args.label_type,
                          load_points=False)
    scorer = parse
    if getattr(args, "executor", "lasp") == "thesis":
        scorer, config = thesis_parse(args.norm_mode)
        print(f"executor: thesis resolver on LaSP features ({config})")
    all_features = torch.load(args.features_path, map_location=DEVICE, weights_only=False)
    
    totals_mode = args.totals
    result = {
        k: {"correct": 0, "total": v if totals_mode == "official" else 0}
        for k, v in OFFICIAL_TOTALS.items()
    }

    def _count_attempt(dataset_idx):
        if totals_mode != "parses":
            return
        is_hard, is_vd = _split_flags(dataset_idx)
        count_result(result, "total", is_hard, is_vd)

    scored = 0
    skipped = 0
    dump_file = open(args.dump, "w") if args.dump else None
    parses_file = Path(args.parses)

    if not parses_file.exists():
        raise FileNotFoundError(f"Parses file not found: {parses_file}")

    with open(parses_file, "r") as f:
        for line in tqdm(f):
            data = json.loads(line)
            dataset_idx = _row_key(data)
            _count_attempt(dataset_idx)
            
            scan_id = data["scan_id"]
            if scan_id not in all_features or data.get("json_obj") is None:
                skipped += 1
                continue

            json_obj = data["json_obj"]
            target_obj_id = data["target_id"]
            sentence = data.get("sentence", data.get("utterance", ""))
            is_hard, is_view_dependent = _split_flags(dataset_idx)
            
            features_this_scene = all_features[scan_id]
            scan_data = dataset.scans.get(scan_id)
            if not scan_data:
                skipped += 1
                continue

            all_locations = scan_data["pred_locs"]
            valid_ids = torch.arange(len(all_locations), device=DEVICE)
            
            try:
                final_concept = scorer(scan_id, json_obj, features_this_scene, valid_ids)
            except (NotImplementedError, ValueError, RuntimeError):
                skipped += 1
                continue
            
            scored += 1

            if args.tie_break == "legacy":
                ranking = torch.topk(final_concept, min(args.top_k, len(final_concept))).indices
            else:
                ranking = torch.argsort(final_concept, descending=True, stable=True)
            
            ranking[0].item()  # Preserve validation of a nonempty scalar ranking.
            top_k_ids = ranking[:min(args.top_k, len(final_concept))]
            obj_ids = scan_data["obj_ids"]
            top_k_candidates = [obj_ids[i] for i in top_k_ids.tolist()]

            if dump_file is not None:
                _mx = float(final_concept.max())
                _tied = int((final_concept == _mx).sum())
                dump_file.write(json.dumps({
                    "dataset_idx": int(dataset_idx),
                    "scan_id": scan_id,
                    "target_id": int(target_obj_id),
                    "pred": int(top_k_candidates[0]),
                    "correct": bool(top_k_candidates[0] == target_obj_id),
                    "n_tied_at_max": _tied,
                    "n_objects": int(len(final_concept)),
                    "argmax_idx": int(torch.argmax(final_concept)),
                    "sel_idx": int(top_k_ids[0]),
                }) + "\n")

            logits = final_concept[top_k_ids]
            if logits.max() > 0:
                logits = logits / logits.max()

            def update_metrics(is_correct):
                if is_correct:
                    count_result(result, "correct", is_hard, is_view_dependent)

            if args.use_vlm:
                filtered_candidates = [
                    cand for cand, logit in zip(top_k_candidates, logits) if logit > args.threshold
                ]
                if all(c == target_obj_id for c in filtered_candidates) and filtered_candidates:
                    update_metrics(True)
                    continue
                if all(c != target_obj_id for c in filtered_candidates) and filtered_candidates:
                    continue
                
                vlm_ans = query_vlm(scan_id, sentence, filtered_candidates)
                if vlm_ans == target_obj_id:
                    update_metrics(True)
            else:
                if top_k_candidates[0] == target_obj_id:
                    update_metrics(True)

    if dump_file is not None:
        dump_file.close()

    print(f"Scored {scored} utterances, skipped {skipped} (totals mode: {totals_mode})")
    for k, v in result.items():
        if v["total"] > 0:
            print(f"{k}: {v['correct']} / {v['total']} ({v['correct'] / v['total']:.4f})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate 3D Visual Grounding Parser.")
    parser.add_argument("--features_path", type=str, required=True, help="Path to precomputed feature tensors.")
    parser.add_argument("--top_k", type=int, default=5, help="Top K candidates to evaluate.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for VLM filtering.")
    parser.add_argument("--use_vlm", action="store_true", help="Enable GPT-4o VLM verification.")
    parser.add_argument("--label_type", choices=["pred", "gt"], default="pred", type=str)
    parser.add_argument("--split", choices=["test", "dev", "train"], default="test", help="Which annotation csv the scene loader reads; features and parses must be on the same split.")
    parser.add_argument("--parses", type=str, default=str(DEFAULT_PARSES_PATH), help="Path to parses JSONL file.")
    parser.add_argument("--dump", type=str, default=None, help="Write per-utterance outcomes to file.")
    parser.add_argument("--totals", choices=["official", "parses"], default="official", help="Denominator mode.")
    parser.add_argument("--tie_break", choices=["stable", "legacy"], default="stable", help="Tie-breaking method.")
    parser.add_argument("--executor", choices=["lasp", "thesis"], default="lasp",
                        help="lasp (default): the upstream parse() above. thesis: "
                             "the extended resolver over these LaSP features, for "
                             "the representation ablation; see thesis_parse")
    parser.add_argument("--norm_mode", choices=["softmax", "zscore", "zscore_sigmoid"],
                        default="zscore_sigmoid",
                        help="with --executor thesis only: the resolver's "
                             "per-relation normalisation")
    
    args = parser.parse_args()
    eval_nr3d(args)
