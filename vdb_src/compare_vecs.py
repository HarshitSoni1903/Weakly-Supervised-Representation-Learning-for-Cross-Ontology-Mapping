"""Compare ablation approach (fresh embed) vs mapper approach (reconstruct from FAISS)"""
from config import BuildConfig
from utils import (load_collection, load_encoder, resolve_device, 
                   embed_texts, build_embedding_text)
import numpy as np

cfg = BuildConfig()
device = resolve_device(cfg.device)
mondo = load_collection(cfg, "mondo")
mesh = load_collection(cfg, "mesh")

tok, mdl = load_encoder(cfg.ft_model_path, device)

# pick a few mondo concepts
test_ids = ["mondo:0005015", "mondo:0800175", "mondo:0015469"]
for cid in test_ids:
    payload = mondo.get_payload_by_id(cid)
    if not payload:
        continue
    
    # ablation approach: build text, embed fresh
    qtext = build_embedding_text(
        payload["label"], payload.get("definition", ""), 
        payload.get("synonyms", []), cfg.synonym_cap
    )
    fresh_vec = embed_texts([qtext], tok, mdl, device, cfg.max_length)[0]
    
    # mapper approach: reconstruct from FAISS
    pos = mondo.id2pos[cid]
    stored_vec = mondo.index.reconstruct(pos)
    
    sim = float(np.dot(fresh_vec, stored_vec))
    diff = np.abs(fresh_vec - stored_vec).max()
    
    # search mesh with both vectors
    scores_fresh, idxs_fresh = mesh.index.search(fresh_vec.reshape(1, -1), 5)
    scores_stored, idxs_stored = mesh.index.search(stored_vec.reshape(1, -1), 5)
    
    print(f"{cid}: {payload['label'][:40]}")
    print(f"  fresh vs stored sim: {sim:.6f}, max_diff: {diff:.6f}")
    print(f"  fresh  top5: {[(mesh.id_at_pos(int(ix)), round(float(s), 4)) for s, ix in zip(scores_fresh[0], idxs_fresh[0])]}")
    print(f"  stored top5: {[(mesh.id_at_pos(int(ix)), round(float(s), 4)) for s, ix in zip(scores_stored[0], idxs_stored[0])]}")
    print(f"  same top1: {idxs_fresh[0][0] == idxs_stored[0][0]}")
    print()
