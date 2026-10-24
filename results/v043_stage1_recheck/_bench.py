import time
from skillbench_v043.dsl import canonical_semantic_programs, family_id, semantic_mask
from skillbench_v043.generator import stable_family_split, generate_task

dev = [e for e in canonical_semantic_programs() if stable_family_split(family_id(e)) == "dev"]
print("dev targets", len(dev), "families", len({family_id(e) for e in dev}))
print("balanced targets", sum(1 for e in dev if 54 <= semantic_mask(e).bit_count() <= 9946))
t0=time.time()
t=dev[0]
r=generate_task(t, split="dev", seed=8000, support_mode="contrastive", max_attempts=20)
print("contrastive 1 task", round(time.time()-t0,2), "s", r.classification_private, r.audit_private["consistent_semantic_classes"], r.audit_private["candidate_disagreement"])
t0=time.time()
r=generate_task(t, split="dev", seed=8000, support_mode="random", max_attempts=20)
print("random 1 task", round(time.time()-t0,2), "s", r.classification_private)
