from collections import defaultdict, deque
from pathlib import Path
import pronto

owl_path = Path("data/chebi_filtered_subset.tsv.owl")
ont = pronto.Ontology(str(owl_path))

def get_children_map(ont):
    children = defaultdict(list)
    for term in ont.terms():
        for parent in term.superclasses(distance=1, with_self=False):
            children[parent.id].append(term.id)
    return children

children_map = get_children_map(ont)

def print_tree(root_id, max_depth=2, max_children=30):
    q = deque([(root_id, 0)])
    seen = set()

    while q:
        cur, depth = q.popleft()
        if cur in seen or depth > max_depth:
            continue
        seen.add(cur)

        term = ont[cur]
        indent = "  " * depth
        print(f"{indent}{term.id} | {term.name}")

        child_ids = sorted(children_map.get(cur, []))[:max_children]
        for child_id in child_ids:
            q.append((child_id, depth + 1))

# Try obvious top level nodes if they exist
for candidate in ["CHEBI:24431", "CHEBI:50906"]:
    if candidate in ont:
        print("\n" + "=" * 80)
        print(f"TREE FROM {candidate} | {ont[candidate].name}")
        print_tree(candidate, max_depth=2)