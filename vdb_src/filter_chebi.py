from collections import defaultdict, deque
from lxml import etree

owl_infile = "data/chebi.owl"
owl_outfile = "data/chebi_subset.owl"

RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
OWL_NS = "http://www.w3.org/2002/07/owl#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"

CHEBI_IRI_PREFIX = "http://purl.obolibrary.org/obo/CHEBI_"

INCLUDE_ROOT = "CHEBI:59999"  # chemical substance

EXCLUDE_ROOTS = [
    "CHEBI:24433",   # group
    "CHEBI:33250",   # atom
    "CHEBI:50906",   # role
    "CHEBI:60004",   # mixture
    "CHEBI:46662",   # mineral
    "CHEBI:137980",  # metalloid atom
    "CHEBI:25585",   # nonmetal atom
    "CHEBI:33318",   # main group element atom
    "CHEBI:33521",   # metal atom
    "CHEBI:33559",   # s-block element atom
]

def chebi_curie_to_iri(curie: str) -> str:
    return CHEBI_IRI_PREFIX + curie.split(":", 1)[1]

owl_parser = etree.XMLParser(remove_comments=False, huge_tree=True)
owl_tree = etree.parse(owl_infile, owl_parser)
owl_root = owl_tree.getroot()

children = list(owl_root)

children_map = defaultdict(set)
all_class_iris = set()

for node in children:
    if not isinstance(node.tag, str):
        continue
    if node.tag != f"{{{OWL_NS}}}Class":
        continue

    about = node.get(f"{{{RDF_NS}}}about")
    if not about or not about.startswith(CHEBI_IRI_PREFIX):
        continue

    all_class_iris.add(about)

    for parent in node.findall(f"{{{RDFS_NS}}}subClassOf"):
        parent_iri = parent.get(f"{{{RDF_NS}}}resource")
        if parent_iri and parent_iri.startswith(CHEBI_IRI_PREFIX):
            children_map[parent_iri].add(about)

print("Total CHEBI classes found:", len(all_class_iris))

def get_descendants(root_iri: str) -> set[str]:
    seen = set()
    q = deque([root_iri])
    while q:
        cur = q.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        for child in children_map.get(cur, ()):
            q.append(child)
    return seen

include_root_iri = chebi_curie_to_iri(INCLUDE_ROOT)
keep_iris = get_descendants(include_root_iri)

exclude_iris = set()
for curie in EXCLUDE_ROOTS:
    exclude_iris |= get_descendants(chebi_curie_to_iri(curie))

keep_iris -= exclude_iris

print("Initial descendants from include root:", len(get_descendants(include_root_iri)))
print("Excluded descendants total:", len(exclude_iris))
print("Final class IRIs to keep:", len(keep_iris))

new_root = etree.Element(owl_root.tag, nsmap=owl_root.nsmap)

kept_blocks = 0
for node in children:
    if not isinstance(node.tag, str):
        continue

    if node.tag == f"{{{OWL_NS}}}Ontology":
        new_root.append(node)
        continue

    if node.tag != f"{{{OWL_NS}}}Class":
        continue

    about = node.get(f"{{{RDF_NS}}}about")
    if about in keep_iris:
        new_root.append(node)
        kept_blocks += 1

new_tree = etree.ElementTree(new_root)
new_tree.write(
    owl_outfile,
    encoding="utf-8",
    xml_declaration=True,
    pretty_print=True,
)

print("Kept OWL class blocks:", kept_blocks)
print("DONE:", owl_outfile)