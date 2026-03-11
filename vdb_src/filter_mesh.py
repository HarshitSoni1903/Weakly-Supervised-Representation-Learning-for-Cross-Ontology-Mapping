from lxml import etree

owl_infile = "data/mesh.owl"
xml_infile = "data/desc2026.xml"
owl_outfile = "data/mesh_disease_C.owl"

RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
OWL_NS = "http://www.w3.org/2002/07/owl#"

keep_iris = set()

xml_parser = etree.XMLParser(remove_comments=False, huge_tree=True)
xml_tree = etree.parse(xml_infile, xml_parser)
xml_root = xml_tree.getroot()

for record in xml_root.findall(".//DescriptorRecord"):
    ui = record.findtext("DescriptorUI")
    if not ui:
        continue

    tree_numbers = [tn.text for tn in record.findall(".//TreeNumber") if tn.text]
    if any(tn.startswith("C") for tn in tree_numbers):
        keep_iris.add(f"http://id.nlm.nih.gov/mesh/{ui}")

print("Disease descriptor IRIs to keep:", len(keep_iris))

owl_parser = etree.XMLParser(remove_comments=False, huge_tree=True)
owl_tree = etree.parse(owl_infile, owl_parser)
owl_root = owl_tree.getroot()

new_root = etree.Element(owl_root.tag, nsmap=owl_root.nsmap)
children = list(owl_root)

i = 0
kept_blocks = 0

while i < len(children):
    node = children[i]

    if node.tag == f"{{{OWL_NS}}}Ontology":
        new_root.append(node)
        i += 1
        continue

    if isinstance(node, etree._Comment):
        if i + 1 < len(children):
            nxt = children[i + 1]
            about = nxt.get(f"{{{RDF_NS}}}about")
            if about in keep_iris:
                new_root.append(node)
                new_root.append(nxt)
                kept_blocks += 1
        i += 2
        continue

    about = node.get(f"{{{RDF_NS}}}about")
    if about in keep_iris:
        new_root.append(node)
        kept_blocks += 1

    i += 1

new_tree = etree.ElementTree(new_root)
new_tree.write(
    owl_outfile,
    encoding="utf-8",
    xml_declaration=True,
    pretty_print=True,
)

print("Kept OWL blocks:", kept_blocks)
print("DONE:", owl_outfile)