import pyobo
import os
owl_path = os.path.abspath("/Users/hsoni/Downloads/New_Caps/data/mesh.owl")
obo = pyobo.get_ontology("mesh")       
obo.write_owl(owl_path)