from wrapper import batch_retrieve_to_csv

inputs = [
    {"id": "HP:0001627"},  # id-only (will fill from hp collection)
    {"label": "Abnormal heart morphology", "definition": "", "synonyms": []},  # label-only
    {
        "label": "Abnormal heart morphology",
        "definition": "Any structural anomaly of the heart.",
        "synonyms": ["Cardiac malformation"],
    },
]

out = batch_retrieve_to_csv(inputs, tgt_collection="mp", out_csv="batch_results.csv", top_k=50)
print(out)