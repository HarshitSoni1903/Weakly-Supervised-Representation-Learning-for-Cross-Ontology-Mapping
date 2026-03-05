from wrapper import batch_retrieve_to_csv

inputs = [
    # {"id": "HP:0001627"},  # id-only (will fill from hp collection)
    {"label": "Abortion, Septic", "definition": "", "synonyms": []},  # label-only
    # {
    #     "label": "Abnormal heart morphology",
    #     "definition": "Any structural anomaly of the heart.",
    #     "synonyms": ["Cardiac malformation"],
    # },
]

out = batch_retrieve_to_csv(inputs, tgt_collection="mesh", out_csv="batch_results.csv", top_k=1)
batch_retrieve_to_csv([{"id":"MONDO:0000001"}], tgt_collection="mesh", out_csv="tmp.csv", top_k=50, query_mode="full_src")
print(out)