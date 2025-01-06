def read_alignment_file(file_path):
    """Read the alignment results file and return a dictionary of entity pairs."""
    alignment_results = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            fr_uri, en_uri = line.strip().split("\t")
            # Extract entity names by removing the base URL
            fr_entity = fr_uri.replace("http://fr.dbpedia.org/resource/", "")
            en_entity = en_uri.replace("http://dbpedia.org/resource/", "")
            alignment_results[en_entity] = fr_entity
    return alignment_results

def read_ground_truth_file(file_path):
    """Read the ground truth file and return a dictionary of entity pairs."""
    ground_truth = {}
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                entity_id, uri = parts
                # Detect EN and FR entities and normalize
                if uri.startswith("http://dbpedia.org/resource/FR-"):
                    fr_entity = uri.replace("http://dbpedia.org/resource/FR-", "")
                    ground_truth[entity_id] = fr_entity
                elif uri.startswith("http://dbpedia.org/resource/"):
                    en_entity = uri.replace("http://dbpedia.org/resource/", "")
                    ground_truth[entity_id] = en_entity
    return ground_truth

def compute_hits_at_1(alignment_results, ground_truth):
    """Compute the Hits@1 score."""
    hits = 0
    total_entities = len(ground_truth)

    for en_entity, predicted_fr_entity in alignment_results.items():
        # Check if there is a match in the ground truth
        if en_entity in ground_truth:
            actual_fr_entity = ground_truth[en_entity]
            if predicted_fr_entity == actual_fr_entity:
                hits += 1

    # Calculate Hits@1 score
    hits_at_1_score = hits / total_entities if total_entities > 0 else 0
    print(f"\n✅ Hits@1 Score: {hits_at_1_score:.4f}")
    return hits_at_1_score

# === Running the entire process in a single Jupyter Notebook cell ===

# File paths for the alignment results and ground truth data
alignment_results_file = "fr_en/aligned_entities.txt"  # File with predicted results
ground_truth_file = "fr_en/ent_ILLs.txt"               # File with ground truth pairs

# Read data
alignment_results = read_alignment_file(alignment_results_file)
ground_truth = read_ground_truth_file(ground_truth_file)

# Compute and display the Hits@1 score
hits_at_1_score = compute_hits_at_1(alignment_results, ground_truth)