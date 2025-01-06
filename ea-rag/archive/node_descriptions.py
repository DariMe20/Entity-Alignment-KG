from rdflib import Graph
import os

def create_merged_graph(relation_file_path, attribute_file_path):
    """Creates and returns an RDF graph by merging relation and attribute files."""
    graph = Graph()
    graph.parse(relation_file_path, format="nt")
    graph.parse(attribute_file_path, format="nt")

    print(f"Merged graph created with {len(graph)} triples.")
    return graph

def format_triples_for_embedding(graph, entity_uri):
    """
    Formats RDF triples where the given entity is a subject or object into a readable format for embeddings.

    Parameters:
    - graph: RDFLib Graph object
    - entity_uri: URI of the entity to query for

    Returns:
    - formatted_text: A single string containing all triples where the entity is subject or object, ready for embedding generation.
    """

    # Extract the entity label from the URI
    entity_label = entity_uri.split('/')[-1]

    def safe_split(uri):
        """Replaces None with the current entity label and extracts labels from URIs."""
        if uri is None:
            return entity_label
        return uri.split('/')[-1] if '/' in str(uri) else str(uri)

    # Prepare SPARQL queries for both subject and object positions
    query_subject = f"""
    SELECT ?s ?p ?o
    WHERE {{
        <{entity_uri}> ?p ?o .
    }}
    """
    
    query_object = f"""
    SELECT ?s ?p ?o
    WHERE {{
        ?s ?p <{entity_uri}> .
    }}
    """

    # Execute the queries
    subject_results = graph.query(query_subject)
    object_results = graph.query(query_object)

    # Prepare the formatted text for embeddings
    formatted_text = []

    # Format triples where the entity is the subject
    formatted_text.append(f"# Triples where '{entity_label}' is the subject:\n")
    for s, p, o in subject_results:
        formatted_text.append(f"{safe_split(s)} {safe_split(p)} {safe_split(o)}.")

    # Format triples where the entity is the object
    formatted_text.append(f"\n# Triples where '{entity_label}' is the object:\n")
    for s, p, o in object_results:
        formatted_text.append(f"{safe_split(s)} {safe_split(p)} {safe_split(o)}.")

    # Combine all triples into a single formatted string
    formatted_text = "\n".join(formatted_text)
    return formatted_text

def describe_node_for_embedding_per_subject(graph, output_folder):
    """Extracts all attributes and relations for every node and saves them as text files."""
    os.makedirs(output_folder, exist_ok=True)
    for subject in set(graph.subjects()):
        subject_label = subject.split('/')[-1]
        subject_file_path = os.path.join(output_folder, f"{subject_label}.txt")

        # Use the new function to generate formatted triples for the subject
        formatted_text = format_triples_for_embedding(graph, subject)

        # Write the formatted data to the file
        with open(subject_file_path, "w", encoding="utf-8") as outfile:
            outfile.write(formatted_text)
    print(f"Files saved in '{output_folder}'.")

def process_multiple_datasets(datasets):
    """Processes multiple RDF datasets and saves results for each."""
    for relation_file, attribute_file, output_folder in datasets:
        graph = create_merged_graph(relation_file, attribute_file)
        describe_node_for_embedding_per_subject(graph, output_folder)

# List of datasets including both relation and attribute triples
datasets_to_process = [
    ("fr_en/en_rel_triples_preprocessed", "fr_en/en_att_triples_preprocessed", "fr_en/en_combined_triples_folder2"),
    ("fr_en/fr_rel_triples_preprocessed", "fr_en/fr_att_triples_preprocessed", "fr_en/fr_combined_triples_folder2")
]

# Run batch processing
process_multiple_datasets(datasets_to_process)