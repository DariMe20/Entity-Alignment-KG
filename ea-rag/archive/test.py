import os
import openai
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables for API keys
load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')

def initialize_langchain():
    """Initialize LangChain components for a Jupyter Notebook."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    pinecone_index_name = "dl-proj-4"
    vector_store = PineconeVectorStore(index_name=pinecone_index_name, embedding=embeddings)

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai.api_key, temperature=0.7)

    prompt_template = PromptTemplate(
        template="""
        Use the following context to identify the most similar French entity for the given English entity:
        Context: {context}
        English Entity: {question}
        Please provide the most similar French entity from the dataset, prefixed with 'FR-'.
        Answer:""",
        input_variables=["context", "question"]
    )

    llm_chain = prompt_template | llm | StrOutputParser()
    return retriever, llm_chain

def process_file(file_path):
    """Read the input file and return the modified entities for both EN and FR."""
    en_entities = []
    fr_entities = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                _, uri = parts
                modified_uri = uri.replace("http://dbpedia.org/resource/", "")
                if uri.startswith("http://dbpedia.org/resource/FR-"):
                    fr_entities.append(f"FR-{modified_uri}")
                else:
                    en_entities.append(f"EN-{modified_uri}")
    return en_entities, fr_entities

def query_llm_for_entity_pairing(en_entities, retriever, llm_chain):
    """Query the LLM to find the most similar French entity for each English entity."""
    results = {}
    for entity in en_entities:
        query = f"Which entity in the French dataset (FR-) is most similar to {entity}? Provide only the FR- prefixed entity name."
        docs = retriever.invoke(query)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Query the LLM
        answer = llm_chain.invoke({"context": context, "question": entity})
        
        # Format the results for saving
        english_uri = f"http://dbpedia.org/resource/{entity.replace('EN-', '')}"
        french_uri = f"http://fr.dbpedia.org/resource/{answer.replace('FR-', '').replace(' ', '_')}"
        results[english_uri] = french_uri
        print(f"{french_uri}\t{english_uri}")
    return results

def save_results_to_txt(results, output_file):
    """Save the alignment results to a .txt file."""
    with open(output_file, "w", encoding="utf-8") as file:
        for en_uri, fr_uri in results.items():
            file.write(f"{fr_uri}\t{en_uri}\n")
    print(f"\n✅ Results saved to {output_file}")

# === Run all steps in the same Jupyter Notebook cell ===
retriever, llm_chain = initialize_langchain()
en_entities, fr_entities = process_file("fr_en/ent_ILLs_small.txt")

# Perform the alignment and save results
results = query_llm_for_entity_pairing(en_entities, retriever, llm_chain)
save_results_to_txt(results, "fr_en/aligned_entities.txt")