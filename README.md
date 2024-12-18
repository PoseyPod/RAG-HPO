# RAG-HPO
This is the repository for the Python based program for automated deep phenotype analysis of clinical information using large language models (LLMs) and Retrieval Augmented Generation (RAG). Plese check back frequently as we update this program to better fit the needs of clinicians and researchers. Currently, we are working on the following: 

- Full containerization of RAG-HPO to run the program without command line interaction
- Improved Error Handling

If you have feedback, suggestions, or are interested in incorporating RAG-HPO into existing pipelines, please contact Jennifer Posey (jennifer.posey@bcm.edu) or Brandon Garcia (brandon.garcia@bcm.edu).

Currently, RAG-HPO exists in two formats: a jupyter notebook and a Flask-based app that must be opened in the command line. Both versions require an API key. We used [Groq.com
](https://console.groq.com) for tests, which has a free API key and access to cloud based LLama-3.1 and other LLMs. The program should accept API keys from any cloud or local based LLM that uses the OpenAI framework. Locally stored LLMs can either be downloaded directly or run through an interface like LM-studio. 


 
![image](https://github.com/user-attachments/assets/5863d790-f887-428b-b63f-c001314143af)


**Vectorization of HPO database **
This tool processes the JSON file of Human Phenotype Ontology (HPO) data, which can be downloaded from the [HPO website](https://hpo.jax.org/data/ontology), to extract key information about HPO terms, such as their labels, definitions, synonyms, and hierarchical relationships. It then combines this data with additional validated phrases provided in a CSV file (HPO_addons.csv) to enrich the dataset. Finally, the tool vectorizes the database, making it ready for use in retrieval-augmented generation (RAG) workflows. The script is designed as a Jupyter Notebook file, so users need to have Jupyter Notebook or Microsoft Visual Studio Code installed. The script will generate a .csv file of the database prior to vectorization for users to inspect. The vectorization process should take about 10 min and can be repeated as needed to update the vector database (The HPO database updates monthly).

To run this script, users will need to make they have the following packages: json, pandas, tqdm, numpy, re, and fastembed. Users may download the requirements.txt and run the following script:

import os

def install_requirements():
    requirements_file = "requirements.txt"
    if not os.path.exists(requirements_file):
        with open(requirements_file, 'w') as f:
            f.write("pandas\n")
            f.write("numpy\n")
            f.write("tqdm\n")
            f.write("fastembed\n")
        print(f"{requirements_file} created.")
    
    print("Installing dependencies...")
    os.system(f"pip install -r {requirements_file}")
    print("All required packages installed successfully.")

if __name__ == "__main__":
    install_requirements()

As available, we will update the HPO_addons.csv file with additional phrases we find helpful in refining RAG-HPO's precision. If you would like to contribute to this file, please email us your additions in a .csv file following the format of the original file. 
