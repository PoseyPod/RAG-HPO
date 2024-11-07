# RAG-HPO
This is the repository for the Python based program for automated deep phenotype analysis of clinical information using large language models (LLMs) and Retrieval Augmented Generation (RAG). Plese check back frequently as we update this program to better fit the needs of clinicians and researchers. Currently, we are working on the following: 

- Full containerization of RAG-HPO to run the program without command line interaction
- Improved Error Handling

If you have feedback, suggestions, or are interested in incorporating RAG-HPO into existing pipelines, please contact Jennifer Posey (jennifer.posey@bcm.edu) or Brandon Garcia (brandon.garcia@bcm.edu).

Currently, RAG-HPO exists in two formats: a jupyter notebook and a Flask-based app that must be opened in the command line. Both versions require an API key. We used [Groq.com
](https://console.groq.com) for tests, which has a free API key and access to cloud based LLama-3.1 and other LLMs. The program should accept API keys from any cloud or local based LLM that uses the OpenAI framework. Locally stored LLMs can either be downloaded directly or run through an interface like LM-studio. 



