# News Summarizer PoC

The service to summarize collected information for a period of time.
(Actually, this is only a Proof-of-Concept for now!)

## The problem

There are lots of information sources that provide tons of news, messages etc. 
There's not enough time to read them all, but FOMO cries.
Lots of tech bloggers post their own summaries - they are cool but sometimes relate to some specific domain or overview only most important news.
To be not in the bubble, it would be good to have an overview of news from different domains.

## The solution

The Proof-of-Concept that wrangles posts from HackerNews and Lobste.rs, 
and provides an overview of posts somehow related to the query. 
If needed, within some period of time.   

## Technical details

* RAG flow
  * data from HN and Lobste.rs (along with webpages) is stored in json files along with . That's not suitable for production, but OK for the PoC. 
  * [Milvus](https://milvus.io/) is used as a vector database. It was chosen as the vector database because lite version is quite simple in installation and quite functional, 
       but does not require separate heavy services. As a more powerful alternative, it's possible to use Elasticsearch, Weaviate or even pgvector (if using postgres).   
  * OpenAI GPT is used as LLM 
* Retrieval evaluation approach is implemented but only one way is scored.
* No evaluation of RAG is provided
* User Interface is implemented in scope of a separate Jupyter notebook 
* Ingestion pipeline
  * A dedicated Jupyter notebook allows to semi-automatically ingest data
* No specific monitoring as for now 
* No containerization as for now - initially I tested docker setup, but the container was huge due to embedded models.
  I decided to skip that for now. This PoC is kind a self-sufficient - plaintext files can be replaced by sqLite DB and do not require side services. 

## Project structure

Below is the working project structure (not all items are committed, also, milvus lock files are not listed): 

- `.env` - env variables. e.g. OpenAI API key
- `.env.template` - env variables template
- `pdm.lock` - lock file for project dependencies 
- `pyproject.toml` - project dependencies (PDM is used as a package manager)
- `notebooks/`
  - `embeddings.py` - embedding-related functionality
  - `hn_news.json` - HackerNews data
  - `llm.py` - LLM-related functionality
  - `lr_news.json` - Lobste.rs data
  - `milvus_summarizer.db` - Milvus database with actual data
  - `milvus_summarizer_eval.db` - Milvus database with data subset for evaluation
  - `summarizer-evaluate-poc.ipynb` - notebook to evaluate retrieval
  - `summarizer-ingest-poc.ipynb` - notebook to ingest data
  - `summarizer-ui-poc.ipynb` - notebook to provide a user interface
  - `summarizer_eval_subset.json` - subset of data for evaluation
  - `summarizer_ground_truth.json` - ground truth for evaluation

## Installation

### Requirements

* python 3.11+
* [PDM](https://pdm-project.org/)
* OpenAI API key
* about 9GB of free disk space 

### Steps

1. Clone the repository
2. `pdm install` - to install all groups from lock file
3. Attach interpreter to Jupyter notebooks.
4. Copy `.env.template` to `.env` and fill in the OpenAI API key

## Usage

* Run `summarizer-ingest-poc.ipynb` to ingest data. It can be run periodically, new data will be appended to existing one.
* Run `summarizer-evaluate-poc.ipynb` to evaluate retrieval. Currently it's only for reference.
* Run `summarizer-ui-poc.ipynb` to see how the output looks like. 
  After the first run (initialization) it's possible to run only cells after "The Working Example" heading.

Data files (`.json`) and vector database files (`.db`, `.db.lock`) can be deleted, they will be recreated during the next run 
(for that it's needed to run the ingestion notebook first).

I had several issues with Milvus Lite, a workaround is used to make it run in multi-notebook mode. 

## Further work items

* Add an embedded database for storing data (e.g. SQLite)
* Dockerize the project
* Set up ingestion pipeline
* Set up meaningful monitoring
