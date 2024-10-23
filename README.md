# Python semantic complexity text analyzer

Contributors:
- [Rémi Venant](https://github.com/theDartagnan), main author & developer
- [Aldan Creo](https://github.com/acmcmc), documentation contributions

pysemcom (pySemanticComplexity) allows to compute **vectors** representing lexical, syntactic and **semantic complexity** for a set of given texts.

The measure of semantic complexity is based on the DBpedia Entity Recognition Graph computation, which is based on several ontologies used in DBPedia.
PyComplex provides several subroutines to process multiple files in parallel.

Most importantly, one of these subprograms is the complete pipeline of translation from texts to vectors of semantic complexity.
For each text file processed in parallel, the pipeline first cleans up the text, splits it into paragraphs, and identifies the DBpedia entity using a Spotlight REST API.
Each entity is then enriched with its types by querying a DBpedia SPARQL endpoint.
Each list of entities (for each document) is then processed in parallel to compute a concept graph composed of the entities, their types, and the hierarchy of ontology classes that define those types.
Finally, each graph is vectorized in parallel.
Note that three ontologies are used so far: Schema, DBpedia and Yago. The resulting `csv` file is composed of one line per file, each having the file name (without extension) and the different complexity features.

## 1. Repository structure
This repository is structured as follows:

- `pysemcom.py`: the main application entry point for the analyzer (usable from the command line)
- `batch`, `dpedia`, `utils`: the various python packages for the application
- `vendor`: local resources from other vendors. Only the ontologies used in DBpedia so far
- `pyComplex`: the python packages and programs for calculating the syntactic, lexical and semantic complexity of a text.
- `dbpedia-spotlight-docker': a docker-compose file for managing a DBpedia Spotlight server.
- requirements.txt': the python package requirements for the afelTraces2rdf application

## 2. Requirements
We require `python >= 3.6`.
Optionally, for syntactic and lexical parsing, you can add the Stanford parser (see section 3.1).

The required packages are listed in `requirements.txt`, and you can install them by running:

```
pip install -r requirements.txt
```

__Working within a virtual environment is recommended.__

## 3. Setup

- Go to `vendor/dbpedia/` and decompress `yago_taxonomy.ttl.bz2` into `yago_taxonomy.ttl` within the same directory
- Install the Python dependencies as described in section 2

### 3.1 Stanford tool
- Go to the `vendor/` directory and create the directory `stanford/`.
- Go to the `stanford/` directory.
- Download the following files within that directory:
	- [Lexical Complexity Analyzer](http://www.personal.psu.edu/xxl13/downloads/lca.tgz)
	- [Syntactic Complexity Analyzer](http://personal.psu.edu/xxl13/downloads/L2SCA-2016-06-30.tgz)
	- [Pos tagger](https://nlp.stanford.edu/software/stanford-postagger-full-2018-02-27.zip)
- Decompress the 3 archives (eg.: tar -xvzf lca.tgz)
- You should have the following sub-directories: 
	- `lca` (with at least `anc_all_count.txt` and `bnc_all_count.txt` files)
	- `L2SCA-2016-06-30` (with at least `standford-tregex.jar` file)
		- `stanford-parser-full-2014-01-04` (with at least `stanford-parser.jar` file)
	- `stanford-postagger-full-2018-02-27` (with at least `stanford-postagger.jar` file and `model` dir)

## 4. Use of the analyzer
The analyzer can be launched from a terminal. Inside the repository folder, you can get help by running:

```
python pysemcom.py --help
```

## 5. Licence
This package is distributed under the [Apache Licence V2](https://www.apache.org/licenses/LICENSE-2.0). Please attribute Rémi Venant through the [AFEL Project](http://afel-project.eu) when reusing and redistributing this code.
