# python semantic complexity text analyzer
Contributors: Rémi Venant

pysemcom (pySemanticComplexity) allow to compute from a bunch of different text files lexical,
syntactical and semantic complexity as vectors for each of these texts.

Semantic complexity relies on DBpedia Entity Recognition Graph computation based on multiple ontologies used in DBPedia.
PyComplex offers several subprograms to process multiples files in parallels.

One of these subprogram is the full pipeline of translation from texts to vectors of semantic complexity.
For each text file processed in parallel, the pipeline first clean and split text in paragraphs,
and identify the DBpedia entity with the use of a Spotlight REST Api. Each entity is then enrich with its types by
queriyng a DBpedia SPARQL endpoint. Each list of entities (for each document) is then processed in parallel to compute
a graph of concept, conposed of the entities, their types and the hierachy of ontology classes that define these types.
Finally each graph is vectorized in parallel. Note that three ontologies are used so far: Schema, DBpedia ontology and
Yago. The result csv file is composed one line per fil, each of them being composed of the file name (without its
extension) and the several complexity features.


## 1. Repository structure
This respository is structed as follow:

- pysemcom.py: the main application entrypoint for the analyzer (usable in command line)
- batch, dpedia, utils: the different python packages for the application
- vendor: local resources from other providers. Only the ontologies used in DBpedia so far
- pyComplex: the python packages and programs to compute syntactic, lexical and semantic complexity of a text.
- dbpedia-spotlight-docker: a docker-compose file to manage a DBpedia Spotlight server
- requirements.txt: the python package requirement for the afelTraces2rdf application
- README.md: this file

## 2. Requirements
The afelTraces2rdf application relies on the following softwares:

- python >= 3.6
- pip

The required packages for the application are listed in the requirements.txt file.
An automatic installation of the packages can be achieved by the following command, to be executed within the repository folder:

    pip install -r requirements.txt

__Working with a virtual environment is recommended.__

## 4. Setup

- Go to vendor/dbpedia and decompress yago_taxonomy.ttl.bz2 into yago_taxonomy.ttl within the same directory
- Install the python dependencies if it is not already done (see section 2. of this document)

## 3. Use of the analyzer
The analyzer can be launched from a terminal. Inside the repository folder, one can run the following command to get the help:

    python pysemcom.py --help

## 4. Licence
The python text complexity analyzer is distributed under the [Apache Licence V2](https://www.apache.org/licenses/LICENSE-2.0). Please attribute Rémi Venant through the [AFEL Project](http://afel-project.eu)* when reusing and redistributing this code.