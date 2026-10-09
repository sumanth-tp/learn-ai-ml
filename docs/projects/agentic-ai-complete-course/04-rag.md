---
id: agentic-course-rag
title: "04. RAG From Scratch: Loaders, Chunking, Embeddings, Vector Stores and a Modular Pipeline (Complete Agentic AI Course in 10 Hours)"
sidebar_label: "4 - RAG"
sidebar_position: 4
slug: /projects/agentic-ai-complete-course/rag
description:
  "Build retrieval augmented generation from first principles: why RAG exists, the LangChain Document, loaders, chunking, MiniLM embeddings, ChromaDB and FAISS vector stores, a retriever, Groq-powered answers, and a modular src/ pipeline."
tags:
  [
    agentic-ai,
    langchain,
    rag,
    document-loaders,
    chunking,
    embeddings,
    chromadb,
    faiss,
    groq,
  ]
---

import Infographic from '@site/src/components/Infographic';

> **Part 4 of 9** ·
> [Watch on YouTube](https://www.youtube.com/watch?v=rV3HJ4LEZ7k) ·
> Notebooks and files from the RAG tutorial folder:
> `notebook/document.ipynb`, `notebook/pdf_loader.ipynb`,
> `notebook/1-langchain-document-components.svg`, `src/data_loader.py`,
> `src/embedding.py`, `src/vectorstore.py`, `src/search.py` and `app.py`. The
> repository also holds `agenticrag/1-agenticrag.ipynb`, `typesense.ipynb` and
> `books.jsonl`; the instructor does not teach those in this section, so they
> appear at the end under a clearly marked heading. Notes follow the video in
> order.

This chapter takes you from the question "why does an LLM need a knowledge base?" to a working retrieval augmented generation (RAG) system: you load PDFs and text files into LangChain `Document` objects, cut them into chunks, turn the chunks into vectors, store the vectors, retrieve the best matches for a question, and hand them to a Groq-hosted model that writes the answer. You build it twice: first as a notebook, then as a small modular Python package you can reuse.

## What this section builds

The instructor opens by listing what the next two hours cover. You will see the whole path a document travels: **data ingestion**, then the **retrieval pipeline**, then **output generation**. Along the way you will use an LLM and an embedding model, and you will meet the question that decides how good a RAG system is, namely how to chunk the data. He promises both sides of the subject, the theory and the code.

He also describes the teaching order. First comes the basic implementation, written line by line in a notebook so each idea is visible. Later, in the "advanced" part, the same logic is rewritten as **modular code**: small classes in separate files that are linked into one pipeline. The point of the modular version is not style. It is to let you see how the stages connect so that you can lift the pattern into a real company project.

He makes two remarks that are worth keeping in perspective. First, he says that most LLM use cases being built inside companies today are RAG use cases, so this crash course is aimed at the most commercially common pattern. That is his estimate, not a measured figure, but the direction is fair: question answering over private documents is one of the most requested applications. Second, he sets a light-hearted target of a thousand likes and five hundred comments for the video. It is an aside, not part of the technical material.

## What RAG is

Before drawing anything he reads a short definition from a slide. Put in plain words, it says:

- RAG is a way of improving the output of a large language model by letting it consult an **authoritative knowledge base that lives outside its training data** before it writes a response.
- LLMs are trained on huge volumes of text and have billions of parameters, which is why they can answer questions, translate and finish sentences without any help.
- RAG **extends** that ability to a specific domain, or to an organisation's own internal knowledge, **without retraining the model**.
- Because it does not need retraining, it is a cost-effective way to keep answers relevant, accurate and useful in many settings.

Hold on to two phrases from that definition, because everything else in the chapter is a consequence of them: "outside its training data" and "without retraining". The rest of the lecture shows what problems those phrases solve.

## Problem 1: the model only knows its training data

To motivate RAG he first draws the ordinary generative AI application. A user sends a **query**. Before the query reaches the **LLM** the application adds a **prompt**, which is just a set of instructions telling the model how to behave. The model then produces an **output**. In this design the LLM's only job is to generate content from whatever it absorbed during training, and that is where the first weakness comes from.

<Infographic
  src="/img/agentic-course/04-llm-limits.svg"
  alt="A plain LLM app with a query and prompt feeding an LLM that produces output, with two disadvantages: hallucination because of a training cut-off, and private startup data that fine-tuning handles badly compared with a RAG pipeline"
  caption="Redrawn from the whiteboard."
/>

His example uses dates. Imagine today is 31 August and the model in your app is GPT-5, the recent OpenAI model. Suppose its training data stopped on 1 August. The model has no idea what happened in the world between 1 and 31 August. Now a user asks about an event from that gap. A model that is only an LLM does not reply "I do not know". It **hallucinates**: it produces a fluent, confident answer that is made up. The instructor's joke is that the model does not want to look like a fool, so it invents something convincing, and the answer is written so persuasively that you may believe it. Hallucination is the first major disadvantage of using an LLM on its own.

:::note The dates are an illustration
The 1 August and 31 August dates are invented to make the idea concrete; they are not a claim about when any real model was trained. The lesson is general: every model has a **knowledge cut-off**, and anything after it is invisible to the model unless you supply it.
:::

## Problem 2: private and changing data

The second disadvantage is about data the model never saw because it was never public. Suppose you run a startup and you want a chatbot that answers questions about your company: HR policies, finance policies, and similar internal documents. These are protected, so they cannot have been part of any public training set. Yet you want your LLM to use them.

The obvious suggestion, which the instructor says many people raise, is to **fine-tune** the model on that data. He agrees it is a valid route but calls it expensive and tedious, because the model has billions of parameters and adjusting them takes a great deal of time and compute. There is a second, quieter problem: company policies keep changing as the startup grows. If every update demanded a new fine-tuning run you would be retraining constantly, which nobody can afford.

The alternative he draws next to the fine-tuning option is a **RAG pipeline** that handles the same private data without touching the model's weights and that can be refreshed whenever the data changes. In his sketch the two options sit side by side, with an arrow from fine-tuning towards RAG to show where he is going.

| Approach | What changes | Cost when data changes | What the instructor says about it |
| --- | --- | --- | --- |
| Plain LLM | Nothing | Not applicable | Hallucinates on anything after its cut-off or outside its training data |
| Fine-tuning | The model's parameters | Repeat the training run each time | Valid, but expensive and tedious |
| RAG | Nothing in the model; the knowledge base is added beside it | Re-ingest the changed documents | The route this course takes |

## Drawing the RAG pipeline

He restates the definition, "optimise the output of a language model by referencing an authoritative knowledge base outside its training data", and then draws how that works. The drawing builds up in stages across the next ten minutes, and the finished version is the board below.

<Infographic
  src="/img/agentic-course/04-rag-whiteboard.svg"
  alt="A RAG whiteboard with a data ingestion pipeline from data through parsing and embedding into a vector DB, and a retrieval pipeline where a user query is embedded, searched against the vector DB, combined with a prompt and sent to an LLM"
  caption="Redrawn from the whiteboard."
/>

### The data ingestion pipeline

On the board there is a user, an LLM, and now a new box: an **external vector database**. The LLM already carries what it learned in training. Your own data, whether it is HR policies, finance rules or anything else, is fed through a **data ingestion pipeline** whose job is to fill that vector database. The pipeline has three steps.

1. **Parsing.** The data can arrive in any format: PDF, HTML, Excel, a SQL database, or unstructured text. Parsing means reading that data, structured or not, and then dividing it into chunks. The instructor calls parsing the most important step. His reasoning is that if you get parsing right, building the rest of a RAG application becomes easy.
2. **Chunking.** The text is divided into smaller pieces. This is necessary because the pieces are what get stored in the vector database.
3. **Embedding.** Each chunk goes through an **embedding model**, which converts text into a **vector**, a numerical representation of the text. Once text is numbers you can apply similarity algorithms, such as **cosine similarity**, to find stored items that resemble a query. There are many embedding models to choose from, for example Google's Gemini embeddings, OpenAI's embeddings and Hugging Face models. They differ in cost, and open-source ones are available too.

The embedded chunks are written into the **vector store** (also called a vector DB), one record per chunk. At the end of this pipeline your company's text exists as vectors in a database. The instructor labels that database the **knowledge base**, and he points out that the LLM itself does not hold this knowledge; at most it has fragments of similar material from training.

### The retrieval pipeline

Now the user asks a question, for example "What is the leave policy of my company?". In RAG the query does **not** go straight to the LLM. It is first converted into a vector, using the same kind of embedding step, because the vector store can only compare vectors with vectors. The vector store then runs a **similarity search** and returns the stored chunks that are most similar to the question. The instructor calls that returned material the **context**.

The context is then combined with a prompt. The prompt tells the model to answer the question using the supplied context. The LLM reads the context and the prompt together and produces the output. This whole second half is the **retrieval pipeline**, and the complete arrangement, ingestion plus retrieval, is what he calls **traditional RAG**.

Compare it with the plain app from the start of the lecture. The prompt and LLM are still there. What has been added is a step that fetches relevant private text and places it in front of the model, so the model does not have to guess.

### Perplexity and a founder's aside

He is clear that RAG does not remove hallucination completely. If the answer is present in the vector database, the model gets the right context and answers well. If the data is **not** in the vector database, the model can still hallucinate. So RAG reduces the problem rather than abolishing it.

As a real-world example he points to **Perplexity**. It is a RAG application at heart: it is connected to retrievers and tools, it searches the web, and then an LLM summarises what was found. In his drawing the answer step has arrows labelled retrievers, tools and web search leading into the model. He adds that Perplexity uses several LLMs behind the scenes.

He also mentions that he is planning to start a company within a couple of weeks, and that the product is itself a RAG application aimed at a problem developers have. That is why he has not been publishing many videos lately. He also promises that later parts of the course will cover other kinds of RAG, in particular **agentic RAG**, "from basic to advanced with implementation". Agentic RAG is not taught in this section, though the repository holds a small notebook on it, which is summarised near the end of this chapter.

## The two pipelines, tidied up

On a fresh page he redraws the idea cleanly as two large boxes: a **data ingestion pipeline** on the left and a **retrieval pipeline** on the right, with a vector store joining them. Study this board, because the rest of the chapter is simply an implementation of it.

<Infographic
  src="/img/agentic-course/04-two-pipelines.svg"
  alt="Two pipelines: data ingestion with data ingest, data parsing and embedding feeding a vector store, and a retrieval pipeline where a user query is embedded, matched to context, combined with a prompt and sent to an LLM to produce output"
  caption="Redrawn from the whiteboard."
/>

The left box has three stages. **Data ingest** reads PDF, HTML, Excel or database files and, in his words, "reads the data into a document". **Data parsing** performs the chunking. **Embedding** turns the text into vectors. In the data-ingestion part he also notes that embeddings can be **open source or paid**, and he scribbles the word "optimisation" because choosing and tuning these stages is a topic of its own.

The right box begins with the **user query**. The query is embedded, the **retriever**, which sits on top of the vector store, finds the **context**, and the context is added to a **prompt** before it is sent to the **LLM**, which writes the output.

He gives names to the stages of the right-hand box because you will hear them constantly:

| Word in "retrieval augmented generation" | What it means in the pipeline |
| --- | --- |
| **Retrieval** | Embedding the query and fetching similar chunks from the vector store |
| **Augmentation** | Adding that context to the prompt, so the model has extra material along with the instruction |
| **Generation** | The LLM writing the final output from context plus prompt |

He closes this first lecture by previewing the next one. It will start from the files, PDF, HTML, Excel, SQL or anything else, and show document parsing into a **document** data structure that can be chunked and stored in a vector store. After that it will use both open-source and paid embeddings and wire up a retriever. He says he prefers making larger videos that cover many things at once, so that you do not have to follow a playlist of fifty small ones. He also lists topics he will return to when he reaches data parsing: chunking strategies, the **semantic chunker**, optimisation and **context engineering**.

## How the course approaches the code

A new lecture begins here. He recaps what is already established: what RAG is, which weaknesses it addresses, and the two pipelines. He then explains the order of the coding.

Everything uses LangChain, and this is still **traditional RAG**; agentic RAG comes in later parts. The start is basic code in a Jupyter notebook so that fundamentals are visible. Complexity then increases: code is written as reusable classes, and finally the stages are linked into a real pipeline in modular files.

His agenda page lists two items:

1. **The document structure.** Anything that goes into a vector database must first be in this structure, so it must be understood first.
2. **The complete RAG pipeline**, split into the data ingestion pipeline and the query retrieval pipeline.

## Ingestion in more detail: parsing, chunking, embedding, storing

Here he zooms into the left box of the earlier board and explains what each step is for, using a second drawing.

<Infographic
  src="/img/agentic-course/04-ingestion-detail.svg"
  alt="From files to a vector store: data ingest, data parsing into the document structure, four chunks, an embedding step, a vector DB and similarity search, with context size limits for both the embedding model and the LLM"
  caption="Redrawn from the whiteboard."
/>

**Data ingestion** can start from any kind of file. The aim of the first step is to read the file contents and convert them into a structure that supports chunking, embedding and storage. That structure is the **document structure**, and it has two parts: **content** and **metadata**. He stresses that parsing quality affects everything downstream. A cleaner parse gives a vector store that returns more accurate results during retrieval.

**Chunking** is the next step. The whole of the parsed data is divided into pieces, chunk 1, chunk 2, chunk 3 and chunk 4 in his picture. The reason is a hard limit: every embedding model, and every LLM, has a fixed **context size**, the maximum amount of text it will accept at once. If you took a 100-page PDF and passed it whole to an embedding model, the model would refuse it, because the input exceeds its limit. The same holds later for the LLM, which also has its own context size, and different models have different sizes. So you divide the text into chunks that fit within those limits.

:::tip Context size in practice
The model used later in this chapter, `all-MiniLM-L6-v2`, only reads about 256 word pieces (tokens) of each input and silently truncates the rest. A 1000-character chunk is roughly 200 to 250 tokens, so the chosen chunk size sits just inside that limit. If you raise `chunk_size` a lot, check your embedding model's maximum sequence length first, otherwise the tail of each chunk is ignored without any warning.
:::

**Embedding** comes after chunking: every chunk is converted from text to a vector. The vectors are then stored in the **vector DB**, where each chunk becomes a record (record one, record two, and so on). Once the records exist you can run a **similarity search** over them.

He then sets you an **assignment**. In the video he builds the pipeline with PDF and text files. You should repeat the same pipeline for another format, such as Excel or CSV, and he asks you to do it, because working with a second format is how you check that you understood the pattern.

## Project setup with uv

He starts from an empty folder, opens a command prompt there, and launches VS Code with `code .`. (The recording is on Windows, in a folder called `YTRAG`, so paths in his terminal look like `E:\YTRAG`.) In the VS Code terminal he does the following.

1. **Initialise the project** with `uv init`. The folder becomes a Python project with a `pyproject.toml`, a `.python-version` file and a sample `main.py`.
2. **Create the environment** with `uv venv`. It reports that it is using CPython 3.13.2 and creating a virtual environment in `.venv`.
3. **Activate it** with `.venv\Scripts\activate` on Windows. On macOS or Linux the equivalent is `source .venv/bin/activate`.
4. **Write `requirements.txt`**. At this point it lists `langchain`, `langchain-core`, `langchain-community`, `pypdf` and `pymupdf`. He explains that the last two are libraries for reading PDF documents, and he returns to why he uses two of them when he reaches the PDF loaders. Later in the chapter, when embeddings and vector stores arrive, he appends `sentence-transformers`, `faiss-cpu` and `chromadb`, and still later `langchain-groq` and `python-dotenv`.
5. **Install** them with `uv add -r requirements.txt`.
6. **Create folders**: `data` for the input files and `notebook` for the notebooks.
7. **Add the notebook kernel** with `uv add ipykernel`, so that VS Code's Jupyter notebooks can use the environment. He then creates `document.ipynb` in `notebook/` and selects the project's Python as the kernel.

```bash
uv init
uv venv
.venv\Scripts\activate          # macOS or Linux: source .venv/bin/activate
uv add -r requirements.txt
uv add ipykernel
```

His first `requirements.txt`, as shown on screen:

```text
langchain
langchain-core
langchain-community
pypdf
pymupdf
```

:::note Package names
The tutorial repository's final `requirements.txt` also lists `typesense`, `langchain_openai` and `langgraph`, which were added for the extra notebooks described at the end of this chapter. The video adds its packages step by step as each part needs them.
:::

He gives one piece of advice before coding: you must be comfortable with Python, because from here on the code is more advanced and he cannot type every line slowly. He says never skip Python, and he will move at a brisker pace and explain rather than dictate.

## The LangChain Document

He returns to the earlier board. In the data ingestion pipeline the first stage is loading data, then chunking, then embedding, then storing. Whatever the source, the thing that comes out of the loading and chunking stages is a **Document**, so the first lesson is to understand what a Document is.

A LangChain `Document` is a small data structure that holds a piece of text together with information about it. It has exactly two core parts:

- **`page_content`** is the text itself. When you read a file, the contents of that file (or that page of it) go here. It is especially helpful for research papers or product manuals where the text is long and you want to search it.
- **`metadata`** is a dictionary of extra facts: the file name, the number of pages, a timestamp, the author, or anything else you think is useful.

He opens the SVG that sits in the repository's `notebook` folder, `1-langchain-document-components.svg`, which lays this out as a picture, and scrolls through it. The picture shows the two fields, example metadata, a row of document loaders, and a row of text splitters. Below is an original redraw that follows the same content and uses the current import paths.

<Infographic
  src="/img/agentic-course/04-document-components.svg"
  alt="A LangChain Document has page_content as text and metadata as a dictionary, with typical metadata fields, a row of loaders that return Documents and a row of text splitters that turn Documents into smaller Documents"
  caption="Redrawn from the notebook picture the instructor opens, with up-to-date import paths."
/>

:::note Import paths in the picture
The repository picture uses `from langchain.schema import Document` and `from langchain.document_loaders import ...`. Those older paths have been moved. In current LangChain the `Document` class lives in `langchain_core.documents`, the loaders live in `langchain_community.document_loaders`, and the text splitters live in the separate `langchain_text_splitters` package. The redraw above and the code below use the current locations.
:::

He lists the loaders you will meet, noting that each one reads a particular kind of source and **always returns Documents**:

| Loader | Reads | Notes |
| --- | --- | --- |
| `PyPDFLoader` | PDF files | Based on the `pypdf` library |
| `PyMuPDFLoader` | PDF files | Based on `PyMuPDF`; he finds it better than the first |
| `TextLoader` | Plain text files | Metadata contains the source path |
| `CSVLoader` | CSV files | The CSV content becomes Documents |
| `WebBaseLoader` | Web pages | |
| `DirectoryLoader` | A whole folder | Wraps another loader and applies it to every matching file |

Why does this matter? Because the very next steps, chunking and embedding, and the final vector store searches, all operate on Documents. The loader can differ; the output type does not.

### Creating a Document by hand

To prove how simple it is, he creates one manually. In the first notebook cell he imports the class from `langchain_core.documents` and, hovering over it, shows the tooltip: it is a "class for storing a piece of text and associated metadata".

```python
###Document Structure

from langchain_core.documents import Document
```

He then builds a Document with the two parameters. The text is a sentence he pretends came from a text file, and the metadata holds a source, a page count, an author and a creation date.

```python
doc=Document(
    page_content="this is the main text content I am using to create RAG",
    metadata={
        "source":"exmaple.txt",
        "pages":1,
        "author":"Krish Naik",
        "date_created":"2025-01-01"
    }
)
doc
```

**What the code does.** `Document(...)` takes two keyword arguments. `page_content` is the string that will eventually be embedded. `metadata` is any dictionary; here it has four keys. Evaluating `doc` on the last line makes the notebook print the object's representation.

Output:

```text
Document(metadata={'source': 'exmaple.txt', 'pages': 1, 'author': 'Krish Naik', 'date_created': '2025-01-01'}, page_content='this is the main text content I am using to create RAG')
```

The file name is spelled `exmaple.txt` in the notebook. It is only a label, so the typo has no effect, and it is kept here so that the code matches the repository.

**Why metadata is worth the effort.** Once this Document has been chunked, embedded and stored, similarity search can be combined with **filters** on metadata. Suppose you search for "the main text content for building RAG" and add a filter that the author is Krish Naik. The vector database then restricts the search to records whose metadata matches, instead of searching everything. Metadata is therefore not decoration. It is how you narrow searches, trace a result back to its file and page, and later show citations. His point is simple: more useful metadata means better retrieval.

## Loading text files

Next he wants real files to load. He creates a folder and two small text files with Python, rather than by hand, simply to show more code. The notebook lives inside `notebook/`, so everything is addressed relative to the parent folder with `../`.

```python
## Create a simple txt file
import os
os.makedirs("../data/text_files",exist_ok=True)
```

`os.makedirs` creates the directory, and `exist_ok=True` means "do nothing if it is already there". He first typed the path as `data/text_files`, which created a stray `data` folder inside `notebook/`. He deleted that and changed the path to start with `../`, which creates `data/text_files` beside the notebook folder. The lesson: a notebook resolves relative paths from its **own** folder, not from the project root.

Then he defines the content of two files in a dictionary whose keys are paths and values are file contents: a short "Python Programming Introduction" and a short "Machine Learning Basics". A loop opens each path in write mode and writes the text.

```python
sample_texts={
    "../data/text_files/python_intro.txt":"""Python Programming Introduction

Python is a high-level, interpreted programming language known for its simplicity and readability.
Created by Guido van Rossum and first released in 1991, Python has become one of the most popular
programming languages in the world.

Key Features:
- Easy to learn and use
- Extensive standard library
- Cross-platform compatibility
- Strong community support

Python is widely used in web development, data science, artificial intelligence, and automation.""",
    
    "../data/text_files/machine_learning.txt": """Machine Learning Basics

Machine learning is a subset of artificial intelligence that enables systems to learn and improve
from experience without being explicitly programmed. It focuses on developing computer programs
that can access data and use it to learn for themselves.

Types of Machine Learning:
1. Supervised Learning: Learning with labeled data
2. Unsupervised Learning: Finding patterns in unlabeled data
3. Reinforcement Learning: Learning through rewards and penalties

Applications include image recognition, speech processing, and recommendation systems
    
    
    """

}

for filepath,content in sample_texts.items():
    with open(filepath,'w',encoding="utf-8") as f:
        f.write(content)

print("✅ Sample text files created!")
```

**What the code does.** `sample_texts` maps a file path to the text that should go in it. The `for` loop walks `sample_texts.items()`, opens each path with `open(filepath, 'w', encoding="utf-8")`, and writes the content. The `with` statement closes the file for you. The final `print` confirms it finished.

**An error on camera and its fix.** The first run raised `FileNotFoundError: No such file or directory: 'data/text_files/machine_learning.txt'`. The cause was the same relative-path problem: he had edited only one of the two dictionary keys to start with `../`, so the other key still pointed at a folder that does not exist from inside `notebook/`. Making both keys start with `../data/text_files/` fixed it. After that the cell printed its success message and the Explorer panel showed `machine_learning.txt` and `python_intro.txt`. He admits that he could simply have created the two files by hand.

### `TextLoader`

He loads the Python introduction using LangChain's `TextLoader`. Two imports appear in the notebook, and he comments on why.

LangChain keeps reorganising its packages. Older tutorials import `TextLoader` from `langchain.document_loaders`, newer ones from `langchain_community.document_loaders`. His rule of thumb is that either works until you see a deprecation warning. Today the community path is the right one, and in LangChain 1.x the old `langchain.document_loaders` module is no longer part of the main package (it survives only in the `langchain-classic` compatibility package), so the code below uses only the community import.

```python
### TextLoader

from langchain_community.document_loaders import TextLoader

loader=TextLoader("../data/text_files/python_intro.txt",encoding="utf-8")
document=loader.load()
print(document)
```

**What the code does.**

- `TextLoader(path, encoding="utf-8")` creates a loader object but reads nothing yet. Evaluating it on its own (as he did first) only shows `<langchain_community.document_loaders.text.TextLoader ...>`.
- `loader.load()` actually reads the file and returns a **list of Documents**. A text file produces one Document.
- `print(document)` shows that list.

Output (shortened):

```text
[Document(metadata={'source': '../data/text_files/python_intro.txt'}, page_content='Python Programming Introduction\n\nPython is a high-level, interpreted programming language known for its simplicity ...')]
```

He points out two things. The loader gave back the data **already in Document form**, with `page_content` and `metadata`, simply because it is a LangChain loader. And the metadata was filled in automatically with a `source` key holding the file path. You are free to add more keys later, but even the default is useful.

### `DirectoryLoader`

Loading file after file is tedious. If all the important files sit in a directory, `DirectoryLoader` reads them in one go. It needs a folder path, a **glob pattern** for the files to match, a **loader class** for how to read each file, and optional **loader keyword arguments** that are passed to that class. Because the pattern is a parameter, you could also pass a list of patterns.

```python
### Directory Loader
from langchain_community.document_loaders import DirectoryLoader

## load all the text files from the directory
dir_loader=DirectoryLoader(
    "../data/text_files",
    glob="**/*.txt", ## Pattern to match files  
    loader_cls= TextLoader, ##loader class to use
    loader_kwargs={'encoding': 'utf-8'},
    show_progress=False

)

documents=dir_loader.load()
documents
```

**What the code does.**

- `"../data/text_files"` is the folder to scan.
- `glob="**/*.txt"` means "every `.txt` file, in this folder or any sub-folder".
- `loader_cls=TextLoader` says each file should be read with the `TextLoader` you just used.
- `loader_kwargs={'encoding': 'utf-8'}` passes `encoding="utf-8"` to every `TextLoader`.
- `show_progress=False` turns off the progress bar (see the error below).
- `dir_loader.load()` returns one Document per file.

**A second error on camera.** In the first run he wrote `show_progress=True`. The loader then raised `ImportError: To log the progress of DirectoryLoader you need to install tqdm, pip install tqdm`. He chose the quicker fix: set `show_progress` to `False`. Installing `tqdm` would work equally well if you want the bar.

With the flag off, the cell returns two Documents, one for `machine_learning.txt` and one for `python_intro.txt`. Their `source` metadata shows the path (on Windows with backslashes, such as `..\data\text_files\machine_learning.txt`). You now have a list of Documents, one per file, and chunking can be applied to that list afterwards.

## Loading PDFs: `PyPDFLoader` and `PyMuPDFLoader`

He copies a few PDFs into a new `data/pdf` folder: the "Attention Is All You Need" paper (`attention.pdf`), a technical report about embedding models (`emneddings.pdf`, the spelling in the repository), a computer vision paper (`objectdetection.pdf`), and a one-page `proposal.pdf`. The aim is to read the text files **and** the PDFs.

He copies the directory-loader cell and adapts it. First he tries to import `PyPDFLoader` from `langchain_core.document_loaders`, finds nothing there, and checks the documentation. It lives in `langchain_community.document_loaders`, alongside `PyMuPDFLoader`.

Why two PDF loaders? Both read PDFs, both return Documents. He opens the documentation hover for each. `PyPDFLoader` loads and parses PDFs using the `pypdf` library. `PyMuPDFLoader` uses the `PyMuPDF` library, which offers richer extraction. His judgement is that PyMuPDF is better than PyPDF. The docs also describe their differences, and he suggests you compare them as you gain experience. In this lesson he uses `PyMuPDFLoader`.

```python
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader

## load all the text files from the directory
dir_loader=DirectoryLoader(
    "../data/pdf",
    glob="**/*.pdf", ## Pattern to match files  
    loader_cls= PyMuPDFLoader, ##loader class to use
    show_progress=False

)

pdf_documents=dir_loader.load()
pdf_documents
```

**What the code does.** Compared with the text version, three things changed: the folder is `../data/pdf`, the glob is `**/*.pdf`, and the loader class is `PyMuPDFLoader`. The `loader_kwargs` line is gone, and that is the fix for the error below.

**An error on camera and its fix.** His first version still carried `loader_kwargs={'encoding': 'utf-8'}`. Every PDF then failed with `Error loading file ..\data\pdf\attention.pdf` followed by a `TypeError` complaining that PyMuPDF's `get_text` call received an unexpected argument. A PDF is a binary format, and PyMuPDF's page text reader has no `encoding` option, although the text loader did. Removing `loader_kwargs` entirely cured it, because PDF loaders need no encoding setting.

After the fix, the cell returns one Document **per page** of every PDF. The metadata differs from the text-file case and is far richer. Among the keys visible on screen are `producer`, `creator`, `creationdate`, `source`, `file_path`, `total_pages`, `format`, `title`, `author`, `subject`, `keywords`, `moddate`, `trapped` and `page`. He notes `total_pages` values of 15 for the first PDF, then 27 and 21 for the next two, and says some of the PDFs he made himself show an author name, since the loader pulls whatever the PDF contains.

Finally he checks the type.

```python
type(pdf_documents[0])
```

Output:

```text
langchain_core.documents.base.Document
```

So the element is a `Document`, and `pdf_documents` is a list of them. That is the key idea of the section: **whatever you load, the result is a list of Documents**.

## Other file types, and the loaders catalogue

His next instruction is another assignment. Having seen text and PDF, you can work out Excel, databases and other formats. The way to do it is to search for "LangChain document loaders" and open the integrations page. It lists loaders for an enormous range of sources, grouped by provider and type. He opens the entry for **AWS S3 Directory** as an example: you install the extra library, supply the bucket details after authenticating, and the loader then reads the files in that bucket. The method is always the same: pick the loader, load, inspect the Document structure that comes back, and judge whether it is good for your use.

With that, he declares data ingestion complete: any source can be turned into the Document data structure. Next come chunking, embedding and the vector store. He has also shown how to read text and PDF files, and he points you to the documentation for the rest.

## Notebook 2: from ingestion to a vector database

He creates a second notebook, `pdf_loader.ipynb`, whose first markdown cell is "RAG Pipelines: Data Ingestion to Vector DB Pipeline". It builds the whole left-hand pipeline: loading, chunking, embeddings and storage. He starts from the PDF folder he already has.

The first code cell holds the imports.

```python
import os
from langchain_community.document_loaders import PyPDFLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
```

**What the imports are for.** `os` handles file paths, `PyPDFLoader` and `PyMuPDFLoader` read PDFs, `RecursiveCharacterTextSplitter` will do the chunking, and `Path` from `pathlib` gives clean, cross-platform folder handling.

:::note Where the splitter lives now
The video imports the splitter from `langchain.text_splitter`. That module has been moved to its own package. In a new project install `langchain-text-splitters` and write `from langchain_text_splitters import RecursiveCharacterTextSplitter`. The class and its arguments are identical.
:::

### Reading every PDF in a folder

He writes a function that reads a whole directory of PDFs and adds extra metadata of his own.

```python
### Read all the pdf's inside the directory
def process_all_pdfs(pdf_directory):
    """Process all PDF files in a directory"""
    all_documents = []
    pdf_dir = Path(pdf_directory)
    
    # Find all PDF files recursively
    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_file in pdf_files:
        print(f"\nProcessing: {pdf_file.name}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            documents = loader.load()
            
            # Add source information to metadata
            for doc in documents:
                doc.metadata['source_file'] = pdf_file.name
                doc.metadata['file_type'] = 'pdf'
            
            all_documents.extend(documents)
            print(f"  ✓ Loaded {len(documents)} pages")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents

# Process all PDFs in the data directory
all_pdf_documents = process_all_pdfs("../data")
```

**What the function does, line by line.**

- `all_documents = []` is an empty list that will collect every page Document from every PDF.
- `Path(pdf_directory)` turns the folder string into a `Path`, and `pdf_dir.glob("**/*.pdf")` finds every PDF in the folder tree. Wrapping it in `list(...)` turns the result into a list, so `len(pdf_files)` works for the printed count.
- The `for` loop handles one PDF at a time. `PyPDFLoader(str(pdf_file))` creates a loader for that file (the loader wants a string path), and `loader.load()` returns one Document for each page.
- The inner loop **enriches the metadata**. It adds `source_file`, the bare file name such as `attention.pdf`, and `file_type`, set to `'pdf'`. He points out that you can invent any number of extra metadata keys this way.
- `all_documents.extend(documents)` appends this PDF's pages to the master list. `extend` adds the elements, whereas `append` would nest a list inside the list.
- The `try`/`except` makes one corrupt PDF print an error instead of killing the whole run.
- The function returns the list, and the last line calls it on `"../data"`.

Output:

```text
Found 4 PDF files to process

Processing: attention.pdf
  ✓ Loaded 15 pages

Processing: emneddings.pdf
  ✓ Loaded 27 pages

Processing: objectdetection.pdf
  ✓ Loaded 21 pages

Processing: proposal.pdf
  ✓ Loaded 1 pages

Total documents loaded: 64
```

So there are 64 page-sized Documents. Inspecting `all_pdf_documents` shows a list of Documents. For each one you see PDF's built-in metadata (author, keywords, modification date and so on), plus the keys he added (`source`, `source_file`, `file_type`, `total_pages`), and the page text in `page_content`.

## Chunking with `RecursiveCharacterTextSplitter`

A page can still be long, and it is the wrong unit for search. He writes a function that takes the list of Documents and returns smaller ones.

```python
### Text splitting get into chunks

def split_documents(documents,chunk_size=1000,chunk_overlap=200):
    """Split documents into smaller chunks for better RAG performance"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")
    
    # Show example of a chunk
    if split_docs:
        print(f"\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")
    
    return split_docs
```

**What the function does.**

- `chunk_size=1000` is the maximum length of a chunk, measured in characters because `length_function=len`.
- `chunk_overlap=200` means that neighbouring chunks share up to 200 characters. He explains it as some text being repeated across two consecutive chunks when the splitting happens. The overlap keeps a sentence cut at a boundary intact in at least one chunk.
- `separators=["\n\n", "\n", " ", ""]` is the list of places the splitter is allowed to cut, tried in order. The splitter is **recursive**: it first tries to break at a paragraph break (`"\n\n"`), then a line break (`"\n"`), then a space between words, and only if all else fails between any two characters (`""`). He asks viewers to say in the comments what each separator is, and he promises later parts will compare other chunking strategies.
- `text_splitter.split_documents(documents)` does the work. It splits each Document's `page_content` and **copies the original metadata onto every chunk**, so a chunk still knows its file and page.
- The block that follows prints the first 200 characters of the first chunk and its metadata, so you can eyeball the result. Then the function returns the chunks.

<Infographic
  src="/img/agentic-course/04-chunk-overlap.svg"
  alt="A long text split into three chunks of up to 1000 characters that overlap by 200 characters, with the order of separators the recursive splitter tries"
  caption="Explanatory board (not shown in the video): chunk size, overlap and the separator order."
/>

He calls it on the list from before.

```python
chunks=split_documents(all_pdf_documents)
chunks
```

Output (shortened):

```text
Split 64 documents into 359 chunks

Example chunk:
Content: Provided proper attribution is provided, Google hereby grants permission to
reproduce the tables and figures in this paper solely for use in journalistic or
scholarly works.
Attention Is All You Need
...
Metadata: {'producer': 'pdfTeX-1.40.25', 'creator': 'LaTeX with hyperref', 'creationdate': '2024-04-10T21:11:43+00:00', 'author': '', 'k...
```

64 page Documents became **359 chunks**. He reminds you that before chunking there was one Document per page, so 64 pages, and afterwards there is one Document per chunk, with the metadata carried along. The embedding stage that follows now works on these 359 pieces.

## Embeddings with `EmbeddingManager`

Two stages remain on his board: **embedding generation** and the **vector store DB**. For these he deliberately writes classes, one per job, with a few methods each, because he wants to demonstrate modular code and later link the pieces. He chooses open-source models so that everyone can follow without paying.

He first adds the libraries he needs for this stage to `requirements.txt` and installs them (`uv add -r requirements.txt`): `sentence-transformers`, which loads Hugging Face embedding models, `faiss-cpu`, and `chromadb`. He mentions that FAISS and Chroma are both good open-source vector stores, and that you may use either. The notebook uses Chroma. The modular package later uses FAISS.

```python
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import uuid
from typing import List, Dict, Any, Tuple
from sklearn.metrics.pairwise import cosine_similarity
```

**What the imports are for.** `numpy` holds the embedding arrays. `SentenceTransformer` loads the embedding model. `chromadb` is the vector database. `uuid` generates a unique id for every stored record, because a vector database needs each record to have one. The `typing` names annotate the methods. `cosine_similarity` from scikit-learn is imported because he plans to use cosine similarity when retrieving. (The class he writes does not call it directly; the similarity is computed from Chroma's distance, as you will see.) Running the cell takes a few seconds and prints a harmless `tqdm` warning about `IProgress`, which only means Jupyter's widget extras are not installed.

```python
class EmbeddingManager:
    """Handles document embedding generation using SentenceTransformer"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager
        
        Args:
            model_name: HuggingFace model name for sentence embeddings
        """
        self.model_name = model_name
        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the SentenceTransformer model"""
        try:
            print(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            print(f"Model loaded successfully. Embedding dimension: {self.model.get_sentence_embedding_dimension()}")
        except Exception as e:
            print(f"Error loading model {self.model_name}: {e}")
            raise

    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        if not self.model:
            raise ValueError("Model not loaded")
        
        print(f"Generating embeddings for {len(texts)} texts...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        return embeddings


## initialize the embedding manager

embedding_manager=EmbeddingManager()
embedding_manager
```

**What the class does.**

- `__init__` is the constructor. He reminds you that every class needs one. It stores the model name, sets `self.model = None` for now, and immediately calls `self._load_model()`. The default model is **`all-MiniLM-L6-v2`**, a small Hugging Face model that converts a piece of text into a vector of **384 numbers**.
- `_load_model` has a leading underscore. He explains that this marks a "protected" helper, meant to be used inside the class only. It loads the model with `SentenceTransformer(self.model_name)` and prints the embedding dimension using `get_sentence_embedding_dimension()`. The `try`/`except` prints a clear message and re-raises if loading fails.
- `generate_embeddings(texts)` takes a **list of strings** and returns a **NumPy array** with one row per text and one column per dimension. It guards against an unloaded model, prints how many texts it is embedding, and calls `self.model.encode(texts, show_progress_bar=True)`, so you see a progress bar for large lists.

In the video he first also wrote a small `get_embedding_dimension()` method, then decided it was unnecessary because the dimension is already printed in `_load_model`, and deleted it. That is why the notebook version has just the two methods.

He creates the object.

```python
## initialize the embedding manager

embedding_manager=EmbeddingManager()
embedding_manager
```

Output:

```text
Loading embedding model: all-MiniLM-L6-v2
Model loaded successfully. Embedding dimension: 384
<__main__.EmbeddingManager at 0x...>
```

The constructor ran, the model downloaded (the first time) and loaded, and the manager is ready to turn text into 384-number vectors.

## The vector store: a ChromaDB class

The second class wraps the database. A vector store is where the vectors from the embedding layer are saved, so that similarity search can be run on them later.

```python
class VectorStore:
    """Manages document embeddings in a ChromaDB vector store"""
    
    def __init__(self, collection_name: str = "pdf_documents", persist_directory: str = "../data/vector_store"):
        """
        Initialize the vector store
        
        Args:
            collection_name: Name of the ChromaDB collection
            persist_directory: Directory to persist the vector store
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_store()

    def _initialize_store(self):
        """Initialize ChromaDB client and collection"""
        try:
            # Create persistent ChromaDB client
            os.makedirs(self.persist_directory, exist_ok=True)
            self.client = chromadb.PersistentClient(path=self.persist_directory)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"}
            )
            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Any], embeddings: np.ndarray):
        """
        Add documents and their embeddings to the vector store
        
        Args:
            documents: List of LangChain documents
            embeddings: Corresponding embeddings for the documents
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents must match number of embeddings")
        
        print(f"Adding {len(documents)} documents to vector store...")
        
        # Prepare data for ChromaDB
        ids = []
        metadatas = []
        documents_text = []
        embeddings_list = []
        
        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # Generate unique ID
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            ids.append(doc_id)
            
            # Prepare metadata
            metadata = dict(doc.metadata)
            metadata['doc_index'] = i
            metadata['content_length'] = len(doc.page_content)
            metadatas.append(metadata)
            
            # Document content
            documents_text.append(doc.page_content)
            
            # Embedding
            embeddings_list.append(embedding.tolist())
        
        # Add to collection
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings_list,
                metadatas=metadatas,
                documents=documents_text
            )
            print(f"Successfully added {len(documents)} documents to vector store")
            print(f"Total documents in collection: {self.collection.count()}")
            
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise

vectorstore=VectorStore()
vectorstore
    
```

**What the class does.**

- `__init__` takes a **collection name** (`"pdf_documents"`) and a **persist directory** (`"../data/vector_store"`). A collection is a named group of records inside the database. Persistence means the data is written to disk, so it survives a restart and can be loaded again later.
- `_initialize_store` is the protected setup. `os.makedirs(self.persist_directory, exist_ok=True)` makes sure the folder exists. `chromadb.PersistentClient(path=...)` creates a client that stores everything under that folder. `get_or_create_collection(name=..., metadata=...)` fetches the collection if it exists and creates it otherwise, with a short description. It then prints the collection name and how many records it already holds.
- `add_documents(documents, embeddings)` takes the chunks and their vectors. It first checks that the two lists are the same length. For each pair it prepares what Chroma needs: an **id** (`doc_<8 random hex characters>_<index>`, built from a UUID so it is unique), a **metadata** dictionary (a copy of the chunk's metadata plus `doc_index` and `content_length`), the **document text**, and the **embedding** converted to a plain list with `.tolist()`. A single `self.collection.add(ids=..., embeddings=..., metadatas=..., documents=...)` call stores all of them, and the method prints how many records were added and the new total.

He explains why these separate classes exist: each does one job, and later code can link them. He reminds you again that the code is simple, but that you do need some coding knowledge if you want to get better at RAG. Running the cell creates the vector store.

```python
vectorstore=VectorStore()
vectorstore
```

Output in the video:

```text
Vector store initialized. Collection: pdf_documents
Existing documents in collection: 0
<__main__.VectorStore at 0x...>
```

The collection is empty because nothing has been added yet. (The repository's saved copy of the notebook shows `718` because the author re-ran the notebook, and that is the reason for the next warning.)

## Embedding the chunks and storing them

Now he joins the pieces. The chunks are already in the variable `chunks`. He extracts the text of every chunk, embeds all the texts, and stores both.

```python
### Convert the text to embeddings
texts=[doc.page_content for doc in chunks]

## Generate the Embeddings

embeddings=embedding_manager.generate_embeddings(texts)

##store int he vector dtaabase
vectorstore.add_documents(chunks,embeddings)
```

**What the code does.** A list comprehension collects `doc.page_content` for each chunk into `texts`. `embedding_manager.generate_embeddings(texts)` returns the matrix of vectors. `vectorstore.add_documents(chunks, embeddings)` writes chunks and vectors together, in the same order, so row `i` of the matrix belongs to chunk `i`.

**A typo on camera.** The first run failed with a `NameError` saying the name `vector_store` is not defined. He had typed the object name with an underscore, but the object created earlier is called `vectorstore`. Fixing the name made the cell run.

Output:

```text
Generating embeddings for 359 texts...
Batches: 100%|██████████| 12/12 [00:06<00:00,  1.78it/s]
Generated embeddings with shape: (359, 384)
Adding 359 documents to vector store...
Successfully added 359 documents to vector store
Total documents in collection: 359
```

Read it as follows. 359 texts were encoded in 12 batches (the model processes them in groups, 32 texts per batch by default), the result is a 359 by 384 array (359 chunks, 384 numbers each), and the collection now holds 359 records. In the file explorer a `vector_store` folder has appeared under `data`: that is the **persistence**. The vectors are saved on disk, so another session can open the folder and query it without redoing the embedding.

:::warning Running the add cell twice duplicates your data
Each record gets a **random** id. If you run the cell again, Chroma happily stores the same 359 chunks a second time. The repository's saved notebook shows exactly this, with counts of 718 and then 1077, and later queries return the same chunk several times. Delete the `data/vector_store` folder before a re-run, or use a stable id such as `file name + page + chunk number` together with `collection.upsert(...)`.
:::

With that, the whole left-hand pipeline from the board exists: documents, chunks, embeddings and a persistent vector store.

<Infographic
  src="/img/agentic-course/04-notebook-classes.svg"
  alt="The notebook functions and three classes, EmbeddingManager, VectorStore and RAGRetriever, with the data that flows between them and the fields of every retrieved hit"
  caption="Explanatory board (not shown in the video): how the notebook's pieces link."
/>

## Retrieval with `RAGRetriever`

He now builds the rest of the first board: when a user asks a question, convert the question to an embedding, hit the vector store, and get the context. The tool for this is a **retriever**. He describes a retriever as a simple **interface** built on top of a vector store: you give it a query, and it gives you back the matching content.

```python
class RAGRetriever:
    """Handles query-based retrieval from the vector store"""
    
    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initialize the retriever
        
        Args:
            vector_store: Vector store containing document embeddings
            embedding_manager: Manager for generating query embeddings
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query
        
        Args:
            query: The search query
            top_k: Number of top results to return
            score_threshold: Minimum similarity score threshold
            
        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        print(f"Retrieving documents for query: '{query}'")
        print(f"Top K: {top_k}, Score threshold: {score_threshold}")
        
        # Generate query embedding
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]
        
        # Search in vector store
        try:
            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=top_k
            )
            
            # Process results
            retrieved_docs = []
            
            if results['documents'] and results['documents'][0]:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                
                for i, (doc_id, document, metadata, distance) in enumerate(zip(ids, documents, metadatas, distances)):
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    similarity_score = 1 - distance
                    
                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            'id': doc_id,
                            'content': document,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance,
                            'rank': i + 1
                        })
                
                print(f"Retrieved {len(retrieved_docs)} documents (after filtering)")
            else:
                print("No documents found")
            
            return retrieved_docs
            
        except Exception as e:
            print(f"Error during retrieval: {e}")
            return []

rag_retriever=RAGRetriever(vectorstore,embedding_manager)
```

**What the class does.**

- The constructor receives the two objects you already built, the `VectorStore` and the `EmbeddingManager`, and keeps references to them. Notice that the **same** embedding manager is used for both documents and queries. A query must be embedded with the same model as the stored chunks, otherwise the two sets of vectors are not comparable.
- `retrieve(query, top_k=5, score_threshold=0.0)` is the important method. `top_k` is how many results you want and `score_threshold` is the minimum similarity score a result needs to be kept.
- Inside, `generate_embeddings([query])[0]` embeds the question (a list of one string, then take the first row). `self.vector_store.collection.query(query_embeddings=[...], n_results=top_k)` asks Chroma for the nearest records. Chroma returns parallel lists under the keys `ids`, `documents`, `metadatas` and `distances`, each wrapped in an outer list because you can ask several queries at once, hence the `[0]`.
- `zip(ids, documents, metadatas, distances)` walks the four lists together so each result is a tuple. For each, the code computes `similarity_score = 1 - distance`. If the score is at least `score_threshold`, it is added to `retrieved_docs` as a dictionary with `id`, `content`, `metadata`, `similarity_score`, `distance` and `rank`. The method returns the list of dictionaries, or an empty list if there are no results or something goes wrong.

Creating the retriever takes one line, passing in the vector store and the embedding manager:

```python
rag_retriever=RAGRetriever(vectorstore,embedding_manager)
```

He retrieves with a question taken from his own data, since he knows the PDFs contain the "Attention Is All You Need" paper.

```python
rag_retriever.retrieve("What is attention is all you need")
```

In the video the printed lines were `Retrieving documents for query: 'What is attention is all you need'`, `Top K: 5, Score threshold: 0.0`, `Generated embeddings with shape: (1, 384)` (one query, 384 numbers) and `Retrieved 1 documents (after filtering)`. The result had `content` beginning `3.2 Attention\nAn attention function can be described as mapping a query and a set of key-value pairs to an output ...`, a metadata dictionary naming `attention.pdf`, and a `similarity_score` of about 0.14 with a `distance` of about 0.86. He reads this as success: the context for the question has been found, quickly, in the vector store.

For a second test he opens the embeddings PDF and searches a phrase from its table of contents.

```python
rag_retriever.retrieve("Unified Multi-task Learning Framework")
```

This returned three results in the video, led by a chunk beginning `erage scores on CMTEB[22] and MTEB[23] benchmarks, ranking first overall ...`, with metadata naming the report as the "QZhou-Embedding Technical Report" and `emneddings.pdf`. He is pleased by how fast and easy it is, and says that, having reached this point, the only step left is to connect an LLM to the retrieved context. That is the next lecture, after which he will take this same code and restructure it as modular files in a `src` folder, as a pipeline you can call in sequence from data loading to vector embedding.

:::warning The similarity score here is wrong by default
This is a flaw in the source, and the video's own output shows it. The code treats `1 - distance` as a **cosine similarity**, and its comment says "ChromaDB uses cosine distance". It does not by default. A new Chroma collection measures **squared L2 distance**, which ranges well above 1 for dissimilar text. So `1 - distance` can be negative, and the `score_threshold` then throws results away for the wrong reason.

You can see it in the first query above: 5 results were requested with a threshold of 0.0, yet only **1** survived the filter, and its "similarity" was just 0.14, even though the chunk is an excellent answer. The other four had distances above 1, so their scores were negative and were silently dropped. Later, `rag_advanced` with `min_score=0.1` returns **zero** documents for "hard negative mining techniques", and the instructor blames a context-size problem. The real cause is this score formula.

The fix is to ask Chroma for cosine distance when you create the collection:

```python
self.collection = self.client.get_or_create_collection(
    name=self.collection_name,
    metadata={
        "description": "PDF document embeddings for RAG",
        "hnsw:space": "cosine",
    },
)
```

Recent Chroma releases prefer the `configuration={"hnsw": {"space": "cosine"}}` spelling for the same setting, so check the version you installed. With cosine distance, `1 - distance` is a true cosine similarity and thresholds such as 0.2 behave as intended. You must delete and rebuild the collection after changing the space.
:::

## From retrieval to an answer: augmented generation

A new lecture starts here, and he recaps. The whole data ingestion pipeline is finished: loading, chunking, converting text to vectors, storing them in a vector DB and persisting that on disk. A retrieval from the user's query also works. What remains is the **query retrieval pipeline with an LLM**, and that is where the "augmented generation" in the name of RAG happens.

He draws the board again, this time keeping only the retrieval half.

<Infographic
  src="/img/agentic-course/04-augmented-generation.svg"
  alt="Retrieval, augmentation and generation: a query is turned into a vector and sent to the vector DB, the returned context is combined with a prompt, and the LLM generates the output"
  caption="Redrawn from the whiteboard."
/>

Read it from left to right.

1. The vector DB is already ready, because the earlier pipeline filled it.
2. A new **query** arrives. It is converted to vectors with the **same embedding** that was used on the documents. The instructor underlines "Query to vectors" and labels that whole stretch **retrieval**.
3. The query vector hits the vector DB, and **context** comes back.
4. The context is combined with a **prompt**, which is the instruction on how the LLM should behave. Joining context and prompt is the step he calls **augmentation**.
5. The LLM reads that augmented prompt and writes the **output**. This last step is **generation**.

He asks you to be sure you understand these three words and the order they happen in before moving on, because every RAG variant you meet later is a modification of this sequence.

## Setting up the Groq model

For the LLM he uses **Groq**, a hosted inference service that serves open models quickly. He has already stored a Groq API key in the `.env` file at the project root. He appends two packages to `requirements.txt`: `langchain-groq`, which gives LangChain's `ChatGroq` class, and `python-dotenv`, which loads `.env` files, and installs them.

He starts a new markdown cell titled "Integration Vectordb Context pipeline With LLM output" and then writes the setup. The model he picks is `gemma2-9b-it` (he pronounces it "gamma 2"), with a low temperature of 0.1 so answers stay close to the retrieved text, and a limit of 1024 generated tokens.

```text
GROQ_API_KEY=...your key here...
```

(That is the entire `.env` file; use your own key and never commit the file.)

```python
### Simple RAG pipeline with Groq LLM
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

### Initialize the Groq LLM (set your GROQ_API_KEY in environment)
groq_api_key = os.getenv("GROQ_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="gemma2-9b-it",temperature=0.1,max_tokens=1024)

## 2. Simple RAG function: retrieve context + generate response
def rag_simple(query,retriever,llm,top_k=3):
    ## retriever the context
    results=retriever.retrieve(query,top_k=top_k)
    context="\n\n".join([doc['content'] for doc in results]) if results else ""
    if not context:
        return "No relevant context found to answer the question."
    
    ## generate the answwer using GROQ LLM
    prompt=f"""Use the following context to answer the question concisely.
        Context:
        {context}

        Question: {query}

        Answer:"""
    
    response=llm.invoke([prompt.format(context=context,query=query)])
    return response.content
```

**What the code does.**

- `load_dotenv()` reads `.env` and places `GROQ_API_KEY` into the process environment. `os.getenv("GROQ_API_KEY")` then reads it back.
- `ChatGroq(groq_api_key=..., model_name="gemma2-9b-it", temperature=0.1, max_tokens=1024)` builds the chat model. Low temperature means less randomness; `max_tokens` caps the length of the reply.
- `rag_simple(query, retriever, llm, top_k=3)` is the whole RAG loop in one function.
  1. `retriever.retrieve(query, top_k=top_k)` fetches the best chunks.
  2. `"\n\n".join([doc['content'] for doc in results])` glues their texts together with blank lines between them. That string is the **context**. If nothing came back the context is an empty string.
  3. `if not context:` returns an honest fallback message ("No relevant context found...") instead of asking the model to make something up.
  4. `prompt = f"""Use the following context to answer the question concisely. Context: ... Question: ... Answer:"""` builds the augmented prompt. This is the augmentation step: the instruction, the context and the question in one string.
  5. `llm.invoke([...])` sends it to Groq and `response.content` returns the text of the reply.

He calls it with the question that his PDFs can answer.

```python
answer=rag_simple("What is attention mechanism?",rag_retriever,llm)
print(answer)
```

Output (the retrieval log lines first, then the answer):

```text
Retrieving documents for query: 'What is attention mechanism?'
Top K: 3, Score threshold: 0.0
Generating embeddings for 1 texts...
Generated embeddings with shape: (1, 384)
Retrieved 3 documents (after filtering)

An attention mechanism is a function that maps a query and a set of key-value pairs to an output vector, using a weighted sum of the values.
```

He summarises what happened in order: the function called `retrieve`, got context, merged it with the prompt, called the LLM, and the answer is grounded in what the vector DB held. This is the pipeline he drew, working end to end.

:::danger Never paste an API key into code, and revoke any key that was on screen
While setting this up, the instructor shows the `.env` file and, as a quick test, pastes the key straight into the notebook cell instead of calling `os.getenv`. He suggests it "just for testing". Do not copy that habit: a key in a notebook ends up in saved outputs, screenshots and version control. Any key that appears in a video or a repository should be treated as leaked and revoked in the provider's console. The code above, and the repository copy, read the key from the environment, which is the right pattern. (The repository's saved notebook also contains a cell, `print(os.getenv("GROQ_API_KEY"))`, that prints the key; it is left out of this chapter for the same reason.)
:::

:::note Two points on this cell
First, `gemma2-9b-it` was a Groq-hosted model when the video was recorded, but Groq has since retired it. Pick a model from Groq's current list, for example `llama-3.1-8b-instant`, and change only the `model_name` string. Second, the repository notebook also contains a `GroqLLM` wrapper class with `generate_response` methods. The video does not use it, and the pipeline that follows calls `llm.invoke` directly.
:::

:::warning A latent bug in the prompt line
`llm.invoke([prompt.format(context=context, query=query)])` calls `.format` on a string that already has the context and query substituted by the f-string. If a PDF page contains a curly brace, which maths-heavy papers often do, `.format` treats it as a placeholder and raises a `KeyError` or `ValueError`. The prompt is already complete, so pass it directly: `llm.invoke(prompt)`. A string is accepted as a human message.
:::

## The enhanced pipeline: sources and confidence

The simple function returns only text. A real application also needs to show **where** the answer came from and **how sure** the retriever was. He pastes in a richer function, `rag_advanced`, and walks through it.

```python
# --- Enhanced RAG Pipeline Features ---
def rag_advanced(query, retriever, llm, top_k=5, min_score=0.2, return_context=False):
    """
    RAG pipeline with extra features:
    - Returns answer, sources, confidence score, and optionally full context.
    """
    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)
    if not results:
        return {'answer': 'No relevant context found.', 'sources': [], 'confidence': 0.0, 'context': ''}
    
    # Prepare context and sources
    context = "\n\n".join([doc['content'] for doc in results])
    sources = [{
        'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
        'page': doc['metadata'].get('page', 'unknown'),
        'score': doc['similarity_score'],
        'preview': doc['content'][:300] + '...'
    } for doc in results]
    confidence = max([doc['similarity_score'] for doc in results])
    
    # Generate answer
    prompt = f"""Use the following context to answer the question concisely.\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"""
    response = llm.invoke([prompt.format(context=context, query=query)])
    
    output = {
        'answer': response.content,
        'sources': sources,
        'confidence': confidence
    }
    if return_context:
        output['context'] = context
    return output

# Example usage:
result = rag_advanced("Hard Negative Mining Technqiues", rag_retriever, llm, top_k=3, min_score=0.1, return_context=True)
print("Answer:", result['answer'])
print("Sources:", result['sources'])
print("Confidence:", result['confidence'])
print("Context Preview:", result['context'][:300])
```

**What the function does.**

- It retrieves with both `top_k` and a `min_score` (passed as `score_threshold`). If nothing passes the filter it returns a dictionary with a "No relevant context found." answer, an empty source list and confidence 0.0, rather than calling the LLM.
- It builds the context exactly as before, then builds a list of **sources**. Each source records the file (`source_file` if present, else `source`, else `'unknown'`), the page number from the metadata, the similarity score, and a 300-character preview of the chunk.
- `confidence = max(...)` takes the best similarity score among the results and presents it as one number. It is a rough guide, not a calibrated probability.
- It invokes the model, then returns a dictionary with `answer`, `sources` and `confidence`. If you pass `return_context=True`, the full context string is included too.

The repository's final line asks about hard negative mining. In the video he first asked "what is attention mechanism" and then changed the question to "hard negative mining techniques", to pull from the embeddings PDF instead of the attention paper. The first call printed an answer starting "An attention mechanism is a function that maps a query and a set of key-value pairs to an output vector, where the output is a weighted sum of the values", with a source list naming `attention.pdf`, page 2, a score of about 0.27 and a 300-character preview. The second call produced this (shortened):

```text
Answer: The text describes several hard negative mining techniques used in contrastive learning for retrieval models:

* **ANCE:** Uses asynchronous ANN indexing and checkpoint states to periodically update hard negatives.
* **Conan-Embedding:** Employs a dynamic strategy, excluding and refreshing samples based on score thresholds.
* **NV-Retriever:** Proposes positive-aware mining with TopK-MarginPos and TopKPercPos filtering to reduce false negatives.
* **LGAI-Embedding:** Builds on NV-Retriever, using ANNA IR as a teacher retriever to identify hard negatives and TopKPercPos ...

Sources: [{'source': 'emneddings.pdf', 'page': 4, 'score': 0.18709993362426758, 'preview': 'QZhou-Embedding Technical Report\n ...'}]
Confidence: 0.18709993362426758
Context Preview: QZhou-Embedding Technical Report
 Kingsoft AI
2.4 Hard Negative Mining Techniques
Hard negatives serve as essential components in contrastive learning for retrieval model
training. ...
```

He calls this an "enhanced RAG pipeline" because the caller now receives the answer plus its sources, a page number and a confidence figure.

## The advanced pipeline: streaming, citations, history, summarising

He pastes a third version and asks you to read it yourself. It is a class, `AdvancedRAGPipeline`, that adds four features: a streaming display of the answer, citations appended to the answer, a history of past queries, and an optional short summary.

```python
# --- Advanced RAG Pipeline: Streaming, Citations, History, Summarization ---
from typing import List, Dict, Any
import time

class AdvancedRAGPipeline:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history = []  # Store query history

    def query(self, question: str, top_k: int = 5, min_score: float = 0.2, stream: bool = False, summarize: bool = False) -> Dict[str, Any]:
        # Retrieve relevant documents
        results = self.retriever.retrieve(question, top_k=top_k, score_threshold=min_score)
        if not results:
            answer = "No relevant context found."
            sources = []
            context = ""
        else:
            context = "\n\n".join([doc['content'] for doc in results])
            sources = [{
                'source': doc['metadata'].get('source_file', doc['metadata'].get('source', 'unknown')),
                'page': doc['metadata'].get('page', 'unknown'),
                'score': doc['similarity_score'],
                'preview': doc['content'][:120] + '...'
            } for doc in results]
            # Streaming answer simulation
            prompt = f"""Use the following context to answer the question concisely.\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"""
            if stream:
                print("Streaming answer:")
                for i in range(0, len(prompt), 80):
                    print(prompt[i:i+80], end='', flush=True)
                    time.sleep(0.05)
                print()
            response = self.llm.invoke([prompt.format(context=context, question=question)])
            answer = response.content

        # Add citations to answer
        citations = [f"[{i+1}] {src['source']} (page {src['page']})" for i, src in enumerate(sources)]
        answer_with_citations = answer + "\n\nCitations:\n" + "\n".join(citations) if citations else answer

        # Optionally summarize answer
        summary = None
        if summarize and answer:
            summary_prompt = f"Summarize the following answer in 2 sentences:\n{answer}"
            summary_resp = self.llm.invoke([summary_prompt])
            summary = summary_resp.content

        # Store query history
        self.history.append({
            'question': question,
            'answer': answer,
            'sources': sources,
            'summary': summary
        })

        return {
            'question': question,
            'answer': answer_with_citations,
            'sources': sources,
            'summary': summary,
            'history': self.history
        }

# Example usage:
adv_rag = AdvancedRAGPipeline(rag_retriever, llm)
result = adv_rag.query("what is attention is all you need", top_k=3, min_score=0.1, stream=True, summarize=True)
print("\nFinal Answer:", result['answer'])
print("Summary:", result['summary'])
print("History:", result['history'][-1])
```

**What the class does.**

- The constructor keeps the retriever and the LLM, and starts an empty `history` list.
- `query(...)` retrieves, builds context and a source list (this time with a 120-character preview), and builds the prompt.
- If `stream=True` it prints the **prompt** in 80-character slices with a short pause between each, to imitate a stream.
- It calls the LLM, then builds numbered **citations** such as `[1] attention.pdf (page 2)` and appends them to the answer.
- If `summarize=True` it makes a second LLM call asking for a two-sentence summary of the answer.
- It saves question, answer, sources and summary into `self.history` and returns a dictionary that includes the whole history.

Output when he asked "what is attention is all you need" with `top_k=3`, `min_score=0.1`, streaming on and summarising on: the "Streaming answer:" line, then the prompt text scrolling past (context chunks about "3.2 Attention", then `Question: what is attention is all you need` and `Answer:`), then a final answer with citations and a summary. His earlier attempts in the same cell show the other side of the story. Asking "Hard Negative Mining Techniques" returned `Retrieved 0 documents (after filtering)` and the answer "No relevant context found.", and so did "what is positional encoding?". He changed `min_score` to 0.1 hoping for something, tried several questions, and then settled on the attention question that worked.

He sums up the three pipelines: a **simple** one, an **enhanced** one that returns sources and confidence, and an **advanced** one with streaming, citations, history and summarisation. He tells you to read the code carefully, and to expect that for some questions nothing is returned. He puts that down to the context-size problem and says optimisations will follow.

:::note What actually caused the empty answers, and what "streaming" is
The "no relevant context" results are almost certainly the **score formula** problem described earlier: with Chroma's default L2 distance, `1 - distance` is negative or tiny for many perfectly good chunks, so the threshold removes them. It is not a context-size issue. Fix the collection to use cosine distance and re-run.

Also, the "streaming" in `AdvancedRAGPipeline` is a simulation. It prints the **prompt** slowly after retrieval; it does not stream the model's reply. Real streaming means iterating `for chunk in llm.stream(prompt): print(chunk.content, end="")`, which prints tokens as the model produces them.
:::

<Infographic
  src="/img/agentic-course/04-src-modules.svg"
  alt="The src package with data_loader.py, embedding.py, vectorstore.py and search.py linked in a chain, with a faiss_store folder on disk, a Groq model and the five steps in which app.py grew"
  caption="Explanatory board (not shown in the video): how the modular files link up. The instructor only names the files and wires them in code."
/>

## The modular pipeline in `src/`

Up to now everything lives in one notebook. He now rebuilds the same idea as a package, which is the shape code takes in a real project. He reminds you that the notebook already covered ingestion, storage and querying, and mentions that he has also shown Typesense, an open-source search engine that can act as a vector store, in the same project. Here, though, the goal is to integrate the stages **as a pipeline**.

Inside the `src/` folder he creates an empty `__init__.py`, which makes `src` a package, and then four files, one per stage:

| File | Job |
| --- | --- |
| `data_loader.py` | Read every supported file in a folder and return LangChain Documents |
| `vectorstore.py` | Hold the vectors on disk and search them (FAISS in this version) |
| `embedding.py` | Chunk the Documents and turn the chunks into vectors |
| `search.py` | Retrieve context from the store and call the LLM to produce the answer |

He then writes them in order of the pipeline: `data_loader.py`, `embedding.py`, `vectorstore.py`, `search.py`, testing each from a small `app.py`.

### `data_loader.py`

He begins with imports, one loader for each file type, and a function that gathers all supported files from a data directory.

```python
from pathlib import Path
from typing import List, Any
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader

def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the data directory and convert to LangChain document structure.
    Supported: PDF, TXT, CSV, Excel, Word, JSON
    """
    # Use project root data folder
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents = []

    # PDF files
    pdf_files = list(data_path.glob('**/*.pdf'))
    print(f"[DEBUG] Found {len(pdf_files)} PDF files: {[str(f) for f in pdf_files]}")
    for pdf_file in pdf_files:
        print(f"[DEBUG] Loading PDF: {pdf_file}")
        try:
            loader = PyPDFLoader(str(pdf_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {pdf_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load PDF {pdf_file}: {e}")

    # TXT files
    txt_files = list(data_path.glob('**/*.txt'))
    print(f"[DEBUG] Found {len(txt_files)} TXT files: {[str(f) for f in txt_files]}")
    for txt_file in txt_files:
        print(f"[DEBUG] Loading TXT: {txt_file}")
        try:
            loader = TextLoader(str(txt_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} TXT docs from {txt_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load TXT {txt_file}: {e}")

    # CSV files
    csv_files = list(data_path.glob('**/*.csv'))
    print(f"[DEBUG] Found {len(csv_files)} CSV files: {[str(f) for f in csv_files]}")
    for csv_file in csv_files:
        print(f"[DEBUG] Loading CSV: {csv_file}")
        try:
            loader = CSVLoader(str(csv_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} CSV docs from {csv_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load CSV {csv_file}: {e}")

    # Excel files
    xlsx_files = list(data_path.glob('**/*.xlsx'))
    print(f"[DEBUG] Found {len(xlsx_files)} Excel files: {[str(f) for f in xlsx_files]}")
    for xlsx_file in xlsx_files:
        print(f"[DEBUG] Loading Excel: {xlsx_file}")
        try:
            loader = UnstructuredExcelLoader(str(xlsx_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Excel docs from {xlsx_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load Excel {xlsx_file}: {e}")

    # Word files
    docx_files = list(data_path.glob('**/*.docx'))
    print(f"[DEBUG] Found {len(docx_files)} Word files: {[str(f) for f in docx_files]}")
    for docx_file in docx_files:
        print(f"[DEBUG] Loading Word: {docx_file}")
        try:
            loader = Docx2txtLoader(str(docx_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} Word docs from {docx_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load Word {docx_file}: {e}")

    # JSON files
    json_files = list(data_path.glob('**/*.json'))
    print(f"[DEBUG] Found {len(json_files)} JSON files: {[str(f) for f in json_files]}")
    for json_file in json_files:
        print(f"[DEBUG] Loading JSON: {json_file}")
        try:
            loader = JSONLoader(str(json_file))
            loaded = loader.load()
            print(f"[DEBUG] Loaded {len(loaded)} JSON docs from {json_file}")
            documents.extend(loaded)
        except Exception as e:
            print(f"[ERROR] Failed to load JSON {json_file}: {e}")

    print(f"[DEBUG] Total loaded documents: {len(documents)}")
    return documents

# Example usage
if __name__ == "__main__":
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")
    print("Example document:", docs[0] if docs else None)
```

**What it does.**

- `load_all_documents(data_dir: str) -> List[Any]` takes the folder name and returns a list of Documents. The docstring says it loads every supported file and converts it to LangChain's Document structure, because chunking can only be applied once everything is in that form.
- `Path(data_dir).resolve()` turns the relative folder name into an absolute path. The `[DEBUG]` prints show you what it is doing.
- Each block follows the same pattern: `data_path.glob('**/*.pdf')` finds the files, a loop creates the matching loader, `loader.load()` reads the file, and `documents.extend(loaded)` adds the result to the master list. A `try`/`except` prints an error and moves on if a file cannot be read.
- He walks through the **PDF block**. For text, CSV and the other formats he leaves the work to you, as an assignment. He points out that GitHub Copilot offers very similar code for those blocks, and says he wants you to write them yourself. The repository file contains them all, so it is shown whole above.
- The `if __name__ == "__main__":` block at the bottom is a quick self-test.

**A slip caught on camera.** The first draft of the function did not return anything, which he notices when he is about to print the result. He goes back and adds `return documents` at the end.

**Test through `app.py`.** He creates `app.py` at the project root and, since `search.py` and the vector store do not exist yet, imports only what is ready.

```python
from src.data_loader import load_all_documents

## Example usage
if __name__ == "__main__":
    docs=load_all_documents("data")
    print(docs)
```

Running `python app.py` from the project root prints the debug lines (`Found 4 PDF files`, one line per file saying how many pages loaded) followed by the Document list. That confirms the PDF part is working and everything is in Document form.

:::warning The other loaders need extras
Only the PDF branch is exercised in the video. The other branches are not ready to run on a clean install. `Docx2txtLoader` needs the `docx2txt` package, `UnstructuredExcelLoader` needs `unstructured` and `openpyxl`, and `JSONLoader` **requires** a `jq_schema` argument (and the `jq` package), so `JSONLoader(str(json_file))` as written will fail and print an `[ERROR]` line. The `try`/`except` hides the crash, but the file is skipped. Treat those blocks as the starting point of your assignment rather than finished code.
:::

### `embedding.py`

The next stage chunks the loaded Documents and embeds the chunks. It repeats what the notebook did, but as a class.

```python
from typing import List, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
from src.data_loader import load_all_documents

class EmbeddingPipeline:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model = SentenceTransformer(model_name)
        print(f"[INFO] Loaded embedding model: {model_name}")

    def chunk_documents(self, documents: List[Any]) -> List[Any]:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"[INFO] Split {len(documents)} documents into {len(chunks)} chunks.")
        return chunks

    def embed_chunks(self, chunks: List[Any]) -> np.ndarray:
        texts = [chunk.page_content for chunk in chunks]
        print(f"[INFO] Generating embeddings for {len(texts)} chunks...")
        embeddings = self.model.encode(texts, show_progress_bar=True)
        print(f"[INFO] Embeddings shape: {embeddings.shape}")
        return embeddings

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data")
    emb_pipe = EmbeddingPipeline()
    chunks = emb_pipe.chunk_documents(docs)
    embeddings = emb_pipe.embed_chunks(chunks)
    print("[INFO] Example embedding:", embeddings[0] if len(embeddings) > 0 else None)
```

**What it does.**

- `EmbeddingPipeline.__init__` takes the model name (`all-MiniLM-L6-v2`), the chunk size (1000) and the overlap (200), stores them, and loads the `SentenceTransformer` once, printing which model it loaded.
- `chunk_documents(documents)` builds a `RecursiveCharacterTextSplitter` from those settings, with `length_function=len` and the same four separators, calls `split_documents`, and prints how many documents became how many chunks.
- `embed_chunks(chunks)` takes the chunk texts and calls `self.model.encode(texts, show_progress_bar=True)`, then prints and returns the embeddings array.
- The self-test at the bottom runs load, chunk and embed in sequence.

While writing `embed_chunks` the editor's AI assistant proposed a method called `embed_documents`. He renamed it `embed_chunks` because the method embeds **chunks**, not raw documents. The order of calls is: load all documents, then `chunk_documents`, then `embed_chunks`.

**Test through `app.py`.** He extends `app.py` to run the new stage:

```python
from src.data_loader import load_all_documents
from src.embedding import EmbeddingPipeline

## Example usage
if __name__ == "__main__":
    docs=load_all_documents("data")
    chunks=EmbeddingPipeline().chunk_documents(docs)
    chunkvectors=EmbeddingPipeline().embed_chunks(chunks)
    print(chunkvectors)
```

He forgot at first to call `chunk_documents` before `embed_chunks`, added it, and ran the file. The run printed the four PDF loads, `[INFO] Split 64 documents into 359 chunks.`, the model load, `Generating embeddings for 359 chunks...`, a progress bar of 12 batches, and the array of numbers, a 359 by 384 matrix. It takes a little time because everything is loaded again.

:::tip Create the pipeline object once
`EmbeddingPipeline()` is written twice in that test, so the embedding model is loaded twice, which you can see as two `Loaded embedding model` lines. Assign it to a variable (`pipe = EmbeddingPipeline()`) and call both methods on it.
:::

### `vectorstore.py`

Next the vectors need a home that survives restarts. Here he switches from Chroma to **FAISS**, Meta's open-source library for fast vector search, and stores the index and the chunk text in a folder.

```python
import os
import faiss
import numpy as np
import pickle
from typing import List, Any
from sentence_transformers import SentenceTransformer
from src.embedding import EmbeddingPipeline

class FaissVectorStore:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", chunk_size: int = 1000, chunk_overlap: int = 200):
        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)
        self.index = None
        self.metadata = []
        self.embedding_model = embedding_model
        self.model = SentenceTransformer(embedding_model)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any]):
        print(f"[INFO] Building vector store from {len(documents)} raw documents...")
        emb_pipe = EmbeddingPipeline(model_name=self.embedding_model, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = emb_pipe.chunk_documents(documents)
        embeddings = emb_pipe.embed_chunks(chunks)
        metadatas = [{"text": chunk.page_content} for chunk in chunks]
        self.add_embeddings(np.array(embeddings).astype('float32'), metadatas)
        self.save()
        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(self, embeddings: np.ndarray, metadatas: List[Any] = None):
        dim = embeddings.shape[1]
        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        if metadatas:
            self.metadata.extend(metadatas)
        print(f"[INFO] Added {embeddings.shape[0]} vectors to Faiss index.")

    def save(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        faiss.write_index(self.index, faiss_path)
        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)
        print(f"[INFO] Saved Faiss index and metadata to {self.persist_dir}")

    def load(self):
        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")
        self.index = faiss.read_index(faiss_path)
        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)
        print(f"[INFO] Loaded Faiss index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index": idx, "distance": dist, "metadata": meta})
        return results

    def query(self, query_text: str, top_k: int = 5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)

# Example usage
if __name__ == "__main__":
    from data_loader import load_all_documents
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
    store.load()
    print(store.query("What is attention mechanism?", top_k=3))
```

**What it does.**

- `FaissVectorStore.__init__` creates the persistence folder (default `faiss_store`), prepares an empty index and an empty `metadata` list, loads the embedding model, and records the chunk size and overlap.
- `build_from_documents(documents)` is the one-call build. It creates an `EmbeddingPipeline`, chunks the documents, embeds the chunks, builds one metadata entry per chunk containing its text, calls `add_embeddings` with the vectors cast to `float32` (the type FAISS requires), and saves.
- `add_embeddings` creates the index the first time with `faiss.IndexFlatL2(dim)`, a flat index that compares a query against every vector using **L2 distance**, then adds the vectors and extends the metadata list.
- `save` writes `faiss.index` (the vectors) with `faiss.write_index` and `metadata.pkl` (the chunk texts) with `pickle`, both inside the persistence folder. He explains it this way: the metadata goes in the pickle file and the vector store goes in the index file.
- `load` reads both files back in binary mode.
- `search(query_embedding, top_k)` asks FAISS for the nearest vectors and returns, for each hit, its index, its distance and its metadata.
- `query(query_text, top_k)` embeds the question with the same model, casts it to `float32`, and calls `search`.

Because FAISS returns **L2 distances**, a **smaller** number means a closer match. These are not similarity scores, so do not apply `1 - distance` to them.

:::note Two cautions about this store
First, the metadata kept for each chunk is only `{"text": ...}`. The file name and page number that the loaders recorded are **dropped**, so this store cannot produce the source and page citations that the notebook pipeline did. Store `chunk.metadata` alongside the text if you want citations. Second, when you ask for more results than the index holds, FAISS pads the answer with the index `-1`. The guard `idx < len(self.metadata)` does not catch `-1`, so such a hit would silently pick up the **last** chunk. Filter with `0 <= idx < len(self.metadata)`.
:::

**Test through `app.py`: build, then load and query.** He rewrites `app.py` to use the store. The first run builds the index from the documents. He then comments that line out, because rebuilding every time wastes minutes, and uses `load` instead.

```python
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore

## Example usage
if __name__ == "__main__":
    docs=load_all_documents("data")
    store=FaissVectorStore("faiss_store")
    store.build_from_documents(docs)
```

The first run loaded the PDFs, split 64 documents into 359 chunks, generated embeddings (`Embeddings shape: (359, 384)`), added 359 vectors to the FAISS index, and saved the index and metadata. A new `faiss_store` folder appears containing `faiss.index` and `metadata.pkl`.

The next version of `app.py` reuses what is on disk:

```python
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore

## Example usage
if __name__ == "__main__":
    #docs=load_all_documents("data")
    store=FaissVectorStore("faiss_store")
    #store.build_from_documents(docs)
    store.load()
    print(store.query("What is attention mechanism?", top_k=3))
```

He explains that you only rebuild when you have new documents, and he suggests you could add a condition for that. Running it prints `Loaded Faiss index and metadata from faiss_store` followed by `Querying vector store for: 'What is attention mechanism?'` and a list of three hits. In the video the top hit was index 12 with a distance of about 0.73 and the text `3.2 Attention ... An attention function can be described as mapping a query and a set of key-value pairs to an output ...`, the second was index 49 with a distance of about 0.86, and so on. The best match is the chunk you saw in the notebook, which confirms the two stores agree.

### `search.py` and the final `app.py`

The last stage joins retrieval to an LLM. He says he will not go line by line, because the idea is the one from the notebook, and points you to the notebook for the details.

```python
import os
from dotenv import load_dotenv
from src.vectorstore import FaissVectorStore
from langchain_groq import ChatGroq

load_dotenv()

class RAGSearch:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "gemma2-9b-it"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        # Load or build vectorstore
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            from src.data_loader import load_all_documents
            docs = load_all_documents("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        groq_api_key = os.getenv("GROQ_API_KEY")
        self.llm = ChatGroq(groq_api_key=groq_api_key, model_name=llm_model)
        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])
        return response.content

# Example usage
if __name__ == "__main__":
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
```

**What it does.**

- `load_dotenv()` at the top loads `.env`, as before.
- `RAGSearch.__init__` creates the `FaissVectorStore`, checks whether `faiss.index` and `metadata.pkl` exist, and either **builds** the store from the `data` folder or **loads** the existing one. Then it creates the Groq chat model with the model name `gemma2-9b-it`.
- `search_and_summarize(query, top_k)` calls `self.vectorstore.query(query, top_k)`, picks the stored text out of every hit, joins the texts with blank lines into the context, and returns "No relevant documents found." if the context is empty. Otherwise it builds a prompt asking for a **summary** of the context for the query, calls `self.llm.invoke([prompt])`, and returns `response.content`.

The pipeline is data loader, embedding, vector store, search, exactly the stages on the board, but each in its own file with one responsibility.

:::note Differences from the file on screen and in the repository
In the video, and in the repository copy, the Groq key is written inside `search.py`, as a literal string in the video and as an empty string in the repository. The version above reads `GROQ_API_KEY` from the environment instead, which is the safe way. The repository's build-on-first-use branch also says `from data_loader import load_all_documents`, which fails when you run from the project root because `data_loader` is not on the import path; the version above uses `from src.data_loader import load_all_documents`. The model name `gemma2-9b-it` is the one he used. Substitute a current Groq model, as noted earlier.
:::

Finally he changes `app.py` to its finished form. He imports `RAGSearch`, no longer needs the earlier calls, comments out the build and query lines, and asks the question through the whole stack.

```python
from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch

# Example usage
if __name__ == "__main__":
    
    docs = load_all_documents("data")
    store = FaissVectorStore("faiss_store")
    #store.build_from_documents(docs)
    store.load()
    #print(store.query("What is attention mechanism?", top_k=3))
    rag_search = RAGSearch()
    query = "What is attention mechanism?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
```

Running it prints the model load lines and `Loaded Faiss index and metadata from faiss_store`, the question goes through the store and then to the LLM, and the summary prints. The instructor says the output appears and that it works if the LLM is configured. The printed summary scrolls past too quickly to read in the recording, so no output is reproduced here. Run it on your own data to see the result.

(You will see `Loaded embedding model` twice in the log. `app.py` builds its own `FaissVectorStore`, and `RAGSearch` builds another inside itself, and each one loads the embedding model. The `store` created in `app.py` is no longer needed once `RAGSearch` is used.)

## Wrap up, and what comes next

He closes by calling this a complete crash course on RAG. He repeats his belief that RAG is one of the most important use cases, because most companies are building RAG applications, which is why he thinks it is a super cool topic. He then moves directly to the next topic in the long course, **vectorless RAG**, a trending approach that retrieves without a vector database. Instead of chunking, embedding and storing in a vector DB, it uses a different retrieval method, which the next chapter covers.

## Things to change before you reuse this code

The chapter flagged these as it went. This table collects them in one place.

| Where | Problem | What to do |
| --- | --- | --- |
| `VectorStore` and `RAGRetriever` (Chroma) | Default distance is L2, but the code computes `1 - distance` as a cosine similarity, so scores are wrong and thresholds drop good results | Create the collection with cosine space (`"hnsw:space": "cosine"`) and rebuild |
| `VectorStore.add_documents` | Random ids mean a second run duplicates every record | Use stable ids with `upsert`, or delete the store before a re-run |
| `rag_simple`, `rag_advanced`, `AdvancedRAGPipeline` | `prompt.format(...)` is called on an already formatted string and breaks on `{` or `}` in the text | Pass `prompt` directly to `llm.invoke` |
| `AdvancedRAGPipeline` | "Streaming" prints the prompt, not the reply | Use `llm.stream(...)` for real streaming |
| Groq key | Pasted into a cell and into `search.py` during the video | Read it from the environment; revoke any key shown on screen |
| `gemma2-9b-it` | Retired by Groq | Use a model from Groq's current list |
| `langchain.text_splitter`, `langchain.document_loaders`, `langchain.schema`, `langchain.prompts` | Old import paths | `langchain_text_splitters`, `langchain_community.document_loaders`, `langchain_core.documents`, `langchain_core.prompts` |
| `data_loader.py` | `JSONLoader` needs `jq_schema`; Word and Excel loaders need extra packages | Install the extras and pass the schema |
| `FaissVectorStore` | Drops file and page metadata; `-1` index bug | Store `chunk.metadata`; filter `idx >= 0` |
| `search.py` | `from data_loader import ...` fails from the project root | `from src.data_loader import ...` |

## Also in the repository, not taught in this section

:::note Not from this part of the video
Everything under this heading is an **addition** drawn from the repository files `agenticrag/1-agenticrag.ipynb`, `typesense.ipynb` and `books.jsonl`. The instructor promises agentic RAG for a later part of the course, and `typesense.ipynb` is also open briefly, but neither notebook is taught in this part of the course. They are included so the chapter covers the folder it is based on.
:::

### Agentic RAG with LangGraph

In ordinary RAG every question follows the same path: retrieve, then generate. In the notebook version of **agentic RAG** the program first **decides** whether retrieval is needed at all, and only then retrieves. The steps are nodes of a LangGraph state graph, which you met in the previous chapter on LangGraph: a `decide` node, a conditional edge, a `retrieve` node and a `generate` node.

<Infographic
  src="/img/agentic-course/04-agentic-rag-graph.svg"
  alt="An agentic RAG graph: start, a decide node, a branch on whether retrieval is needed leading either to retrieve and then generate, or directly to generate, then end"
  caption="Explanatory board of the repository notebook, added to this chapter (not shown in the video)."
/>

The notebook uses OpenAI models (`gpt-4.1` and `OpenAIEmbeddings`), so it needs an `OPENAI_API_KEY` in `.env`. The code below follows the notebook, with the old import paths replaced by the current ones.

```python
import os
from typing import TypedDict, List
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1", temperature=0)
embeddings = OpenAIEmbeddings()
```

The **state** is a typed dictionary that every node reads and updates:

```python
class AgentState(TypedDict):
    question: str
    documents: List[Document]
    answer: str
    needs_retrieval: bool
```

A tiny knowledge base of four sentences is embedded into a FAISS index and exposed as a retriever:

```python
### Sample Docuemnt And VectorStore
# Sample documents for demonstration
sample_texts = [
    "LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain with the ability to coordinate multiple chains across multiple steps of computation in a cyclic manner.",
    "RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with text generation. It retrieves relevant documents and uses them to provide context for generating more accurate responses.",
    "Vector databases store high-dimensional vectors and enable efficient similarity search. They are commonly used in RAG systems to find relevant documents based on semantic similarity.",
    "Agentic systems are AI systems that can take actions, make decisions, and interact with their environment autonomously. They often use planning and reasoning capabilities."
]

documents=[Document(page_content=text) for text in sample_texts]

##create vector store
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(k=3)
```

Three node functions follow. `decide_retrieval` sets `needs_retrieval` using a keyword check, `retrieve_documents` fills `documents` from the retriever, and `generate_answer` builds a prompt with the context if documents exist, or answers directly if they do not.

```python
def decide_retrieval(state: AgentState) -> AgentState:
    """
    Decide if we need to retrieve documents based on the question
    """
    question = state["question"]
    
    # Simple heuristic: if question contains certain keywords, retrieve
    retrieval_keywords = ["what", "how", "explain", "describe", "tell me"]
    needs_retrieval = any(keyword in question.lower() for keyword in retrieval_keywords)
    
    return {**state, "needs_retrieval": needs_retrieval}
```

```python
def retrieve_documents(state: AgentState) -> AgentState:
    """
    Retrieve relevant documents based on the question
    """
    question = state["question"]
    documents = retriever.invoke(question)
    
    return {**state, "documents": documents}
```

```python
def generate_answer(state: AgentState) -> AgentState:
    """
    Generate an answer using the retrieved documents or direct response
    """
    question = state["question"]
    documents = state.get("documents", [])
    
    if documents:
        # RAG approach: use documents as context
        context = "\n\n".join([doc.page_content for doc in documents])
        prompt = f"""Based on the following context, answer the question:

Context:
{context}

Question: {question}

Answer:"""
    else:
        # Direct response without retrieval
        prompt = f"Answer the following question: {question}"
    
    response = llm.invoke(prompt)
    answer = response.content
    
    return {**state, "answer": answer}
```

A small router function chooses the next node, and the graph is assembled and compiled.

```python
def should_retrieve(state: AgentState) -> str:
    """
    Determine the next step based on retrieval decision
    """
    if state["needs_retrieval"]:
        return "retrieve"
    else:
        return "generate"
```

```python
# Create the state graph
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("decide", decide_retrieval)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("generate", generate_answer)

# Set entry point
workflow.set_entry_point("decide")

# Add conditional edges
workflow.add_conditional_edges(
    "decide",
    should_retrieve,
    {
        "retrieve": "retrieve",
        "generate": "generate"
    }
)

# Add edges
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile the graph
app = workflow.compile()
app
```

Finally, a helper runs the graph and two questions are tried.

```python
def ask_question(question: str):
    """
    Helper function to ask a question and get an answer
    """
    initial_state = {
        "question": question,
        "documents": [],
        "answer": "",
        "needs_retrieval": False
    }
    
    result = app.invoke(initial_state)
    return result
```

```python
question1 = "What is LangGraph?"
result1 = ask_question(question1)
result1["answer"]

question2 = "How does RAG work?"
result2 = ask_question(question2)
print(f"Question: {question2}")
print(f"Retrieved documents: {len(result2['documents'])}")
print(f"Answer: {result2['answer']}")
```

In the notebook's saved output, the second question retrieved 4 documents and produced a numbered explanation of retrieval and generation.

:::warning Weak spots in this notebook
The "decision" is a keyword test on the words `what`, `how`, `explain`, `describe` and `tell me`. It is a substring check, so a question like "Show me the roadmap" counts as needing retrieval because "show" contains "how". A real agentic system asks the **LLM** to decide, or lets it call a retrieval tool. Also, `vectorstore.as_retriever(k=3)` does not set the number of results: `k` belongs inside `search_kwargs`, so write `as_retriever(search_kwargs={"k": 3})`. As written, the default of 4 results applies, which is why all four sample documents came back.
:::

### Typesense and `books.jsonl`

**Typesense** is an open-source search engine that can be run yourself or used as a cloud service. It does fast keyword search with typo tolerance, filtering and faceting, and it can also store vectors, so it can sit under a RAG application. The repository's `typesense.ipynb` shows two things: Typesense on its own, using `books.jsonl`, a file of 9,979 book records (title, authors, publication year, ratings and an image URL), and then Typesense as a LangChain vector store.

Create a client for your own Typesense Cloud cluster. Take the host and key from your Typesense dashboard and keep them in environment variables.

```python
import os
import typesense

client = typesense.Client({
    'nodes': [{
        'host': os.environ["TYPESENSE_HOST"],   # xxx.a1.typesense.net for Typesense Cloud
        'port': '443',
        'protocol': 'https'
    }],
    'api_key': os.environ["TYPESENSE_API_KEY"],
    'connection_timeout_seconds': 2
})
```

Define a schema for a `books` collection and create it. Fields marked `facet` can be counted and grouped in results.

```python
books_schema = {
  'name': 'books',
  'fields': [
    {'name': 'title', 'type': 'string'},
    {'name': 'authors', 'type': 'string[]', 'facet': True},
    {'name': 'publication_year', 'type': 'int32', 'facet': True},
    {'name': 'ratings_count', 'type': 'int32'},
    {'name': 'average_rating', 'type': 'float'}
  ],
  'default_sorting_field': 'ratings_count'
}
print(client.collections.create(books_schema))
```

Import the JSON Lines file, one JSON record per line, and run a search. `query_by` lists the fields to search, and `sort_by` orders the hits.

```python
with open('books.jsonl', 'r', encoding='utf-8') as jsonl_file:
    data = jsonl_file.read()
    client.collections['books'].documents.import_(data)
```

```python
search_parameters={
    'q':"harry potter",
    'query_by':"title,authors",
    'sort_by':"ratings_count:desc"
}

client.collections['books'].documents.search(search_parameters)
```

The notebook then filters with `filter_by` (`publication_year:<1998`) and runs a typo-tolerant search with facet counts. The query `'experyment'`, misspelled on purpose, still finds books about experiments, and `facet_by: 'authors'` returns how many hits each author has.

```python
search_parameters = {
  'q'         : 'harry potter',
  'query_by'  : 'title',
  'filter_by' : 'publication_year:<1998',
  'sort_by'   : 'publication_year:desc'
}

client.collections['books'].documents.search(search_parameters)
```

```python
search_parameters = {
  'q'         : 'experyment',
  'query_by'  : 'title',
  'facet_by'  : 'authors',
  'sort_by'   : 'average_rating:desc'
}

client.collections['books'].documents.search(search_parameters)
```

The second half uses LangChain's Typesense vector store with Hugging Face embeddings and a Groq model. It loads a text file, splits it, embeds it, and builds a retriever.

```python
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Typesense
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

loader = TextLoader("test.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()

docsearch = Typesense.from_documents(
    docs,
    embeddings,
    typesense_client_params={
        "host": os.environ["TYPESENSE_HOST"],
        "port": "443",
        "protocol": "https",
        "typesense_api_key": os.environ["TYPESENSE_API_KEY"],
        "typesense_collection_name": "lang-chain",
    },
)

query = "What is artificial intelligence"
found_docs = docsearch.similarity_search(query)
print(found_docs[0].page_content)

retriever = docsearch.as_retriever()
retriever.invoke("Artificial intelligence indepth explanation")[0]
```

:::note Changes from the repository notebook
The repository version imports `HuggingFaceEmbeddings` from `langchain.embeddings`, which emits a deprecation warning; the up-to-date package is `langchain-huggingface`. It also contains a real Typesense host and API key typed into the cells. Those are not reproduced here: use your own cluster, keep the key in the environment, and rotate any key that has been committed to a public repository.
:::

## What you can now do

- I can explain, in plain words, what RAG is and why a plain LLM hallucinates on data after its cut-off or outside its training set.
- I can explain why fine-tuning is a poor fit for private data that keeps changing, and why RAG is the alternative the instructor picks.
- I can draw the two pipelines, data ingestion (parse, chunk, embed, store) and retrieval (embed query, search, add context to a prompt, generate), and name the retrieval, augmentation and generation steps.
- I can describe a LangChain `Document`, its `page_content` and `metadata`, and say why metadata matters for filtering and citations.
- I can load text files with `TextLoader`, a folder with `DirectoryLoader`, and PDFs with `PyPDFLoader` or `PyMuPDFLoader`, and fix the `show_progress` and `encoding` errors met on camera.
- I can split Documents with `RecursiveCharacterTextSplitter`, and explain `chunk_size`, `chunk_overlap` and the separator order.
- I can embed chunks with `all-MiniLM-L6-v2` and store them in a persistent ChromaDB collection, and I know why re-running the add step duplicates records.
- I can build a retriever that embeds a query and returns ranked chunks, and I can spot and fix the Chroma distance versus similarity mistake.
- I can turn retrieved context and a prompt into an answer with Groq, return sources and a confidence value, and explain why the demo's streaming is not real streaming.
- I can structure the same pipeline as a `src/` package with `data_loader.py`, `embedding.py`, `vectorstore.py`, `search.py` and `app.py`, save and reload a FAISS index, and keep API keys out of the code.
