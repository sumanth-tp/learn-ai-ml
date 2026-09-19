---
id: document-loaders
title: "Document Loaders in LangChain | Generative AI using LangChain | Video 10 | CampusX"
sidebar_label: "12 · Document loaders"
sidebar_position: 12
slug: /genai/document-loaders
description: "The first RAG component — TextLoader, PyPDFLoader, DirectoryLoader, WebBaseLoader and CSVLoader, plus lazy loading and custom loaders."
tags: [langchain, rag, document-loaders, pdf, csv, web-scraping, lazy-loading]
---

> **Video 12 of 21** (playlist video 10) · [Watch on YouTube](https://www.youtube.com/watch?v=bL92ALSZ2Cg)
> Notes follow the video section by section.

## A change of plan

The original plan was to teach the **memory** component after chains and runnables. But research showed that the LangChain team is gradually **removing the memory component from LangChain and moving it into LangGraph**. So memory will be taught alongside LangGraph, when that playlist starts.

Instead, today starts something completely new: **building RAG-based applications using LangChain.**

## Recap

So far this playlist has focused on two things. First, the **important components** of LangChain — models, prompts, chains — each explained in detail with hands-on code. Second, the **core concepts** of LangChain, like runnables, covered across two or three videos so that you are comfortable whenever you see them again.

At this point our fundamentals are clear and we are ready to build any kind of LLM-based application. Next comes RAG, and that is the focus of the next four videos.

## What RAG is — a quick introduction

In the wave of GenAI so far, the biggest use case is **chatbots**. Take ChatGPT: you open the website, enter your text, ask your question, press enter and get an instant response.

This mostly works fine, but there are cases where it cannot help you:

- **Current affairs.** ChatGPT is trained on past data and does not have information about what happened today or yesterday.
- **Your personal data.** Ask a question about the emails you have received in the last week — obviously it cannot answer, because it has not seen that data.
- **Your company's documentation.** Same reason.

In all those situations, where the LLM does not have the data to answer you, **RAG-based applications help.**

What you do in a RAG-based application is provide an **external knowledge base** to your LLM. That knowledge base can be anything — your company's database, many PDFs, your personal documents. You connect it with the LLM. Now, when a user asks a question the LLM does not know, the LLM can go to the knowledge base, find the answer, and answer with its help.

> RAG is a technique that combines **information retrieval** with **language generation**, where a model retrieves relevant documents from a knowledge base and then uses them as context to generate accurate and grounded responses.

Information retrieval happens from the external knowledge base; language generation happens with the LLM.

### The benefits of RAG

- **Up-to-date information** from any LLM
- **Privacy.** Imagine you have to ask questions on your personal documents. One option is to upload the document to ChatGPT — but if it is confidential information, that is not a good idea. With RAG you can ask questions on your document **without uploading it**.
- **No limit on document size.** Suppose your document is 1 GB. The context length of ChatGPT will not allow it to read the entire document and answer. In RAG you process the entire document easily by dividing it into chunks.

That is why RAG-based applications are a very powerful trend in the industry right now.

## The plan for the RAG section

Rather than teaching how to build a RAG application directly, we first cover the **important components** of RAG-based applications, and then build one.

The four most important components:

1. **Document loaders** — today
2. **Text splitters**
3. **Vector databases**
4. **Retrievers**

By combining these four you create any RAG-based application. No matter how difficult the architecture, it is mostly made of these four.

## What document loaders are

> Document loaders are components in LangChain used to load data from various sources into a standardised format, usually as **Document objects**, which can then be used for chunking, embedding, retrieval and generation.

The basic idea: to build RAG-based applications you have to load data, and that data can exist in different sources — a PDF, a text file, a database, a cloud provider. There are hundreds of sources.

We have to make sure that when we bring data from any of these sources, it comes in a **common, standardised format** that you can use easily with any other LangChain component.

So the team created a standardised format and named it **Document**. Whenever you fetch data from any source using a document loader, it always appears in Document format.

**Every Document object contains two things:**

| Field | What it holds |
|---|---|
| `page_content` | the actual content of the data |
| `metadata` | information around it — the source, where the file came from, when it was created, when it was first modified, the author's name |

So document loaders are utilities whose job is to fetch data from different data sources and convert it into a standardised format: the Document object.

All document loaders are found in the **`langchain_community`** package.

## 1. TextLoader

The simplest document loader in LangChain. Its work is very simple: it picks up text files and brings them into LangChain as Document objects.

Ideally you use it when you need to process a log file, a code snippet, or a transcript — for example the transcript of a YouTube video.

```python
# text_loader.py
from langchain_community.document_loaders import TextLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

loader = TextLoader("cricket.txt", encoding="utf-8")

docs = loader.load()

print(type(docs))
print(len(docs))
print(docs[0])
print(docs[0].page_content)
print(docs[0].metadata)
```

You create a loader object, passing the **path** of your text file. You can also tell it which **encoding** format your file uses — here UTF-8, needed because there are some special characters. There is a good chance you do not need to specify encoding for your own file.

The loader object has a **`load`** function, which loads your text file as a Document into memory.

:::note Every loader returns a list
Print the type of `docs` and you see it is a Python **list**. Whatever document loader you use — `TextLoader`, `PyPDFLoader`, anything — **every document loader in LangChain loads a document as a list of documents.** Your document is divided into multiple parts, and all those parts are stored in a list and given to you.

In this particular case there is only one document in the list, which you extract with `docs[0]`. Its type is `Document`, and inside it are `page_content` and `metadata`, which you can extract separately.
:::

### Using it in a chain

```python
model = ChatOpenAI()

prompt = PromptTemplate(
    template="Write a summary for the following poem - \n {poem}",
    input_variables=["poem"],
)

parser = StrOutputParser()

chain = prompt | model | parser

print(chain.invoke({"poem": docs[0].page_content}))
```

The whole flow becomes very simple. If you want, you could even wrap `loader.load()` and the page-content extraction in a `RunnableLambda` so that they also become part of the chain — that is how much flexibility you have.

## 2. PyPDFLoader

The most-used document loader. It reads your PDF files and converts them into Document objects.

**Its biggest feature: it works page by page.** If you have a 25-page PDF, sending it to `PyPDFLoader` creates **25 Document objects** — so you get a list of 25 Documents. For each page you get a Document with its own page content and its own metadata, including the page number and source.

```python
# pdf_loader.py
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("dl-curriculum.pdf")

docs = loader.load()

print(len(docs))                 # 23 for a 23-page PDF
print(docs[0].page_content)
print(docs[1].metadata)
```

Make sure you have `pypdf` installed, otherwise the code will not work.

**The best part about LangChain:** you can use any document loader and the usage format is exactly the same.

### Choosing the right PDF loader

`PyPDFLoader` internally uses the **PyPDF** library to read PDF files, and that is why it is **not great with scanned PDFs and complex layouts**. If you have simple textual PDF files, use it. If you are creating a PDF from a photo, you need something else.

| Loader | Use it for |
|---|---|
| `PyPDFLoader` | simple text-based PDFs |
| `PDFPlumberLoader` | PDFs with a lot of tabular structure, where you want to extract table data |
| `UnstructuredPDFLoader` | PDFs containing scanned images, and structure extraction |
| `AmazonTextractPDFLoader` | scanned images needing high-quality OCR |
| `PyMuPDFLoader` | PDFs with a lot of layout |

The full list is in the LangChain documentation. Do not read them all one by one — it varies project to project. Know the basic concepts, and when you need a specific loader for a project, come back and refer to the documentation.

## 3. DirectoryLoader

So far we have loaded a single text file or a single PDF. **What if you have a folder containing multiple PDFs or text files and you want to load all of them together?**

The answer is `DirectoryLoader` — a document loader that helps you load multiple documents from within a single directory.

```python
# directory_loader.py
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="books",
    glob="*.pdf",
    loader_cls=PyPDFLoader,
)

docs = loader.load()

print(len(docs))
print(docs[0].page_content)
print(docs[325].metadata)
```

You tell it three things:

1. The **path** of your directory
2. Which files to load, via the **`glob`** parameter — a pattern; all files satisfying it are picked up
3. Which **loader class** to use — here `PyPDFLoader`, since all the files are PDFs

### Glob patterns

| Pattern | Meaning |
|---|---|
| `**/*.txt` | all text files inside all subfolders |
| `*.pdf` | all PDF files in the root directory |
| `data/*.csv` | all CSV files inside the `data` folder |
| `**/*` | all files inside all subfolders |

With three books of 326, 392 and 468 pages, you get **1186 Documents** — because those three numbers add up to 1186. Every page of every PDF becomes a Document.

Check the metadata and you see when the file was created, when it was last modified, the book name in `source`, which PDF it is part of, how many pages it has, and which page this is. Since indexing starts at zero, page index 325 is the last page of the first book, and 326 is the first page of the second.

## `load` vs `lazy_load`

Did you notice the `DirectoryLoader` code took some time to run? It was loading three PDFs simultaneously — say around 10 seconds. **Imagine if that folder had 100 PDFs of the same size.** It would take a very long time.

And a second problem: **you are loading all three PDFs into memory at once**, into RAM, so you can run operations on them. With three PDFs that is possible. With 100 or 500, it is not.

To solve both problems, LangChain has **lazy loading**.

Every document loader has a `load` function, and **every document loader also has a `lazy_load` function**. Both load the document into memory, but they do the job differently.

| | **`load()`** | **`lazy_load()`** |
|---|---|---|
| Strategy | **eager loading** — loads everything at once | **on demand** — one document at a time |
| Returns | a **list** of Document objects | a **generator** of Document objects |
| Memory | all documents loaded at once | one document in memory at a time, then removed |
| Use when | the number of documents is small, and you need everything in memory at once | you are working with a large number of documents or files, or you want stream processing without using a lot of memory |

With `load`, a 500-page PDF creates 500 Documents and loads them all into memory at once, giving you a list to work with. With `lazy_load` you get a **generator** — you load one document at a time, run whatever operations you want on it, and then it is removed from memory before the next one loads.

```python
# Eager: a long pause, then everything prints at once
for document in loader.load():
    print(document.metadata)

# Lazy: printing starts immediately, memory stays flat
for document in loader.lazy_load():
    print(document.metadata)
```

Run the eager version over 1186 documents and nothing happens for a long time — the code is creating all the documents in memory first, then you get a list, then you loop. Run the lazy version and it **has already started printing**: each document comes into memory, we print its metadata, it is removed, the next one comes. The whole thing takes consistent time.

**In a nutshell:** if you have a lot of documents that are not possible to load into memory at once, you have the option of lazy loading.

## 4. WebBaseLoader

> `WebBaseLoader` is a document loader in LangChain used to load and extract text content from web pages.

Suppose you have a product page on a shopping site with a lot of text written in many different places, and you want to perform queries on it.

**How it works internally:** it uses two Python libraries. **Requests**, to make an HTTP request to that web page, and **BeautifulSoup**, to understand the HTML structure of the page and bring it into LangChain in textual format.

You generally use it when working with a **static website** — a blog, a news articles site, or any public website.

**Limitations:** it works better with static pages. If your web page is very **JavaScript-heavy**, where a lot happens based on user action, this loader may not work well. There is a separate loader for that called `SeleniumURLLoader`.

```python
# web_loader.py
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

url = "https://example.com/product/some-laptop"

loader = WebBaseLoader(url)

docs = loader.load()

print(len(docs))
print(docs[0].page_content)
```

With BeautifulSoup the HTML tags are removed and the text inside is extracted.

**Note:** the number of documents you get is **one per URL**. But you have the flexibility to pass a **list of URLs** instead of a single one — then you get one Document per URL.

### Asking questions about the page

```python
model = ChatOpenAI()

prompt = PromptTemplate(
    template="Answer the following question \n {question} from the following text - \n {text}",
    input_variables=["question", "text"],
)

parser = StrOutputParser()

chain = prompt | model | parser

print(chain.invoke({
    "question": "What is the product that we are talking about?",
    "text": docs[0].page_content,
}))
```

:::tip A project idea
What if you create a **browser extension** where, as soon as you open any website, you open the plugin and chat about that page in real time? To implement it you build the extension, and behind the scenes you run an API talking to an LLM using LangChain's document loader. Not very complex, but it has a wow factor.
:::

## 5. CSVLoader

Used to load CSV files into LangChain. If you have a CSV file and want to run queries on it using an LLM, you use this loader.

**It creates a separate Document object for each row.**

```python
# csv_loader.py
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="Social_Network_Ads.csv")

docs = loader.load()

print(len(docs))       # 400 for 400 rows
print(docs[0])
print(docs[1])
```

With a CSV of five columns and 400 rows, you get **400** Documents. Print one and you see the page content as a string where each column name and its value is given, and in the metadata the source and the row number.

For a very large CSV you can use lazy loading, looping each row through a generator and performing operations on it. You can ask questions like *"what is the maximum value in a particular column?"* very easily.

This is a loader you will use in future if you work around data analysis.

## The full list of loaders

Many other document loaders exist. In the documentation you will find them in well-defined categories:

- **Web pages** — `WebBaseLoader`, sitemap loaders, browser-based agents
- **PDFs** — the list above
- **Cloud services** — loading data from S3, Azure, Dropbox, Google Drive
- **Social platforms**
- **Messaging services**
- **Productivity tools**
- **Common file types** — CSV, directory, JSON, and even a **YouTube transcript** loader

Click any link and you get its use case and related code. But again, **you do not need to read all the document loaders.** Learn on a project-by-project basis. If you really need to load YouTube transcripts in your next project, go and read that loader from the documentation. Otherwise there is no point reading them all.

## Custom document loaders

Suppose you are doing a project where your data source has no document loader available in LangChain. **In that case you can create your own custom document loader.**

You create a class, inherit it from the **`BaseLoader`** class, and add your own `load` and `lazy_load` functions with your custom logic.

In fact, all the document loaders that exist in LangChain are there **because** LangChain gives you this feature. Many programmers in the community created loaders for their own use cases and then added them to LangChain. That is why, over time, LangChain has such a pool of document loaders — and why they all live in the `langchain_community` package, because it is all developed by the community.

## Checklist

- [ ] I know what problem RAG solves and its three benefits
- [ ] I can name the four core RAG components
- [ ] I can name the two fields of a Document object
- [ ] I know every loader returns a list
- [ ] I can predict the Document count for a PDF, a CSV, a directory and a web page
- [ ] I can pick a PDF loader based on the document type
- [ ] I can explain glob patterns
- [ ] I can explain when `lazy_load` is required rather than preferred
- [ ] I know how to write a custom loader
