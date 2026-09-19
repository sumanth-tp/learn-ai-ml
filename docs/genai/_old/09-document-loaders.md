---
id: document-loaders
title: "Document Loaders"
sidebar_label: "Document loaders"
sidebar_position: 10
slug: /genai/document-loaders
description: "Bring data from any source into LangChain as Document objects — text, PDF, directories, web pages and CSV — plus lazy loading for large corpora."
tags: [langchain, rag, document-loaders, pdf, csv, web-scraping]
---

**In one line.** Document loaders pull data from any source and hand it back in one standard shape — the `Document` object — so every downstream component can work with it.

This is the first of four RAG components. The other three are [text splitters](/docs/genai/text-splitters), [vector stores](/docs/genai/vector-stores) and [retrievers](/docs/genai/retrievers).

## The `Document` object

Data lives everywhere — PDFs, text files, databases, cloud buckets, web pages, CSVs. If every source produced a different shape, every downstream component would need to handle every shape. So LangChain standardises on one:

```python
Document(
    page_content="the actual text",
    metadata={"source": "book.pdf", "page": 3, "author": "..."},
)
```

Two fields: the content, and everything else you know about it. Metadata matters more than it looks — it is what lets you later filter retrieval by source, page or date.

:::note Loaders always return a **list**
Every loader returns `list[Document]`, even when there is only one. A PDF gives one Document per page; a CSV gives one per row; a web page gives one per URL.
:::

All loaders live in `langchain_community.document_loaders`, because most were contributed by the community.

## TextLoader

The simplest. Use it for log files, code, transcripts.

```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("cricket.txt", encoding="utf-8")
docs = loader.load()

print(type(docs))            # list
print(len(docs))             # 1
print(docs[0].page_content)
print(docs[0].metadata)      # {'source': 'cricket.txt'}
```

Feeding it into a chain:

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()
prompt = PromptTemplate(template="Write a summary of the following poem.\n{poem}",
                        input_variables=["poem"])

chain = prompt | model | parser
print(chain.invoke({"poem": docs[0].page_content}))
```

## PyPDFLoader

The most-used loader. **One Document per page.**

```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("dl_curriculum.pdf")
docs = loader.load()

print(len(docs))              # 23 for a 23-page PDF
print(docs[0].page_content)
print(docs[0].metadata)       # source, page, total_pages, author, creation date...
```

Requires `pip install pypdf`.

### Choosing a PDF loader

`PyPDFLoader` handles plain textual PDFs. It struggles elsewhere, so LangChain ships alternatives:

| Loader | Best for |
|---|---|
| `PyPDFLoader` | simple text-based PDFs |
| `PDFPlumberLoader` | PDFs with tables you need to extract |
| `UnstructuredPDFLoader` | scanned images, complex structure |
| `AmazonTextractPDFLoader` | scanned documents needing high-quality OCR |
| `PyMuPDFLoader` | layout-heavy documents |

Match the loader to the document. A scanned PDF through `PyPDFLoader` yields empty or garbled text — and it is a genuinely confusing failure if you do not know why.

## DirectoryLoader

Load a whole folder at once.

```python
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path="books",
    glob="*.pdf",
    loader_cls=PyPDFLoader,
)
docs = loader.load()
print(len(docs))   # total pages across every PDF
```

Three arguments: the folder, a glob pattern, and which loader to apply to each match.

| Pattern | Matches |
|---|---|
| `*.pdf` | PDFs in the root folder |
| `**/*.txt` | text files in every subfolder, recursively |
| `data/*.csv` | CSVs inside `data/` |
| `**/*` | everything, recursively |

Load three PDFs of 326, 392 and 468 pages and you get 1186 Documents — one per page.

## `load` vs `lazy_load`

That `DirectoryLoader` call takes a noticeable pause, for two reasons: it processes every file before returning anything, and it holds all 1186 Documents in memory at once. Scale to 500 PDFs and both become fatal.

Every loader offers both methods:

| | `load()` | `lazy_load()` |
|---|---|---|
| Strategy | eager | on demand |
| Returns | a `list` of Documents | a **generator** |
| Memory | everything at once | one Document at a time |
| First output | after all files are processed | almost immediately |
| Use when | few documents, all needed together | many documents, or stream processing |

```python
# Eager: long pause, then everything prints at once
for document in loader.load():
    print(document.metadata)

# Lazy: printing starts immediately, memory stays flat
for document in loader.lazy_load():
    print(document.metadata)
```

**Rule of thumb:** if the corpus might not fit comfortably in RAM, use `lazy_load`.

## WebBaseLoader

Load a web page's text. Uses `requests` to fetch and BeautifulSoup to strip HTML.

```python
from langchain_community.document_loaders import WebBaseLoader

url = "https://example.com/product/some-laptop"
loader = WebBaseLoader(url)
docs = loader.load()

print(len(docs))             # 1 per URL
print(docs[0].page_content)
```

Pass a list of URLs to get one Document per URL.

**Limitation:** it works on static pages. JavaScript-heavy pages that build content client-side return little or nothing — use `SeleniumURLLoader` for those.

Combine with a chain to ask questions about a page:

```python
prompt = PromptTemplate(
    template="Answer the following question.\n{question}\nFrom the following text.\n{text}",
    input_variables=["question", "text"],
)
chain = prompt | model | parser
print(chain.invoke({
    "question": "What is the product being described?",
    "text": docs[0].page_content,
}))
```

:::tip Project idea
A browser extension that lets you chat with whatever page you are on: `WebBaseLoader` behind an API, an LLM in front. Modest complexity, high wow factor.
:::

## CSVLoader

**One Document per row.**

```python
from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path="social_network_ads.csv")
docs = loader.load()

print(len(docs))             # 400 for 400 rows
print(docs[0].page_content)  # "User ID: 15624510\nGender: Male\nAge: 19\n..."
print(docs[0].metadata)      # {'source': '...', 'row': 0}
```

Each row becomes a string of `column: value` pairs. For a large CSV, use `lazy_load`.

## Beyond these four

LangChain has hundreds of loaders, grouped by category: web pages, PDFs, cloud storage (S3, Azure, Dropbox, Google Drive), social platforms, messaging services, productivity tools, and common file types including JSON and YouTube transcripts.

Do not read them all. Learn the pattern — construct the loader, call `load()` or `lazy_load()`, get `list[Document]` — and look up a specific loader when a project needs it.

## Custom loaders

If no loader fits your source, write one. Inherit `BaseLoader` and implement `load` and `lazy_load`:

```python
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document

class MyCustomLoader(BaseLoader):
    def __init__(self, source):
        self.source = source

    def lazy_load(self):
        for record in fetch_from_my_source(self.source):
            yield Document(
                page_content=record["text"],
                metadata={"source": self.source, "id": record["id"]},
            )
```

This is exactly how the community built the hundreds of loaders that now ship with LangChain.

## Pitfalls

- **Expecting a single Document.** You always get a list.
- **Using `PyPDFLoader` on scanned PDFs.** Wrong tool; you get empty text.
- **Calling `load()` on a large corpus.** Memory blows up. Use `lazy_load`.
- **Forgetting `encoding="utf-8"`** on text files with special characters.
- **Expecting `WebBaseLoader` to run JavaScript.** It does not.

## Checklist

- [ ] I can name the two fields of a `Document` and why metadata matters
- [ ] I know every loader returns a list
- [ ] I can predict the Document count for a PDF, a CSV and a web page
- [ ] I can pick a PDF loader based on the document type
- [ ] I can explain when `lazy_load` is required rather than preferred
