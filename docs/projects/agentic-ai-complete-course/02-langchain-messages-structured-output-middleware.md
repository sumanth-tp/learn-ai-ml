---
id: agentic-course-langchain-messages-structured-output-middleware
title: "02. LangChain messages, structured output and middleware (Complete Agentic AI Course in 10 Hours)"
sidebar_label: "2 - LangChain: messages, structured output, middleware"
sidebar_position: 2
slug: /projects/agentic-ai-complete-course/langchain-messages-structured-output-middleware
description: "Learn how LangChain v1 messages carry a conversation, how to force a model to answer in a Pydantic, TypedDict or dataclass schema, and how middleware (summarisation and human approval) controls an agent."
tags: [agentic-ai, langchain, messages, structured-output, pydantic, middleware, summarization, human-in-the-loop]
---

import Infographic from '@site/src/components/Infographic';

> **Part 2 of 9** · [Watch on YouTube](https://www.youtube.com/watch?v=rV3HJ4LEZ7k) ·
> Notebooks: `updatedlangchain/4-messages.ipynb`, `updatedlangchain/5-structuredoutput.ipynb`,
> `updatedlangchain/6-middleware.ipynb` (the tab `langchain_middleware_examples.ipynb` is open in the editor but he never runs it).
> Notes follow the video in order.

By the end of this chapter you can build a conversation out of typed messages, make a model answer in a schema you define instead of free prose, and wrap an agent in middleware that summarises a long chat or pauses for a human to approve a risky tool call.

The chapter has three parts, exactly as the instructor teaches them:

1. **Messages**: the four message types and how a list of them becomes a conversation.
2. **Structured output**: getting objects back instead of text, with Pydantic, `TypedDict` and dataclasses.
3. **Middleware**: the airport security analogy, the hooks of an agent, summarisation middleware with three different triggers, and human-in-the-loop middleware with approve, edit and reject.

## Where we are

The previous part covered tools: how to bind a function to a chat model and let the model decide to call it. Everything so far has produced a *generative AI application* in the simplest sense, a prompt goes in and an answer comes out. This part adds the vocabulary and the control surfaces you need to build something sturdier.

He opens with the promise of what comes next: a new data structure called a *message* (system message, human message, AI message, tool message), then structured output, then middleware. The notebooks all live in the same `updatedlangchain` folder, one per topic, so each section below says which notebook it comes from.

### Setup used throughout this chapter

The environment is the same one built in the earlier parts of the course. The project is a `uv` project on Python 3.13 with a `.venv` kernel selected in the notebook, and these packages (from the repository's `pyproject.toml`, versions shown are the minimums it pins):

```bash
uv add "langchain>=1.1.0" "langchain-groq>=1.1.0" "langchain-openai>=1.1.0" "python-dotenv>=1.2.1" "ipykernel>=7.1.0"
```

API keys live in a `.env` file next to the notebooks and are read through environment variables. Never paste a real key into a notebook.

```text
GROQ_API_KEY=...
OPENAI_API_KEY=...
```

- `GROQ_API_KEY` powers the Qwen model on Groq that the messages and structured-output notebooks use for most examples.
- `OPENAI_API_KEY` powers the OpenAI models (`gpt-5`, `gpt-4o`, `gpt-4o-mini`) used for the agent examples and all of the middleware demos, because middleware belongs to `create_agent`.

## Messages

### What a message is

His definition, in plain words: a **message** is the basic unit of context in LangChain. Every time you talk to a chat model, what goes in and what comes out are messages, and together they describe the *state of the conversation*. A message is an object with three parts.

| Part | What it holds | Examples |
| --- | --- | --- |
| Role | Which kind of message this is. He stresses that the role is the important bit, because it tells the model how to treat the text. | system, human (user), AI, tool |
| Content | The actual payload. | Text, and also images, audio, documents |
| Metadata | Optional extras that travel with the message. | A name, an id, response information, token usage |

LangChain gives you one standard message type that behaves the same no matter which provider is behind the model, so the code you write against Groq also works against OpenAI.

He then points back at something you have already been seeing without the name: whenever you call `model.invoke(...)`, what you get back is an **AI message**, and whenever you give the model a question, it is treated as a **human message**. The rest of the section makes those two ideas explicit and adds the other two types.

<Infographic
  src="/img/agentic-course/02-msg-types.svg"
  alt="Board showing the three parts of a message (role, content, metadata), the four message types, and the two ways to call model.invoke"
  caption="Explanatory board (not shown in the video): what a message is, the four kinds, and text prompt versus message prompt."
/>

### Initialise the model

He starts from a fresh notebook, `4-messages.ipynb`. The first cell is the model setup you already know: import `init_chat_model`, copy the Groq key from the environment, and create the Qwen3 32B reasoning model. The string `"groq:qwen/qwen3-32b"` means *provider, colon, model name*.

```python
import os
from langchain.chat_models import init_chat_model

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

model = init_chat_model("groq:qwen/qwen3-32b")
```

Line by line:

- `os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")` makes sure the key is present in the environment where the Groq client looks for it. In this notebook the value is already in the environment (from the shell or a `.env` loaded earlier); if it is not set at all, `os.getenv` returns `None` and assigning that to `os.environ` raises a `TypeError`. If you rely on a `.env` file alone, call `load_dotenv()` first, as the middleware notebook does.
- `init_chat_model("groq:qwen/qwen3-32b")` returns a chat model object. Every later cell reuses the variable `model`.

### Text prompts

The simplest call is to hand the model a plain string.

```python
model.invoke("Please tell what is artificial intelligence")
```

He points at two facts. First, a bare string is quietly turned into a **human message** before it reaches the model. Second, what you get back is an **AI message**. Because this is a reasoning model, the `content` begins with a `<think>` block (the model's visible chain of thought) and only then the answer.

Output (trimmed; the real text is long):

```text
AIMessage(content='<think>\nOkay, the user is asking for a definition of artificial intelligence. ...
</think>\n\n... ### Summary:\nAI is a dynamic field reshaping industries and daily life. ...',
  response_metadata={'token_usage': {'completion_tokens': 1330, 'prompt_tokens': 14, 'total_tokens': 1344, ...},
                     'model_name': 'qwen/qwen3-32b', 'finish_reason': 'stop', 'model_provider': 'groq'},
  usage_metadata={'input_tokens': 14, 'output_tokens': 1330, 'total_tokens': 1344})
```

He then names this style a **text prompt**: a string, ideal for a straightforward generation task where you do not need to keep any conversation history.

```python
model.invoke("what is langchain")
```

Here he deliberately says nothing about *how* the model should behave. There is no persona and no rules, only a question. That is exactly the situation a text prompt is made for, and he lists when to choose it:

- you have a single, standalone request;
- you do not need conversation history;
- you want minimal code.

### Message prompts

The alternative is to pass **a list of message objects**. A list lets you describe a whole conversation, not just the latest question, and lets you say which role each line comes from. He first reads the four types off the notebook:

| Message | Meaning |
| --- | --- |
| System message | Instructions that tell the model how to behave and gives it context for the interaction. |
| Human message | What the user says, input and interaction with the model (text, images, audio, files). |
| AI message | What the model generated: text content, tool calls and metadata. |
| Tool message | The output of a tool call, handed back to the model. |

He expands on the system message because it is the one people find abstract: it is an *instruction to the LLM about how it should behave*. It primes the model before the user says anything, so it is where you set tone, define the model's role and set the guidelines for its replies.

Now the imports for the three types he uses first.

```python
from langchain.messages import SystemMessage, HumanMessage,AIMessage

messages=[
    SystemMessage("You are a poetry expert"),
    HumanMessage("Write a poem on artificial intelligence")
]

response=model.invoke(messages)
response.content
```

Walk through it:

- `from langchain.messages import SystemMessage, HumanMessage, AIMessage` brings in the classes. (The middleware notebook later imports the same classes from `langchain_core.messages`. Both paths work; `langchain.messages` is the short, v1-style one.)
- `messages = [...]` is a list in conversation order. The first item is the system message, "You are a poetry expert". The second is the human message asking for a poem on artificial intelligence. In a real chatbot this list grows turn by turn, which is why he says to think of it as a *conversation history*.
- `model.invoke(messages)` sends the whole list. `response.content` is the text of the reply.

He reads the start of the output aloud to prove both messages arrived: the model's reasoning begins with "the user wants a poem about artificial intelligence", showing it took the instruction and the request into account.

**Output**

```text
'<think>\nOkay, the user wants a poem about artificial intelligence. Let me start by thinking about the key themes related to AI. There\'s the creation aspect, how humans build AI. ...'
```

He adds that you could also put AI messages into this list. That is exactly what he does a little later.

### A second system message example

Next he swaps in a different one-line system prompt and a technical question.

```python
system_msg = SystemMessage("You are a helpful coding assistant.")

messages = [
    system_msg,
    HumanMessage("How do I create a REST API?")
]
response = model.invoke(messages)
print(response.content)
```

Here `system_msg` is built first and then dropped into the list, which is a tidy pattern when you want to reuse the same system message across calls. The answer is good but **generic**: the model lists Flask, Django, Node.js and more, because the system message only said "helpful coding assistant" and never named a language.

**Output**

```text
<think>
Okay, the user is asking how to create a REST API. Let me start by breaking down what they might need. ...
First, they need to understand the fundamentals. REST stands for Representational State Transfer ...
... I can list popular options: Python (Flask, Django), Node.js (Express), Ruby (Sinatra), Java (Spring Boot), etc.
```

### Detailed system messages

His point here: a one-liner works, but when you want a more specific answer you give the model **more context in the system message**. He keeps the user question identical, and changes only the system prompt.

```python
## Detailed info to the LLM through System message
from langchain.messages import SystemMessage, HumanMessage

system_msg = SystemMessage("""
You are a senior Python developer with expertise in web frameworks.
Always provide code examples and explain your reasoning.
Be concise but thorough in your explanations.
""")

messages = [
    system_msg,
    HumanMessage("How do I create a REST API?")
]
response = model.invoke(messages)
print(response.content)
```

The triple-quoted string defines a richer persona: a senior Python developer who knows web frameworks, always shows code, explains the reasoning, and stays concise but thorough. Same question, very different answer. The output now commits to Python and Flask, gives the install command, a worked code example, and practical advice such as disabling debug mode in production and adding input validation.

Take-away he states: the more precise information you put in the system message, the more precise the response.

### Role, content, metadata in practice

He now returns to the three-part definition and shows the **metadata** on a human message. `name` identifies which user is speaking and `id` gives the message a unique identifier, handy for tracing.

```python
## Message Metadata
human_msg = HumanMessage(
    content="Hello!",
    name="alice",  # Optional: identify different users
    id="msg_123",  # Optional: unique identifier for tracing
)
```

```python
response = model.invoke([
  human_msg
])
response
```

The first cell builds the message and does not print anything. The second sends it alone (a list with a single human message). The model replies in a friendly way, "Hello! How can I assist you today?", after reasoning that a greeting deserves a friendly answer. The returned `AIMessage` also carries `response_metadata` and `usage_metadata`:

**Output**

```text
AIMessage(content='<think>\nOkay, the user said "Hello!" so I should respond in a friendly way. ...\n</think>\n\nHello! How can I assist you today?',
  response_metadata={'token_usage': {'completion_tokens': 94, 'prompt_tokens': 10, 'total_tokens': 104, ...},
                     'model_name': 'qwen/qwen3-32b', ...},
  usage_metadata={'input_tokens': 10, 'output_tokens': 94, 'total_tokens': 104})
```

:::note Metadata is for you, not for the model
`name` and `id` are bookkeeping fields for your application, such as telling users apart in a multi-user log or tracing one message through a system. They do not change what the model is told to do. Whether a provider also uses `name` in the prompt depends on the provider.
:::

### Writing an AI message by hand

You can also *author* an AI message yourself. It does not have to come from the model. That is useful for rebuilding a conversation history, for example when you load a past chat from a database.

```python
from langchain.messages import AIMessage, SystemMessage, HumanMessage

# Create an AI message manually (e.g., for conversation history)
ai_msg = AIMessage("I'd be happy to help you with that question!")

# Add to conversation history
messages = [
    SystemMessage("You are a helpful assistant"),
    HumanMessage("Can you help me?"),
    ai_msg,  # Insert as if it came from the model
    HumanMessage("Great! What's 2+2?")
]

response = model.invoke(messages)
print(response.content)
```

Read the list like a script:

1. system: "You are a helpful assistant";
2. human: "Can you help me?";
3. AI: "I'd be happy to help you with that question!", typed by us, not generated, but labelled as an AI turn, "as if it came from the model";
4. human: "Great! What's 2+2?".

The model sees the whole exchange and answers the last question. Because Qwen3 is a reasoning model, the answer is again wrapped in a long think block ("2 plus 2 is 4, that's straightforward, but should I explain it?").

### Token usage on the response

Back on the real response, he looks at the metadata.

```python
response.usage_metadata
```

Output:

```text
{'input_tokens': 53, 'output_tokens': 258, 'total_tokens': 311}
```

It tells you how many tokens went in, how many came out and the total, which is how you keep an eye on cost.

:::note Name of the attribute
He says "response dot metadata" while explaining. There is no attribute with that exact name. The two attributes that exist are `response.usage_metadata` (the token counts shown above, the one he types) and `response.response_metadata` (provider details such as model name, finish reason and a provider-specific token usage block).
:::

### The tool message

The last of the four types links back to the tools part. When a model needs a tool, it does not run it. It produces an AI message that contains a **tool call**, your code runs the tool, and the result goes back to the model inside a **tool message**.

<Infographic
  src="/img/agentic-course/02-msg-tool-roundtrip.svg"
  alt="Board showing a list of three messages: human question, AI message with a get_weather tool call, and a tool message carrying the result, then fed to the model"
  caption="Explanatory board (not shown in the video): the get_weather example as an explicit list of messages."
/>

In the notebook he writes the whole exchange by hand so you can see every field.

```python
from langchain.messages import AIMessage
from langchain.messages import ToolMessage

# After a model makes a tool call
# (Here, we demonstrate manually creating the messages for brevity)
ai_message = AIMessage(
    content=[],
    tool_calls=[{
        "name": "get_weather",
        "args": {"location": "San Francisco"},
        "id": "call_123"
    }]
)

# Execute tool and create result message
weather_result = "Sunny, 72°F"
tool_message = ToolMessage(
    content=weather_result,
    tool_call_id="call_123"  # Must match the call ID
)

# Continue conversation
messages = [
    HumanMessage("What's the weather in San Francisco?"),
    ai_message,  # Model's tool call
    tool_message,  # Tool execution result
]
response = model.invoke(messages)  # Model processes the result
```

Walk through the cell:

- `AIMessage(content=[], tool_calls=[...])` is a model turn that says nothing in words but asks for a tool. The call has a `name` (`get_weather`), `args` (the location, San Francisco) and an `id` (`call_123`). He hard-codes it, which is why it is "for brevity".
- `weather_result = "Sunny, 72°F"` stands in for whatever your real function returned.
- `ToolMessage(content=weather_result, tool_call_id="call_123")` wraps that result. The comment "Must match the call ID" is the important part: the id is how the model knows *which* request this result answers, which matters when several tools are called in one turn.
- The list is the human question, then the AI tool call, then the tool result, in that order. `model.invoke(messages)` lets the model read the result and write the final reply.

Inspecting the tool message and the reply:

```python
tool_message
```

```python
response
```

**Output**

```text
ToolMessage(content='Sunny, 72°F', tool_call_id='call_123')
```

**Output**

```text
AIMessage(content='<think>\nOkay, the user asked for the weather in San Francisco. I used the get_weather function and got back "Sunny, 72°F". ...\n</think>\n\nThe current weather in San Francisco is **sunny** with a temperature of **72°F**. It looks like a pleasant day! 😊', ...)
```

He draws attention to two things: printing `tool_message` shows it really is a `ToolMessage`, and `model.invoke(messages)` gives back an AI message as always. The final cell in the notebook (cell 22) is empty.

He closes the section by saying he will keep covering the updated LangChain topics as they appear, and signposts the next one: structured output with Pydantic, nested structures and `TypedDict`.

## Structured output

### Why you need it

So far every answer has been a block of prose. That is fine for chat, but not for software. Suppose you ask a model for an essay or for details of a film and you want the program that receives the answer to read specific fields, such as a title or a year, without scraping paragraphs. You need the model to respond **in a format that matches a schema you define**.

The notebook's definition: models can be asked to answer in a given schema. That makes the output easy to parse and use in later steps. LangChain supports several schema types and ways of enforcing them. He covers three: **Pydantic**, **TypedDict** and **dataclasses**.

<Infographic
  src="/img/agentic-course/02-schema-routes.svg"
  alt="Board comparing Pydantic, TypedDict and dataclass schemas: runtime validation, what you get back, how to describe fields, nesting, when to pick each"
  caption="Explanatory board (not shown in the video): the three schema routes side by side."
/>

| | Pydantic `BaseModel` | `TypedDict` | `dataclass` |
| --- | --- | --- | --- |
| Checks types at runtime | Yes: a wrong type raises a validation error | No: it is a plain dictionary | No: the class does not enforce anything itself |
| What comes back | An instance, `Movie(title=..., year=...)` | A dict | An instance, `ContactInfo(name=..., ...)` |
| Field descriptions | `Field(description="...")` | `Annotated[str, ..., "description"]` | A docstring on the class (per-field comments are for humans, not sent to the model) |
| Nested structures | Yes (`cast: list[Actor]`) | Yes, without validation inside | Yes |
| He describes it as | "the richest feature set" | "a simpler alternative using built-in typing" | "a class that mostly holds data" |

### Pydantic

He begins with Pydantic, which gives you field validation, descriptions and nested structure. First he loads the same Qwen model in `5-structuredoutput.ipynb`.

On camera he first typed `from langchain.chat_models import init_chat_models` (with a stray `s`) and got an `ImportError`. He spotted it, noting that the function is called `init_chat_model`, and fixed it. The cell below is the corrected version.

```python
import os
from langchain.chat_models import init_chat_model
os.environ["GROQ_API_KEY"]=os.getenv("GROQ_API_KEY")
model=init_chat_model("groq:qwen/qwen3-32b")
model
```

The output is the model object, and it is worth a glance because it shows the model's **profile**, the table of capabilities LangChain knows about this model:

**Output**

```text
ChatGroq(profile={'max_input_tokens': 131072, 'max_output_tokens': 16384, 'image_inputs': False, 'audio_inputs': False,
  'video_inputs': False, 'image_outputs': False, 'audio_outputs': False, 'video_outputs': False,
  'reasoning_output': True, 'tool_calling': True}, ..., model_name='qwen/qwen3-32b', ...)
```

Now he defines the schema. He imports `BaseModel` and `Field` from `pydantic`, and writes a class `Movie`. `BaseModel` is the base class of every Pydantic model; hovering over it in the editor shows that it provides validation, descriptions and nested structure.

```python
from pydantic import BaseModel,Field

class Movie(BaseModel):
    title:str=Field(description="The title of the movie")
    year:int=Field(description="This year the movie was released")
    director:str=Field(description="The director of the movie")
    rating:float=Field(description="The movies rating out of 10")
```

What each line does:

- `class Movie(BaseModel)` declares the **shape of the answer** you want: a movie with four fields.
- `title: str` says the title must be a string. `year: int` says the year must be an integer. `director: str`, and `rating: float` (a float, because ratings such as 8.8 have decimals).
- `Field(description="...")` attaches a plain-English description to each field. This is not decoration: LangChain sends it to the model, so the model knows *which* piece of information belongs in *which* slot. He also points out that `Field` accepts many other parameters (you can see them by hovering), for example numeric limits such as `ge=0, le=10` to constrain the rating to the 0 to 10 range.
- Type hints plus Pydantic mean that if the model produced `"abc"` for `year`, you would get a validation error instead of silently wrong data.

Then he attaches the schema to the model.

```python
model_with_structure=model.with_structured_output(Movie)
model_with_structure
```

`model.with_structured_output(Movie)` returns a **new** runnable that wraps the same chat model plus instructions to produce `Movie`. The original `model` is unchanged, which is why he names the result `model_with_structure`.

Displaying it shows a `RunnableBinding` with two interesting things: the underlying `ChatGroq` model, and a `PydanticToolsParser`.

**Output**

```text
RunnableBinding(bound=ChatGroq(profile={...}, model_name='qwen/qwen3-32b', ...),
  kwargs={'tools': [{'type': 'function', 'function': {'name': 'Movie', 'description': '',
    'parameters': {'properties': {'title': {'description': 'The title of the movie', ...
  | PydanticToolsParser(first_tool_only=True, tools=[<class '__main__.Movie'>])
```

Reading it: for this Groq model, structured output is implemented by **tool calling**. The `Movie` schema is registered as a tool called `Movie` (with your field descriptions), the model is made to call it, and the parser turns the tool-call arguments into a real `Movie` object. You do not have to do any of that by hand.

### With and without a schema

To see the difference, he asks the same question twice. First against the plain model.

```python
model.invoke("Provide details about the moview Inception")
```

The result is a normal `AIMessage` full of prose: a reasoning block about Christopher Nolan, dreams within dreams, Dom Cobb, and the plot. Useful to a human, awkward for a program.

Now the same question against the structured model.

```python
response=model_with_structure.invoke("Provide details about the moview Inception")
response
```

**Output**

```text
Movie(title='Inception', year=2010, director='Christopher Nolan', rating=8.8)
```

This is a real Python object with exactly the four fields, ready to use anywhere in your code, `response.year`, `response.director`, and so on. He also notes that these details come from whatever the model absorbed during training, so trust the *shape* of the answer more than every detail in it.

### Why validation matters

He pauses on the part that makes Pydantic different. Because the schema says `title: str`, `year: int` and `rating: float`, **the values are checked**. A number where a string is required, or text where an integer is required, raises an error. This runtime *field validation* is the main reason to pick Pydantic when you will rely on the data afterwards.

:::note One nuance
Pydantic's default mode is forgiving about harmless conversions. A string such as `"2010"` is accepted and converted to the integer 2010, but a value that cannot be converted (for instance `"abc"`) is rejected. If you need strict behaviour, Pydantic has a strict mode.
:::

### Message output alongside the parsed structure

Sometimes you want the parsed object **and** the raw model message (to read its token usage or its reasoning). The option for that is `include_raw=True`.

```python
from pydantic import BaseModel, Field

class Movie(BaseModel):
    """A movie with details."""
    title: str = Field(..., description="The title of the movie")
    year: int = Field(..., description="The year the movie was released")
    director: str = Field(..., description="The director of the movie")
    rating: float = Field(..., description="The movie's rating out of 10")

model_with_structure = model.with_structured_output(Movie, include_raw=True)  

response = model_with_structure.invoke("Provide details about the movie Inception")
response
```

Notice three things. The class now has a docstring, `"""A movie with details."""`, which becomes the description of the whole schema. The fields use `Field(..., description=...)`. And the call passes `include_raw=True`.

The result is a dictionary with three keys:

**Output**

```text
{'raw': AIMessage(content='', additional_kwargs={'reasoning_content': "Okay, the user is asking for details about the movie Inception. Let me check the tools available. There's a Movie function ...",
                  'tool_calls': [...]},
                  tool_calls=[{'name': 'Movie', 'args': {'director': 'Christopher Nolan', 'rating': 8.8, 'title': 'Inception', 'year': 2010}, 'id': 'r0n9zde78', 'type': 'tool_call'}],
                  usage_metadata={'input_tokens': 231, 'output_tokens': 170, 'total_tokens': 401, 'output_token_details': {'reasoning': 122}}),
 'parsed': Movie(title='Inception', year=2010, director='Christopher Nolan', rating=8.8),
 'parsing_error': None}
```

- `raw` is the untouched AI message. Its `tool_calls` entry proves that the model really did call the `Movie` "tool", and its metadata shows the tokens used.
- `parsed` is the clean `Movie` object, the same as before.
- `parsing_error` is `None` when everything went well. If the model's output could not be parsed into `Movie`, this slot would carry the error instead of your program crashing, so you can handle it.

:::note A slip on camera
While reading the cell he calls the `Field(...)` fields "optional". It is the other way round: the `...` (Ellipsis) means the field is **required**. To make a field optional you give it a default, as the `budget` field below does with `= Field(None, ...)`.
:::

### Nested structures

Real data nests. A movie has actors, and each actor has a name and a role. Pydantic models can contain other Pydantic models.

<Infographic
  src="/img/agentic-course/02-nested-schema.svg"
  alt="Board showing class MovieDetails containing a list of Actor objects, and the MovieDetails result returned for Inception"
  caption="Explanatory board (not shown in the video): the nested Actor inside MovieDetails, with the result he got back."
/>

```python
from pydantic import BaseModel, Field

class Actor(BaseModel):
    name: str
    role: str

class MovieDetails(BaseModel):
    title: str
    year: int
    cast: list[Actor]
    genres: list[str]
    budget: float | None = Field(None, description="Budget in millions USD")

model_with_structure = model.with_structured_output(MovieDetails)

response = model_with_structure.invoke("Provide details about the movie Inception")
response
```

Walk through it:

- `Actor` has two string fields, `name` and `role`.
- `MovieDetails` uses it: `cast: list[Actor]` means *a list of Actor objects*. `genres: list[str]` is a plain list of strings.
- `budget: float | None = Field(None, description="Budget in millions USD")` is an **optional** field: it may be a float or `None`, defaults to `None`, and the description tells the model what unit to use.
- Everything else is as before: `with_structured_output(MovieDetails)`, then `invoke`.

The model fills in the whole tree.

**Output**

```text
MovieDetails(title='Inception', year=2010,
  cast=[Actor(name='Leonardo DiCaprio', role='Dom Cobb'), Actor(name='Joseph Gordon-Levitt', role='Arthur'),
        Actor(name='Elliot Page', role='Ariadne'), Actor(name='Tom Hardy', role='Bane')],
  genres=['Science Fiction', 'Action', 'Heist'], budget=160.0)
```

He points out the list of actors in `cast`, the list of genres, and the budget of 160.0, which is 160 million dollars because the description said "millions USD". The summary he gives: field validation plus a schema you design means the model produces exactly the shape your code expects.

:::warning Check the facts
The model's output here is not fully accurate. Tom Hardy played Eames in *Inception*, not "Bane" (that is a different Nolan film). A schema guarantees the **shape** of the answer, not its truth.
:::

### TypedDict

The second route is `TypedDict`, from Python's typing tools. It is a simpler alternative for when you do **not** need runtime validation: a `TypedDict` is, at runtime, just a plain dictionary, with type hints for humans and tools.

```python
from typing_extensions import TypedDict,Annotated

class MovieDict(TypedDict):
    """A movie with details."""
    title: Annotated[str, ..., "The title of the movie"]
    year: Annotated[int, ..., "The year the movie was released"]
    director: Annotated[str, ..., "The director of the movie"]
    rating: Annotated[float, ..., "The movie's rating out of 10"]


model_withtypedict=model.with_structured_output(MovieDict)
response=model_withtypedict.invoke("Please provide the details of the movie avengers")
response
```

What each part does:

- `class MovieDict(TypedDict)` plays the role that `Movie(BaseModel)` played, but describes a dict.
- `Annotated[str, ..., "The title of the movie"]` is how you attach a description: the type first, then `...`, then the description string.
- The docstring describes the whole schema.
- The rest is identical: `with_structured_output(MovieDict)` and an `invoke`, this time for *Avengers*.

**Output**

```text
{'director': 'Joss Whedon', 'rating': 8, 'title': 'The Avengers', 'year': 2012}
```

The answer is a plain dictionary. Look at `'rating': 8`: the schema said `float` and the model returned an integer, and nothing complained. That is the "no runtime validation" point made visible. He is relaxed about it: with a `TypedDict` the schema guides the model but nothing checks the answer, so a stray integer where a float was declared is acceptable.

He then does the nested version with `TypedDict` for *Avengers*.

```python
class Actor(TypedDict):
    name: str
    role: str

class MovieDetails(TypedDict):
    title: str
    year: int
    cast: list[Actor]
    genres: list[str]
    budget: float | None = Field(None, description="Budget in millions USD")

model_with_structure = model.with_structured_output(MovieDetails)

response = model_with_structure.invoke("Provide details about the movie Avengers")
response
```

**Output**

```text
{'budget': 220000000,
 'cast': [{'name': 'Robert Downey Jr.', 'role': 'Iron Man'}, {'name': 'Chris Evans', 'role': 'Captain America'},
          {'name': 'Mark Ruffalo', 'role': 'Hulk'}, {'name': 'Chris Hemsworth', 'role': 'Thor'},
          {'name': 'Scarlett Johansson', 'role': 'Black Widow'}, {'name': 'Jeremy Renner', 'role': 'Hawkeye'}],
 'genres': ['Action', 'Science Fiction', 'Adventure'],
 'title': 'The Avengers',
 'year': 2012}
```

He explains that nesting works the same way as with Pydantic, but nothing validates the values, for example a string where a number belongs would also pass.

:::warning Source correction: `Field` does not belong in a TypedDict
In the last line of the `MovieDetails` class, the notebook writes `budget: float | None = Field(None, description="Budget in millions USD")`. That line is copied from the Pydantic version and does not do what it appears to do in a `TypedDict`: a `TypedDict` cannot have defaults, and `Field` is a Pydantic feature, so the description never reaches the model. You can see the effect in the output: `budget` came back as `220000000` (dollars), whereas the Pydantic version, which carried the description, returned `160.0` (millions).

The correct way to describe and optionalise a field in a `TypedDict` is `budget: Annotated[float | None, ..., "Budget in millions USD"]`.
:::

### Model profile

Before leaving the Groq model, he shows a handy property. If you ask the *structured* wrapper for its profile, you get an `AttributeError`, because the wrapper is a runnable, not the model. Ask the **original** model.

```python
model.profile
```

**Output**

```text
{'max_input_tokens': 131072,
 'max_output_tokens': 16384,
 'image_inputs': False,
 'audio_inputs': False,
 'video_inputs': False,
 'image_outputs': False,
 'audio_outputs': False,
 'video_outputs': False,
 'reasoning_output': True,
 'tool_calling': True}
```

Read it as a capability sheet for Qwen3 32B: it takes about 131 thousand input tokens, can produce up to about 16 thousand, does **not** accept images, audio or video, **does** produce reasoning output and **does** support tool calling. These are the facts you check when choosing a model for a task.

### Dataclasses, and structured output inside agents

The third route is the **dataclass**. Dataclasses have been in Python since version 3.7. A dataclass is a class that mostly stores data and is created with the `@dataclass` decorator. By itself it has no input validation.

For this one he changes the setting. Rather than `with_structured_output`, he shows how to attach a schema to a whole **agent** with `create_agent`, using GPT-5. That means loading the OpenAI key.

```python
import os
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
```

He first shows the Pydantic version in an agent.

```python
from pydantic import BaseModel, Field
from langchain.agents import create_agent


class ContactInfo(BaseModel):
    """Contact information for a person."""
    name: str = Field(description="The name of the person")
    email: str = Field(description="The email address of the person")
    phone: str = Field(description="The phone number of the person")

agent = create_agent(
    model="gpt-5",
    response_format=ContactInfo  # Auto-selects ProviderStrategy
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result
```

What is new here:

- `from langchain.agents import create_agent` brings in the agent builder from the earlier parts.
- `ContactInfo` is a Pydantic model with three string fields (name, email, phone), each with a description. Inheriting `BaseModel` means every field is validated.
- `create_agent(model="gpt-5", response_format=ContactInfo)` is the key line. Instead of calling `with_structured_output` yourself, you give the agent a `response_format`, and its final answer is produced in that shape. The comment says it "auto-selects ProviderStrategy": for a model whose provider can enforce a JSON schema natively (OpenAI can), LangChain uses that; for others it falls back to a tool-calling strategy.
- The input is a dictionary with a `messages` list, each message a dict with `role` and `content`. The text asks the agent to extract contact info from "John Doe, john@example.com, (555) 123-4567".

The returned `result` is a dictionary. He first prints the whole thing: it contains the human message, the AI message (whose content is the JSON text) and, crucially, a separate `structured_response` key.

**Output**

```text
{'messages': [HumanMessage(content='Extract contact info from: John Doe, john@example.com, (555) 123-4567', ...),
              AIMessage(content='{"name":"John Doe","email":"john@example.com","phone":"(555) 123-4567"}', ...)],
 'structured_response': ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')}
```

Then he reads just the field he wants.

```python
result["structured_response"]
```

**Output**

```text
ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```

Validation applied to every field, which is the useful property of Pydantic.

Next the `TypedDict` version, which he introduces to make the comparison.

```python
## Typedict
from typing_extensions import TypedDict
from langchain.agents import create_agent


class ContactInfo(TypedDict):
    """Contact information for a person."""
    name: str # The name of the person
    email: str # The email address of the person
    phone: str # The phone number of the person

agent = create_agent(
    model="gpt-5",
    response_format=ContactInfo  # Auto-selects ProviderStrategy
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]
# {'name': 'John Doe', 'email': 'john@example.com', 'phone': '(555) 123-4567'}
```

The structure is identical, but the schema is a `TypedDict` (note the plain `# comment` after each field, which is only for readers). He removes the tools to keep the example small. The result is a dictionary rather than an object.

**Output**

```text
{'name': 'John Doe', 'email': 'john@example.com', 'phone': '(555) 123-4567'}
```

Finally the dataclass version.

```python
## Dataclass

from dataclasses import dataclass
from langchain.agents import create_agent

@dataclass
class ContactInfo:
    """Contact information for a person."""
    name: str # The name of the person
    email: str # The email address of the person
    phone: str # The phone number of the person


agent = create_agent(
    model="gpt-5",
    response_format=ContactInfo  # Auto-selects ProviderStrategy
)

result = agent.invoke({
    "messages": [{"role": "user", "content": "Extract contact info from: John Doe, john@example.com, (555) 123-4567"}]
})

result["structured_response"]
```

Here `@dataclass` turns `ContactInfo` into a data holder with `name`, `email` and `phone` strings. Again `create_agent(..., response_format=ContactInfo)` and `result["structured_response"]`, and the answer comes back as a `ContactInfo` instance.

**Output**

```text
ContactInfo(name='John Doe', email='john@example.com', phone='(555) 123-4567')
```

:::note Docstrings, not comments
In the `TypedDict` and dataclass versions the `# The name of the person` comments help readers but are not sent to the model. What the model sees is the class **docstring** (`"""Contact information for a person."""`) and the field names and types. If you want per-field descriptions to reach the model, use `Field(description=...)` in Pydantic or `Annotated[..., "description"]` in a `TypedDict`.
:::

His summary of the section: you now know how to get structured output from a model with Pydantic, `TypedDict` and dataclasses. These are just different ways of doing the same job and you can pick whichever suits you. The next topics he mentions are streaming and short-term memory, then he turns to middleware.

## Middleware

### What middleware is for

Middleware, in the notebook's definition, is a way to **more tightly control what happens inside an agent**. It is useful for:

- tracking agent behaviour with logging, analytics and debugging;
- transforming prompts, tool selection and output formatting;
- adding retries, fallbacks and early-termination logic;
- applying rate limits, guardrails and PII (personally identifiable information) detection.

He admits that the definition alone leaves most people confused, so he explains it with an analogy.

### The airport security analogy

> Instructor's analogy: think of an airport.

You are the **passenger**. Your goal is the **flight at gate 18**. But you cannot walk straight to the gate. You must pass a series of checkpoints first: **security check**, then **immigration**, then **boarding**, and only then do you reach the flight.

Each checkpoint does its own inspection:

- At security, officers look inside your luggage, for example making sure you are not carrying batteries. He labels that **middleware 1**: it performs the checks that are needed on your luggage and belongings.
- At immigration, officers check your passport, including whether it is still valid. That is **middleware 2**.
- Before boarding, staff check your boarding pass to see that it is correct for this flight. That is **middleware 3**.

<Infographic
  src="/img/agentic-course/02-airport-security.svg"
  alt="Board of the airport security analogy: passenger goes through security check, immigration and boarding to flight 18, each with a numbered middleware, and the same idea applied around an agent"
  caption="Redrawn from the whiteboard (the lower strip, middleware around an agent, is the mapping he describes aloud)."
/>

Now map it to software. Replace the passenger with a **request**, and the flight with the **agent**. Before the request reaches the agent, it passes through checkpoints, and each can do something: a plain check, logging, exception handling, a model call, anything. That is why he says middleware lets you control *what happens inside the agent* so tightly. You can create middleware 1, middleware 2, middleware 3 and put any logic you need in each.

### An agent with and without middleware

Next he recalls what an agent is. It contains a **model** and **tools**, which is the ReAct pattern: the request goes to the model, the model decides whether a tool is needed, the tool runs and gives context back, and eventually you get the result.

With middleware, the same agent looks different. He shows the diagram from the LangChain docs: on the left the plain loop, on the right the same loop with **hooks** added.

<Infographic
  src="/img/agentic-course/02-agent-hooks.svg"
  alt="Two diagrams side by side. Left: request, model, tools loop, result. Right: the same agent with before_agent, before_model, wrap_model_call, wrap_tool_call, after_model and after_agent hooks"
  caption="Redrawn from the LangChain docs diagram that he annotates with the word 'hooks'."
/>

A **hook** is a trigger point: a moment in the agent's run at which your middleware gets called. The diagram marks them:

| Hook | When it runs |
| --- | --- |
| `before_agent` | Once, before the agent starts working on the request |
| `before_model` | Before each call to the model |
| `wrap_model_call` | Around the model call itself, so you can change or retry it |
| `wrap_tool_call` | Around each tool call |
| `after_model` | After each model response |
| `after_agent` | Once, when the agent has finished, before the result is returned |

His list of what you might do at a hook: logging, summarisation, and many other things.

### Built-in middleware

LangChain ships ready-made middleware for common jobs. He starts with the most common one, **summarisation middleware**, and sketches it on the board.

<Infographic
  src="/img/agentic-course/02-builtin-summarization.svg"
  alt="Board listing built-in middleware (summarization, human in the loop, model call limit) next to a diagram of an agent whose message list is summarised by an LLM when it reaches ten messages"
  caption="Redrawn from the whiteboard."
/>

Imagine an agent connected to a tool, with an input on one side and an output on the other. Each turn adds messages to the conversation list, so the list keeps growing. Summarisation middleware watches that list. Once it reaches a size you choose, say **10 messages**, it asks an LLM to summarise the whole stack and replaces the old messages with the summary. The agent then carries on with a short context instead of an ever-growing one.

Two more entries on his list: **human in the loop** (a human gives feedback or approves before an action) and **model call limit** (limit how many times the model is called, to prevent excessive cost). Then he opens the docs page "Built-in middleware" to show the full menu.

| Middleware | What it does (as listed on the docs page he shows) |
| --- | --- |
| Summarization | Automatically summarises the conversation history when it approaches token limits |
| Human-in-the-loop | Pauses execution so a human can approve tool calls |
| Model call limit | Limits the number of model calls to prevent excessive cost |
| Tool call limit | Controls tool execution by limiting call counts |
| Model fallback | Automatically falls back to alternative models when the primary fails |
| PII detection | Detects and handles personally identifiable information |
| To-do list | Equips agents with task planning and tracking capabilities |
| LLM tool selector | Uses an LLM to select relevant tools before calling the main model |
| Tool retry | Automatically retries failed tool calls with exponential backoff |
| Model retry | Automatically retries failed model calls with exponential backoff |
| LLM tool emulator | Emulates tool execution using an LLM, for testing purposes |
| Context editing | Manages conversation context by trimming or clearing tool uses |
| Shell tool | Exposes a persistent shell session to agents for command execution |
| File search | Provides Glob and Grep search tools over filesystem files |

(He names summarisation, human in the loop, model call limit, tool call limit, model fallback, to-do list, LLM tool selector and tool retry aloud. The rest of the rows are read from the page on screen.)

His plan: cover the most important ones with working examples so that you can apply any of the others independently, since in the end which one you use depends on your use case.

### Notebook setup

He switches to `6-middleware.ipynb`, restates the definition bullets in the first markdown cell, and loads the environment with the OpenAI key. Middleware belongs to `create_agent`, and these demos run on OpenAI models.

```python
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
```

`load_dotenv()` reads the `.env` file, and the last line copies the key into the environment. (The same setup is why the earlier notebooks could use `os.getenv`.)

### Summarisation middleware

The notebook's definition: summarisation middleware **automatically summarises conversation history when approaching token limits, preserving recent messages while compressing older context**. It is useful for:

- long-running conversations that would exceed the context window, especially chatbots;
- multi-turn dialogues with extensive history;
- applications where keeping the meaning of the whole conversation matters.

There are several kinds of **trigger** for when summarising should start: a number of **messages**, a number of **tokens**, and a **fraction** of the model's context window. He demonstrates all three, starting with messages.

#### Trigger on message count

First the imports and the agent.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import HumanMessage, SystemMessage

### Messagebased summarization
agent=create_agent(
    model="gpt-4o-mini",
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("messages",10),
            keep=("messages",4)
        )
    ]
)
```

Line by line:

- `create_agent` builds the agent; `SummarizationMiddleware` is the built-in summariser.
- `InMemorySaver` is a **checkpointer**: it stores the conversation state so each turn can continue from the last. Without one, an agent forgets everything between calls. (He imports `HumanMessage` and `SystemMessage` here too, though only `HumanMessage` is used.)
- `model="gpt-4o-mini"` is the model the agent itself uses. He defines no tools, so there is only a model.
- `checkpointer=InMemorySaver()` plugs in the memory.
- `middleware=[...]` takes a **list**, so you can stack any number of middleware, separated by commas.
- Inside `SummarizationMiddleware(...)`:
  - `model="gpt-4o-mini"` is the model that writes the summaries. He recommends a **cheap model** here, because every time the conversation grows past the trigger you pay for another summarisation call.
  - `trigger=("messages", 10)` says *start summarising when the thread reaches 10 messages*. Inputs and outputs both count. A real chatbot would use a much bigger number; he uses 10 so that you can watch it happen.
  - `keep=("messages", 4)` says *when you summarise, compress everything older and keep the 4 most recent messages untouched*, so the model still sees the latest exchange word for word.

:::note About the checkpointer
He describes the checkpointer as saving the conversation "on the hard disk". `InMemorySaver` keeps the state in the program's **memory (RAM)**, so it disappears when the process stops. For persistence across restarts you need a database-backed checkpointer such as the SQLite or Postgres savers from LangGraph.
:::

Next, the thread. A thread id identifies one conversation, so the checkpointer knows which stored history to continue.

```python
### Run with thread id
config={"configurable":{"thread_id":"test-1"}}
```

He calls this "a unique user". Strictly, a thread is a unique *conversation*; one user can have many.

Then the test. He feeds in a series of arithmetic questions, each a new human message on the same thread, and prints the **number of messages** in the thread after each turn.

```python
# Alternative test data
questions = [
    "What is 2+2?",
    "What is 10*5?",
    "What is 100/4?",
    "What is 15-7?",
    "What is 3*3?",
    "What is 4*4?",
]

for q in questions:
    response=agent.invoke({"messages":[HumanMessage(content=q)]},config)
    print(f"Messages: {response}")
    print(f"Messages: {len(response['messages'])}")
```

(He talks about four questions, two plus two, ten times five, a hundred divided by four, fifteen minus seven. The notebook's comment says "Alternative test data" and has six, adding `3*3` and `4*4`.)

- `agent.invoke({"messages": [HumanMessage(content=q)]}, config)` sends one question. The `config` carries the thread id, so each question is appended to the **same** stored conversation.
- `len(response['messages'])` is the size of the thread right now.

Reading the printed counts:

**Output**

```text
Messages: 2
Messages: 4
Messages: 6
Messages: 8
Messages: 10
Messages: 6
```

Each turn adds two messages (the human question and the AI answer), so you see 2, 4, 6, 8, 10. On the sixth question the thread would have passed 10 messages, so the middleware steps in just before the model is called: it replaces the older messages with one summary message, keeps the latest 4, and then the new AI answer is added. The count drops to 6. The first message of the new thread begins like this:

**Output**

```text
Here is a summary of the conversation to date:

Human asked several arithmetic questions:

1. What is 2 + 2?
   - AI responded: "2 + 2 equals 4."

2. What is 10 * 5?
   - AI responded: "10 * 5 equals 50."
...
```

This is the key property of middleware, he says: you attach a rule to the agent and it applies itself, without you writing summarisation logic in your own loop.

<Infographic
  src="/img/agentic-course/02-summarization-runs.svg"
  alt="Three small bar charts showing the number of messages after each turn for the message-count, token and fraction triggers, with the turn where summarisation happened highlighted"
  caption="Explanatory board (not shown in the video): message counts after each turn for the three runs in his notebook."
/>

#### Trigger on token count

The second trigger uses **tokens**. For this run he adds a tool, because tool results are big and fill the context quickly.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

@tool
def search_hotels(city: str) -> str:
    """Search hotels - returns long response to use more tokens."""
    return f"""Hotels in {city}:
    1. Grand Hotel - 5 star, $350/night, spa, pool, gym
    2. City Inn - 4 star, $180/night, business center
    3. Budget Stay - 3 star, $75/night, free wifi"""


agent=create_agent(
    model="gpt-4o-mini",
    tools=[search_hotels],
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("tokens",550),
            keep=("tokens",200),
        ),
    ]
)

config = {"configurable": {"thread_id": "test-1"}}

# Token counter (approximate)
def count_tokens(messages):
    total_chars = sum(len(str(m.content)) for m in messages)
    return total_chars // 4  # 4 chars ≈ 1 token
```

What is new:

- `@tool` turns a function into a tool. `search_hotels(city)` returns a long hard-coded string, three hotels with stars, prices and facilities. He says to imagine it came from a real API. The docstring says it returns a long response "to use more tokens", which is the whole point.
- `tools=[search_hotels]` gives the agent that tool.
- `trigger=("tokens", 550)` means *summarise when the thread passes 550 tokens*, and `keep=("tokens", 200)` means *keep roughly the most recent 200 tokens*.
- `count_tokens` is his own helper for printing. It adds up the characters in every message and divides by 4, using the rule of thumb that **four characters make about one token**. It is only an estimate for display and is not the count the middleware uses.

On camera, the cell failed first. He had typed the model name as `gtp-4o-mini`, and the library answered that it could not find the model and asked him to specify it directly. Correcting the spelling to `gpt-4o-mini` (the cell above is the fixed version) made it run.

Now the run: one question per city, on the same thread.

```python
# Run test
cities = ["Paris", "London", "Tokyo", "New York", "Dubai", "Singapore"]

for city in cities:
    response = agent.invoke(
        {"messages": [HumanMessage(content=f"Find hotels in {city}")]},
        config=config
    )
    
    tokens = count_tokens(response["messages"])
    print(f"{city}: ~{tokens} tokens, {len(response['messages'])} messages")
    print(f"{(response['messages'])}")
```

Each city produces **four messages**: the human question, the AI message that calls the tool, the tool message with the hotels, and the AI's final answer. Reading the first line of each result:

**Output**

```text
Paris: ~149 tokens, 4 messages
London: ~302 tokens, 8 messages
Tokyo: ~456 tokens, 12 messages
New York: ~396 tokens, 8 messages
Dubai: ~232 tokens, 5 messages
Singapore: ~361 tokens, 9 messages
```

The size climbs, 149, 302, 456, and then **falls** at New York to 396 tokens and 8 messages. That is the summarisation: the middleware's own count went past 550 on the way into that turn, so it compressed the old turns and kept about the last 200 tokens. It happens again at Dubai (232 tokens, 5 messages). Printing the messages shows human messages starting "Here is a summary of the conversation to date: User requested hotel information for Paris and received details on three hotels, including the Grand Hotel, City Inn, and Budget Stay ...". The final answers still mention the Grand Hotel and the rest because the summary carries them.

:::note Why the printed numbers do not cross 550
The helper counts only message text and rounds crudely, while the middleware counts every message including the tool-call arguments and per-message overhead. So the threshold fires while his printed estimate is still a little below 550. Treat his figure as a rough indicator.
:::

#### Trigger on a fraction of the context window

The third option expresses the trigger as a **fraction of the model's context window**. That is handy because the same configuration then adapts to whichever model you use. He pastes this cell.

```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

@tool
def search_hotels(city: str) -> str:
    """Search hotels."""
    return f"Hotels in {city}: Grand Hotel $350, City Inn $180, Budget Stay $75"

# LOW fraction for testing!
agent = create_agent(
    model="gpt-4o-mini",
    tools=[search_hotels],
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model="gpt-4o-mini",
            trigger=("fraction", 0.005),  # 0.5% = ~640 tokens
            keep=("fraction", 0.002),     # 0.2% = ~256 tokens
        ),
    ],
)

config = {"configurable": {"thread_id": "test-1"}}

# Token counter
def count_tokens(messages):
    return sum(len(str(m.content)) for m in messages) // 4

# Test
cities = ["Paris", "London", "Tokyo", "New York", "Dubai", "Singapore"]

for city in cities:
    response = agent.invoke(
        {"messages": [HumanMessage(content=f"Hotels in {city}")]},
        config=config
    )
    tokens = count_tokens(response["messages"])
    fraction = tokens / 128000  # gpt-4o-mini context
    print(f"{city}: ~{tokens} tokens ({fraction:.4%}), {len(response['messages'])} msgs")
    print(response['messages'])
```

Points to read:

- `trigger=("fraction", 0.005)` means *when the conversation uses 0.5 per cent of the context window* (0.005 is 0.5%, not 0.5; in the same way 0.002 below is 0.2%, not the "2%" he says aloud). The comment says that is about 640 tokens.
- `keep=("fraction", 0.002)` means *keep the most recent 0.2 per cent*, about 256 tokens. These values are deliberately tiny so the demo triggers fast; a real application would use something like 0.7 or 0.8.
- The helper divides the estimated tokens by `128000`, the context size of `gpt-4o-mini`, to print each turn as a percentage.

:::note Context size
While explaining he says the model has "160k" tokens. In the notebook the fraction is divided by 128000, which is the context window of `gpt-4o-mini`: 0.5% of 128,000 is 640, which matches the comment. Fraction triggers need the library to know the model's maximum input size, which it reads from the model profile you saw earlier.
:::

Output (first line of each turn):

```text
Paris: ~64 tokens (0.0500%), 4 msgs
London: ~133 tokens (0.1039%), 8 msgs
Tokyo: ~203 tokens (0.1586%), 12 msgs
New York: ~276 tokens (0.2156%), 16 msgs
Dubai: ~349 tokens (0.2727%), 20 msgs
Singapore: ~365 tokens (0.2852%), 12 msgs
```

The conversation grows steadily to 20 messages, then at Singapore it collapses to 12: the fraction trigger fired and a summary appeared at the head of the thread ("Here is a summary of the conversation to date: 1. User requested information about hotels in multiple cities: Paris, London, Tokyo, New York ..."). He did not need to pick a model-specific token number.

He recaps the three: by **message count**, by **token size**, and by **fraction** of the context. Then he points at the docs page for more examples.

:::tip API names across versions
The trigger and keep tuples shown here are the current API. Older LangChain 1.0 pre-release posts used separate arguments called `max_tokens_before_summary` and `messages_to_keep`. If you meet them in an older tutorial, they map onto `trigger` and `keep`. The middleware also accepts a list of triggers if you want, for example, "messages or tokens, whichever comes first"; check the docs page for your version.
:::

#### Other built-ins

He ends the section by mentioning two other built-ins he will not demonstrate. **Tool call limit** is applied the same way, by putting it in the `middleware` list. **Model fallback** switches to another model when the first fails, for instance when the API key stops working or the provider is down. Human in the loop is next.

## Human-in-the-loop middleware

### What it is and why

Notebook definition: human-in-the-loop (HITL) middleware **pauses agent execution so a human can approve, edit or reject a tool call before it runs**. It suits:

- high-stakes operations that need human approval, such as database writes and financial transactions;
- compliance workflows where human oversight is mandatory;
- long-running conversations where human feedback guides the agent.

He opens his scribble page to explain why. An agent has an input and an output. When it works *autonomously*, it does its task with no human involvement. That is fine for harmless jobs, but suppose the task is a **financial transaction**, for example an agent that buys stocks. This is a **critical task**: a single mistake by the model, such as buying the wrong stock the next day, could cost a lot of money.

<Infographic
  src="/img/agentic-course/02-hitl-scribble.svg"
  alt="Board with an autonomous agent taking input and producing output, a human intervention arrow into the agent, and a side chain from financial transaction to stock buy to critical task"
  caption="Redrawn from the whiteboard."
/>

So you cannot depend completely on the autonomous agent. Instead you put a **human** into the loop. Whenever the agent decides on a critical action, it first requests confirmation from the human, and the task does not complete until the human gives feedback. That is the origin of the name *human in the loop*. For any critical task, he says, human intervention is needed, because LLMs make mistakes.

### Build the agent

Continuing in the same notebook, the imports:

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver

def read_email_tool(email_id: str) -> str:
    """Mock function to read an email by its ID."""
    return f"Email content for ID: {email_id}"

def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Mock function to send an email."""
    return f"Email sent to {recipient} with subject '{subject}'"
```

The cell has two parts. The imports bring in `HumanInTheLoopMiddleware` alongside `create_agent` and the in-memory checkpointer. Then two **mock tools** for an email assistant:

- `read_email_tool(email_id)` pretends to read an email by id and returns a string.
- `send_email_tool(recipient, subject, body)` pretends to send an email and returns a confirmation string.

:::note Mock tools
These are dummy functions that only return text. To send real email you would use an SMTP server or a mail API, but the lesson is the approval mechanism, not email. Note also that plain functions with docstrings and type hints are accepted as tools by `create_agent`; they do not have to carry `@tool`.
:::

Now the agent.

```python
agent=create_agent(
    model="gpt-4o",
    tools=[read_email_tool,send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool":{
                    "allowed_decisions":["approve","edit","reject"]
                },
                "read_email_tool":False,

            }
        )
    ]
)
```

- `model="gpt-4o"` and `tools=[read_email_tool, send_email_tool]`: the agent can read and send.
- `checkpointer=InMemorySaver()` is **required** for human in the loop. The agent must be able to stop in the middle of a run and later continue from the exact same place, and the checkpointer is where it remembers where it stopped.
- `HumanInTheLoopMiddleware(interrupt_on={...})` declares *which tools need a human*.
  - `"send_email_tool": {"allowed_decisions": ["approve", "edit", "reject"]}`: whenever the agent tries to call this tool, pause and let the human choose one of three decisions. **Approve** lets the call go ahead exactly as it is. **Edit** lets the human change the arguments first, for example fix a mistyped recipient. **Reject** stops the call.
  - `"read_email_tool": False`: no pause. Reading mail is harmless, so it runs automatically.

<Infographic
  src="/img/agentic-course/02-hitl-flow.svg"
  alt="Flow of a human-in-the-loop run: invoke, tool call, interrupt, human decision approve or edit or reject, resume with Command"
  caption="Explanatory board (not shown in the video): the pause and resume cycle the next three examples follow."
/>

### Approve

First the request. A thread id of `test-approve` identifies this run.

```python
config = {"configurable": {"thread_id": "test-approve"}}
# Step 1: Request
result = agent.invoke(
    {"messages": [HumanMessage(content="Send email to john@test.com with subject 'Hello' and body 'How are you?'")]},
    config=config
)
```

The prompt tells the agent to send an email to `john@test.com` with a given subject and body. The agent now reasons: it has two tools, and sending is the one that matches. Because `send_email_tool` is on the interrupt list, the run **stops before the tool executes**. Nothing is sent yet. He prints `result` to look:

```python
result
```

The dictionary has the usual `messages` (the human message, then an AI message whose `tool_calls` asks for `send_email_tool` with the recipient, subject and body), plus a new key, `__interrupt__`:

**Output**

```text
'__interrupt__': [Interrupt(value={
    'action_requests': [{'name': 'send_email_tool',
                         'args': {'recipient': 'john@test.com', 'subject': 'Hello', 'body': 'How are you?'},
                         'description': "Tool execution requires approval\n\nTool: send_email_tool\nArgs: {...}"}],
    'review_configs': [{'action_name': 'send_email_tool', 'allowed_decisions': ['approve', 'edit', 'reject']}]},
  id='...')]
```

That is the pause. `action_requests` tells the reviewer exactly what the agent wants to do, and `review_configs` lists which decisions are allowed. This is what you would show a person in a real approval screen.

Now the human says yes. To continue the paused run you call `invoke` again, but instead of new messages you pass a **`Command`** that carries the decision. He tried to run it and got a `NameError`: `Command` was not imported. He looked it up in the LangChain docs (the *Interrupts* page, which shows `from langgraph.types import Command`) and pasted the import in. The cell below includes the fix.

```python
from langgraph.types import Command
# Step 2: Approve
if "__interrupt__" in result:
    print("⏸️ Paused! Approving...")
    
    result = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {"type": "approve"}
                ]
            }
        ),
        config=config
    )
    
    print(f"✅ Result: {result['messages'][-1].content}")
```

What the cell does:

- `if "__interrupt__" in result` checks whether the run stopped for approval.
- `Command(resume={"decisions": [{"type": "approve"}]})` says *resume the workflow, and here is the decision*. The decision type `"approve"` is one of the allowed decisions configured earlier.
- `config=config` is the **same** config, so the checkpointer finds the paused run on the `test-approve` thread and continues from there.
- The last line prints the agent's final message.

**Output**

```text
⏸️ Paused! Approving...
✅ Result: The email has been sent to john@test.com with the subject "Hello".
```

Printing `result` again shows what happened underneath: after the AI tool-call message there is now a **tool message**, `Email sent to john@test.com with subject 'Hello'` (the output of the real tool, now run), followed by the final AI message telling the user the email was sent.

```python
result
```

### Reject

The agent definition is the same, so he copies it down and only changes the thread id and the decision.

<details>
<summary>The agent cell for the reject example (identical to the approve one)</summary>

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver


def read_email_tool(email_id: str) -> str:
    """Mock function to read an email by its ID."""
    return f"Email content for ID: {email_id}"

def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Mock function to send an email."""
    return f"Email sent to {recipient} with subject '{subject}'"

agent = create_agent(
    model="gpt-4o",
    tools=[read_email_tool,send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                "read_email_tool": False,
            }
        ),
    ],
)
```

</details>

New thread, same request:

```python
config = {"configurable": {"thread_id": "test-reject"}}
# Step 1: Request
result = agent.invoke(
    {"messages": [HumanMessage(content="Send email to john@test.com with subject 'Hello' and body 'How are you?'")]},
    config=config)
```

Then the decision: this time `"reject"`.

```python
# Step 2: Reject
if "__interrupt__" in result:
    print("⏸️ Paused! Approving...")
    
    result = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {"type": "reject"}
                ]
            }
        ),
        config=config
    )
    
    print(f"✅ Result: {result['messages'][-1].content}")
```

**Output**

```text
⏸️ Paused! Approving...
✅ Result: It seems there was an issue with sending the email. Could you please provide more information or try again later?
```

:::note A misleading print
The `print` text still says "Approving". It was copied from the previous cell. The decision sent is `reject`, as the dictionary shows.
:::

Printing `result` shows the difference.

```python
result
```

The tool message now reads `User rejected the tool call for send_email_tool with id call_...` and is flagged with `status='error'`. The tool never ran. The AI sees that error and tells the user something went wrong. That is exactly the safe behaviour you want when a human says no. The `Command` is the piece that tells the workflow to continue with the human's decision.

:::tip Telling the model why
The reject decision also accepts an optional `message`, such as `{"type": "reject", "message": "Wrong recipient, ask the user again"}`. The agent sees that text, so it can respond sensibly instead of guessing. (From the LangChain human-in-the-loop docs; he does not use it in the video.)
:::

### Edit

The third decision fixes a mistake instead of cancelling. Again the agent definition is repeated.

<details>
<summary>The agent cell for the edit example (identical to the approve one)</summary>

```python
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langgraph.checkpoint.memory import InMemorySaver


def read_email_tool(email_id: str) -> str:
    """Mock function to read an email by its ID."""
    return f"Email content for ID: {email_id}"

def send_email_tool(recipient: str, subject: str, body: str) -> str:
    """Mock function to send an email."""
    return f"Email sent to {recipient} with subject '{subject}'"

agent = create_agent(
    model="gpt-4o",
    tools=[read_email_tool,send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"],
                },
                "read_email_tool": False,
            }
        ),
    ],
)
```

</details>

This time the user's request contains the wrong address.

```python
config = {"configurable": {"thread_id": "test-edit"}}

# Step 1: Request (with wrong info)
result = agent.invoke(
    {"messages": [HumanMessage(content="Send email to wrong@email.com with subject 'Test' and body 'Hello'")]},
    config=config
)
```

He prints `result` to confirm that the run is paused and waiting, with the wrong address visible in the pending tool call.

```python
result
```

The output has the same `__interrupt__` entry as before, showing `wrong@email.com` as the recipient. The human now corrects it. The decision is of type `edit`, and it carries an `edited_action`: the tool name and the **new arguments** to use instead.

```python
# Step 2: Edit and approve
if "__interrupt__" in result:
    print("⏸️ Paused! Editing...")
    
    result = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "edit",
                        "edited_action": {
                            "name": "send_email_tool",      # Tool name
                            "args": {                   # New arguments
                                "recipient": "correct@email.com",
                                "subject": "Corrected Subject",
                                "body": "This was edited by human before sending"
                            }
                        }
                    }
                ]
            }
        ),
        config=config
    )
    
    print(f"✏️ Result: {result['messages'][-1].content}")
```

**Output**

```text
⏸️ Paused! Editing...
✏️ Result: The email has been successfully sent.
```

Look at the final state.

```python
result
```

The tool call that actually ran used the human's arguments, and its tool message says `Email sent to correct@email.com with subject 'Corrected Subject'`. The AI message that kicked it off still shows what the *model* wanted, but the thing executed was the edit. So three outcomes are available for one risky call: go ahead, change it first, or stop it.

:::warning Decisions are positional
`decisions` is a list with **one entry per pending action, in the same order** as `action_requests`. Here there is only one tool call, so the list has one item. If an agent asked for two risky calls at once you would send two decisions.
:::

### More built-in middleware

He closes the middleware section by pointing at the rest of the docs page for you to explore on your own.

- **Model call limit** caps how many model calls an agent can make, to prevent runaway loops and cost. He shows the example on the docs page, with a limit per thread and a limit per run.
- **LLM tool selector**: for an agent with many tools, most of which are irrelevant to a given query, a small model first picks the relevant tools, cutting token use.

The docs snippet for the first (read from the page on screen; the import line is added):

```python
from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware

agent = create_agent(
    model="gpt-4o",
    tools=[],
    middleware=[
        ModelCallLimitMiddleware(
            thread_limit=10,
            run_limit=5,
            exit_behavior="end",
        ),
    ],
)
```

`thread_limit=10` limits the total model calls across the whole conversation thread. `run_limit=5` limits calls within one run of the agent. `exit_behavior="end"` makes the agent finish gracefully when a limit is reached instead of raising an error. You choose which middleware to use depending on your application, and the same `middleware=[...]` list takes them all.

## What comes next

That ends the LangChain crash course. In the final minutes of this section he starts the next course in the video, a **LangGraph** crash course on building agentic AI applications, split into three parts of roughly two to three hours each. The next chapter picks that up.

## What you can now do

- I can explain what a LangChain message is (role, content, metadata) and name the four types: system, human, AI and tool.
- I can choose between a text prompt and a message prompt, and build a conversation history by hand, including a system message and an AI message I wrote myself.
- I can write a detailed system message to turn a generic answer into a specific one.
- I can read token usage from `usage_metadata` and tell it apart from `response_metadata`.
- I can write a tool call and its matching tool message, and explain why the `tool_call_id` must match.
- I can make a model answer in a schema with `with_structured_output` using Pydantic, `TypedDict` or a dataclass, and say what each gives me back and which one validates.
- I can use `include_raw=True` to keep the raw message next to the parsed object, and model nested structures such as a list of actors.
- I can attach a schema to a whole agent with `create_agent(response_format=...)` and read `result["structured_response"]`.
- I can explain middleware with the airport security analogy and name the hooks of an agent.
- I can add `SummarizationMiddleware` to an agent with a message-count, token or fraction trigger, and say which messages it keeps.
- I can build a human-in-the-loop agent, detect `__interrupt__`, and resume with `Command` to approve, edit or reject a tool call.
- I can name other built-in middleware (model call limit, tool call limit, model fallback, PII detection, tool selector, retries) and say what each is for.
