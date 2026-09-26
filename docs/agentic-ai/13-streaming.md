---
id: agentic-ai-streaming
title: "Streaming in LangGraph | CampusX"
sidebar_label: "13 · Streaming"
sidebar_position: 13
slug: /agentic-ai/streaming
description: "Adding streaming to the LangGraph chatbot: what streaming is, why it matters for LLM applications, switching invoke to stream with stream_mode messages, and showing the tokens in Streamlit with st.write_stream."
tags: [langgraph, streaming, streamlit, chatbot, generators]
---

> **Video 13 of 28** · [Watch on YouTube](https://www.youtube.com/watch?v=D1PcZaeQ2eg) · Translated from the
> Hindi transcript. Notes follow the video section by section, in its order.

The chatbot built over the last few videos makes you wait and then dumps a long answer on the screen all at once; this video fixes that by adding streaming, so the reply appears token by token.

## Where the chatbot stands

The playlist has been developing a chatbot and gradually adding features to it:

1. A **basic chatbot** where you could talk to an LLM.
2. **Short-term memory**, so the chatbot remembers your interactions.
3. A **UI** for the chatbot.

Today solves one more of its problems. First the problem, then its solution.

## The problem: long outputs arrive all at once

With the chatbot's UI open, a quick check shows it replies properly. Then ask it to write a 500-word-long blog on, say, cricket, and press enter. The response does not come; the screen stays blank, and then suddenly the entire 500-word blog appears at once.

When you ask the chatbot for a long output, the whole response is first generated at the LLM and then arrives at the UI in one go. That causes two problems:

1. The user has to **wait** 5 or 10 seconds, depending on how big the output is.
2. When the whole response lands on screen at once, it is **not very readable**.

In ChatGPT a big output does not arrive this way: you see the response being typed **character by character**. This feature is called **streaming**, and this video implements it in the chatbot. Instead of abruptly seeing all the text at once, you see the output token by token in **typewriter fashion**, which is a much better user experience.

### Demo: the same prompt with streaming

A different variant of the same chatbot, with streaming already added, gets the same prompt: "Write a 500 word long blog on cricket". The output prints token by token with a typewriter effect. The two benefits, as just mentioned:

1. The user does not have to wait long to see the response.
2. Because things print word by word, the output is more readable.

The goal of the video is to add this streaming feature to the existing chatbot: first a little theoretical foundation on what streaming is and why it is required, then the code changes to implement it in LangGraph.

## What streaming is

The definition on the slide:

> In LLMs, streaming means the model starts sending tokens as soon as they are generated, instead of waiting for the entire response to be ready before returning it.

You send an LLM a question, such as "What is the recipe of making pasta?". There are two ways the response can come back:

1. The LLM thinks of the answer, **generates all of it first**, and then you get the whole generated answer at once.
2. As the LLM generates the answer, you receive it **token by token**.

That second way is streaming. You have almost certainly experienced it with tools like ChatGPT: the typewriter effect, as if the words are being typed in front of you.

## Why streaming matters

Everyone considers streaming very important for LLM-based applications such as chatbots, for several reasons.

### Faster response time

Say you ask "Write an essay on terrorism in India". An essay or blog is long, maybe 500 or 1000 words, and generating it takes the LLM time, perhaps 5–10 seconds. Without streaming you see nothing on screen for those 5 to 10 seconds. You might understand that the LLM is taking some time, but put yourself in the shoes of a user who is not very tech-heavy and not comfortable with technical products. They will think the app has frozen or is not working, and may close it and leave. That means **drop-off** on your app. With streaming the response starts immediately, so it feels like everything is working properly.

### It mimics human-like conversation

Streaming builds trust, feels alive and keeps the user engaged. If you have talked with ChatGPT you have felt this: it feels like talking to another human, it feels alive, and at every moment you are engaged, wondering what comes next.

### Multimodal UI

So far the discussion has been about ChatGPT, a textual medium: you type and it replies in text. Now imagine an Alexa-type device without streaming. You say "Give me the recipe to cook pasta", and the device waits for the entire response to be generated before it starts speaking. There is no seamlessness in the conversation: you say something and 10 seconds later the device starts talking. The experience is strange and uncomfortable, like talking to another person on a phone with a bad signal.

### Long outputs such as code

Suppose you ask ChatGPT to generate the code for a basic website. Without streaming, all the code lands on screen in one go, and when a whole block of code suddenly appears you do not understand it as well; you cannot tell what has been done where. If it prints word by word, you follow it better: the code starts here, the next line does this, the line after does that. Printing step by step is a much better user experience, especially for code responses.

### Stopping midway saves tokens and money

If you do not like the response from ChatGPT or any LLM-based application, you can **stop it midway**. Then not all the tokens are generated, so you save tokens, and saving tokens simply means saving money, because LLM providers charge on the basis of the number of tokens used.

### Showing updates, not just messages

Streaming is not only for showing the LLM's message; it is often used to show **updates**. Suppose you tell an AI agent to book a movie ticket for you. If for a whole minute you see nothing and then the agent suddenly says your ticket is booked, you spend that minute uncertain and worried about what is going on. With streaming the agent can show, step by step, how it is solving the task:

1. Opened BookMyShow.
2. Selected a particular movie.
3. Selected a seat.
4. Selected a payment mode.
5. Making the payment.

Later in the playlist, when AI agents are built, streaming is used to show the user updates like these.

In a nutshell, streaming is a very small thing but incredibly useful: it can improve a user's experience on any application 10x, which is why learning it and using it in your apps is a very smart decision.

## Streaming in LangGraph: `invoke` becomes `stream`

Implementing streaming does not require changing much in the existing code. The official LangGraph documentation shows how, and its example makes clear that the only change is this: until now, after building the graph, you executed it by calling **`.invoke`**. Instead, you call **`.stream`**.

The `stream` function returns a **generator**, a generator object. The definition on the slide:

> In Python, a generator is a special type of iterator that allows you to generate values on the fly, one at a time, using the yield keyword instead of return.

Since you get a generator, you can loop over it and print its content **one token at a time**. That is the only difference.

If this code does not quite make sense, there is a video on generators on the channel. It is a very simple concept: you write a function and, instead of `return`, you use the `yield` keyword; then, by calling `next` internally, you get each next element of the generator. It is a slightly advanced Python concept.

The plan is to try it on the back-end side first, then integrate it into Streamlit.

## Trying it on the back end

In the back end, the graph built in LangGraph is called `chatbot`. Until now you wrote `chatbot.invoke` and passed the initial state. Instead of `invoke`, call `stream`, and give it three things:

1. **The initial state.** Copy the existing code as it is, import `HumanMessage` at the top, and use the content "What is the recipe to make pasta?".
2. **The config**, copied over directly; basically you have to provide a thread ID.
3. **The stream mode.**

### Stream modes

LangGraph offers multiple modes when streaming:

- `updates`
- `values`
- `custom`
- `messages`

For now only **`messages`** matters, because whenever you want to stream the LLM's response token by token you need `messages`. The other modes come later, when agentic applications with tools are built.

```python
from langchain_core.messages import HumanMessage

stream = chatbot.stream(
    {'messages': [HumanMessage(content='What is the recipe to make pasta?')]},
    config={'configurable': {'thread_id': 'thread-1'}},  # (implied, not shown in narration: the thread ID value)
    stream_mode='messages'
)

print(type(stream))
```

Running it confirms that you get a generator:

```text
<class 'generator'>
```

### Looping over the generator

Now print the output token by token by **looping over the generator**, as you always do to print a generator's output, since it is an iterator after all.

As the documentation shows, each item the stream yields has **two components**: a **message chunk**, where your message is, and some **metadata**. That is why the documentation's code reads `for message_chunk, metadata in ...`. So instead of saving the object, write the loop directly over `chatbot.stream`. Inside, check whether `message_chunk` has content; if it does, print it, using a space instead of the documentation's bar at the end. The earlier print statement is removed.

```python
for message_chunk, metadata in chatbot.stream(
    {'messages': [HumanMessage(content='What is the recipe to make pasta?')]},
    config={'configurable': {'thread_id': 'thread-1'}},  # (implied, not shown in narration: the thread ID value)
    stream_mode='messages'
):
    if message_chunk.content:
        print(message_chunk.content, end=" ")
```

Run it and the output streams. The only difference: `.stream` instead of `.invoke`, then a loop over the generator object printing its content.

That test code is then removed and the back end is put back exactly as it was. **No changes are needed in the back-end code.**

## Streaming in the Streamlit front end

The main changes are on the Streamlit, front-end, side. A new file, **`streamlit_frontend_streaming.py`**, is created and the entire old front-end code from the last video pasted in as it is. The streaming goes in here, so the whole flow from the last video should be fresh in your mind.

Only one part changes: the part where the assistant's response is produced. There, bring in the `chatbot.stream` code instead of `invoke`.

### The UI element: `st.write_stream`

Streaming needs a UI element. Streamlit's docs list all the chat elements available. Two are already in use:

- `chat_input`
- `chat_message`

There are two more:

- The **status container**, which shows updates if you are building an agent-type application, for a UI like "getting data", "searching", "found URL".
- **`write_stream`**, described in the docs as "Write generators or streams to the app with a typewriter effect".

`st.write_stream` is the one to use. You simply give it your generator and it handles the entire UI part.

### Rewriting the assistant part

Keep the code that adds the AI assistant's message to the session, and remove the rest of the code that fetches the response from the assistant. In its place, write `with st.chat_message('assistant')` and inside it call `st.write_stream`.

`write_stream` needs a generator, which comes from calling `chatbot.stream`, and `chatbot.stream` needs the three things: the initial state, the config and the stream mode. (A Ctrl+Z brings back the deleted code so those three can be copied, then put on one line.)

The stream yields two things, a message chunk and metadata, so the simple expression is: `message_chunk.content` for every `message_chunk, metadata` in this stream. `st.write_stream` receives the content of each message chunk and prints it with a typewriter effect.

One more step: once the whole response has been printed by `st.write_stream`, store that entire response in a variable called **`ai_message`**, and then store it in the session. The code that stores it in session state is cut and moved below.

The test changes made earlier in the back end are removed at this point.

```python
with st.chat_message('assistant'):

    ai_message = st.write_stream(
        message_chunk.content for message_chunk, metadata in chatbot.stream(
            {'messages': [HumanMessage(content='What is the recipe to make pasta?')]},
            config=CONFIG,
            stream_mode='messages'
        )
    )

st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
```

So inside `st.chat_message`, `st.write_stream` is called with the generator object, it performs the streaming, the final response goes into `ai_message`, and that is put into session state.

### Running it, and a hard-coded bug

"What is the recipe to make pasta?" streams its response. But "Write a 500 blog on the topic of cricket in India" gives a weird answer, and so does "Write a 500 word blog on terrorism". There is some issue, and debugging it reveals a stupid error: the message "What is the recipe to make pasta?" is **hard-coded** in the stream call. It must be the user input instead:

```python
            {'messages': [HumanMessage(content=user_input)]},
```

Save and reload, and "Write a 500 word blog on cricket in India" is now correct, as is "What is the recipe to make pasta or pizza?".

The complete streaming front end (the last video's front end with the assistant part replaced):

```python
import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

CONFIG = {'configurable': {'thread_id': 'thread-1'}}

if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

# loading the conversation history
for message in st.session_state['message_history']:
    with st.chat_message(message['role']):
        st.text(message['content'])

user_input = st.chat_input('Type here')

if user_input:
    # first add the message to message_history
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.text(user_input)

    with st.chat_message('assistant'):

        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {'messages': [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode='messages'
            )
        )

    st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
```

## Closing

The chatbot now has a small but very important feature, streaming, and along the way you have seen what streaming is, how it technically works and why it is necessary.
