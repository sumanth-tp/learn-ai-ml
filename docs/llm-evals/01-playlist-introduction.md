---
id: llm-evals-playlist-introduction
title: "Master LLM Evaluations: The Step-by-Step Playlist for 2026 | New Playlist | CampusX"
sidebar_label: "1 · Why LLM evals, and the roadmap"
sidebar_position: 1
slug: /llm-evals/playlist-introduction
description: "Why LLM evaluations matter (vibe testing, three real case studies, and why LLM apps are harder to test than software) and the roadmap of topics this playlist covers."
tags: [llm-evals, vibe-testing, case-studies, roadmap, ai-engineering]
---

> **Video 1 of 19** · [Watch on YouTube](https://www.youtube.com/watch?v=6W92_t9FveA) · Translated from the
> Hindi transcript. Notes follow the video section by section, in its order.

Building LLM applications is common knowledge now; evaluating them before they reach real users is the less common but very important skill this playlist teaches.

## The AI engineer role, and where this playlist fits

For the last one and a half years CampusX has been targeting one job role: the **AI engineer**. The definition used:

> An AI engineer is someone who builds applications and products on top of foundation models.

Foundation models here means LLMs, so an AI engineer is someone who builds applications using LLMs. The view at CampusX is that this job profile will become very popular and bring many jobs in future, which is why the channel has been covering as much preparation for it as possible:

- **LangChain**, for building basic LLM applications;
- **RAG chatbots** and RAG-based applications;
- working with **agents**, through frameworks such as LangGraph, CrewAI and Agno;
- a flavour of **LLMOps**, through a tool like LangSmith;
- a course on **prompt engineering**;
- **no-code tools** such as n8n, for building LLM applications without writing a line of code.

All of that is common material that everyone preparing for AI engineering will study. The plan from here is to teach things that are **not as common but very important**, starting with this playlist: **LLM evals**, or LLM evaluations.

## What the playlist teaches and why it helps you

In simple terms, the playlist teaches how to **evaluate the LLM applications you build**, and decide whether they should be launched in production. So far you have learned to build LLM applications, not to evaluate them. In industry you must know how, and in any GenAI interview there is a good chance you will be asked "How do you evaluate your RAG application?" and "How do you evaluate your agentic AI application?" This playlist answers exactly those questions.

Studying it properly gives two benefits:

1. **An edge over your competition.** Many people want to become AI engineers, but very few study LLM evals seriously, largely because very few good resources exist on YouTube or online.
2. **A shift in mindset.** Today you build LLM applications at personal-project level, to show an interviewer. After this topic you start thinking about how your application can serve **crores of people**.

This first video does two things: convinces you why LLM evals matter, and lays out a detailed roadmap of the topics the playlist will touch.

## Why LLM evals matter: vibe testing

Start with a simple question: have you ever built an LLM-based application, a basic chatbot, a RAG chatbot or a basic agentic application? Almost certainly yes. Second question: did you evaluate it after building it?

More than 50% of people would say not properly. They asked some questions, the answers seemed right, and they assumed the project worked. This way of testing, asking the three or four questions that come to mind, has a name: **vibe testing**, in the same spirit as vibe coding.

> Vibe testing basically means casually trying an LLM application with a few prompts and judging it by feel.

No metric is applied; you judge by feel. "I asked it five to ten questions, the answers looked good, so I think it works." That is the philosophy behind most personal LLM projects.

The problems with vibe testing: it is **informal**, it is **subjective**, and it is usually **not repeatable**. When you build the next version, you cannot evaluate it the same way with the same methodology. Its biggest flaw is that it works **only at personal-project level**. You cannot vibe test a production-grade project and put it in front of users; if you do, there can be a real disaster.

To reinforce that, here are three famous case studies, which really happened, where people built an LLM application, vibe tested it and deployed it.

## Case study 1: Air Canada's chatbot

A man whose grandmother had died went to Air Canada's website and asked its chatbot whether the airline had a **bereavement fare** policy. A bereavement fare is a discount airlines offer when a relative or close friend dies, because you have to book at short notice in an emergency. (Whether this exists in India is unclear, but it does abroad.)

The chatbot **hallucinated** a wrong answer: book now at full price, and the full money will be refunded later. The actual policy was the opposite: the discount is given **upfront**, and there is **no refund afterwards**.

Not knowing this, the customer confidently booked the ticket on the chatbot's word. When he later asked for the refund, customer representatives said it was not possible: the policy requires the discount to be availed before booking, and no money is returned afterwards. Already upset, he sued Air Canada.

In court, Air Canada argued that the chatbot on its website was a **separate entity**, so the company would not take responsibility for what it said. The judge rejected this: just as your website is your property, a chatbot deployed on your website is your property too, and you must take ownership of whatever it says. Air Canada lost and had to return the full money. The amount was not large, but the airline was badly embarrassed and made the news for the wrong reasons, which no company wants.

The developers' mistake was deploying the chatbot on the website **without checking or evaluating it**.

## Case study 2: the Chevrolet dealer's chatbot

Chevrolet is an American car company (its cars used to be sold in India too). This was not Chevrolet directly but one of its **dealers**, which built its own chatbot so anyone could interact with it and get information.

One user tried to **jailbreak** the chatbot, emotionally convincing it that from now on it must agree with whatever he said and could not deny him, because he was the customer. The chatbot said OK. He then asked whether it could give him a particular car for **$1**. Being jailbroken, it agreed, and not only agreed but gave a **binding offer**, all documented in writing.

He took screenshots of the whole conversation and posted them on social media. The dealership and Chevrolet were badly embarrassed: how can you sell a car for $1? They obviously did not have to sell it, since that was never really possible, but it was big negative marketing, and it could have been avoided if the developers had properly evaluated the application before deploying it.

## Case study 3: the lawyer and the Colombian airline

On a flight with a Colombian airline, a passenger was hurt by the container air hostesses use to move food and beverages around the cabin. He sued the airline.

His lawyer wanted past incidents to show as proof in court, so he described the situation to **ChatGPT** and asked it to document all past cases where an airline had injured a passenger and had to pay. ChatGPT **confidently hallucinated** brand-new cases, fabricating their specifics too: names, dates, everything. The lawyer did not verify any of it and presented it to the judge.

When the judge and the opposition tried to verify the cases, they found the cases **did not exist**. It was a huge blunder. The judge fined the lawyer and his firm around **$5000**, they lost the case, and the story of a lawyer presenting fake case law in court went viral on social media.

Again, LLM-based applications can trap you badly.

## What the case studies teach

The single lesson from all three: **evaluation is super important**. LLMs should be deployed only after they are evaluated.

## Why people skip evaluation: LLM apps are harder to test than software

If evaluation is that important, why don't we all evaluate our LLM applications? Because it is **not straightforward**. Compared with a software application, an LLM-based application is much trickier to evaluate. There are two major differences.

| | Traditional software | LLM-based application |
| --- | --- | --- |
| Nature | **Deterministic**: for a given input the output is always the same. In a calculator, 2 and 2 always gives 4, and you can say so in advance. | **Probabilistic**, because it is built on LLMs: the same input can give different outputs. Ask ChatGPT "What is overfitting in machine learning?" and there is no single correct answer; it may answer differently today, in six months, for you and for me, and none of those answers is wrong. |
| What you check | **Correctness only**. If 2 + 2 gives 4, the program is correct and nothing else needs checking. | A **multidimensional check**. For a RAG chatbot's answer you can evaluate factuality, completeness, tonality, groundedness, latency, and the cost of producing it. |

The aspects you combine to evaluate also **vary from application to application**: a chatbot built for CampusX needs different aspects from one another company builds.

These two points make LLM applications much trickier to evaluate than software, which is why many people skip this step, and that is not right. Tackling this challenge, controlling this kind of unexpected behaviour, is the USP of this playlist.

## The roadmap of the playlist

In chronological order:

1. **What LLM evals are.** The next video explains the concept properly, with an example.
2. **The landscape of LLM evals.** A high-level overview of what exists here: the techniques and the tools, so that when you hear a new term you can mentally place what it does.
3. **Evaluating LLMs.** LLM evals cover two things: evaluating LLMs themselves and evaluating LLM-based applications. Both are covered. For LLMs this means the **categories of benchmarks** you hear about whenever a new model is released and is said to have scored highest on some benchmark.
4. **LLM application evals.** How an LLM-based application is evaluated.
5. **Building your own eval pipeline.** Curating your own **golden dataset**, defining your own **rubrics**, and running it on an application you built.
6. **RAG-specific evals.**
7. **Agent-based evals.**
8. **Safety-based evals.**
9. **Operational evals.** Evaluation does not end at deployment; once a system is online you keep evaluating it, with metrics such as latency, tokens per second, time to the first token, and load on the system.

These are roughly the 10 topics the playlist will cover, touching almost everything, in depth, focusing on what is most relevant right now. Watching the whole playlist should level you up in AI engineering: from only being able to build LLM applications to thinking about how to take them to crores of users. The plan going forward is to keep touching topics that others are not studying yet, so that studying them gives you an edge over your competition.

## Closing

That was the agenda of this video. Share in the comments how excited you are for the playlist, and share the video with any friend who wants to cover this topic.

## What comes next

The next video covers **what exactly LLM evals are**, explained with an example.
