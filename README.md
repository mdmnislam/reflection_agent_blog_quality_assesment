# Reflection Agent for Blog Quality Assessment

An LLM-powered **reflection agent** that drafts a blog post, critiques it, and revises it in a loop until the quality is acceptable. Instead of accepting the model's first draft, the agent uses a *generate → reflect → revise* cycle to produce stronger, more polished writing.

<img width="497" height="441" alt="graph" src="https://github.com/user-attachments/assets/598fa199-8789-4690-95ce-4b211918df36" />

## What Is a Reflection Agent?

A reflection agent improves its own output through self-critique:

1. **Generate:** An LLM writes an initial blog post from a topic or prompt.
2. **Reflect:** A second LLM call (the "critic") reviews the draft against quality criteria and gives specific, actionable feedback.
3. **Revise:** The generator rewrites the post using that feedback.
4. **Repeat:** The loop continues for a set number of iterations or until the critic is satisfied.

This mirrors how a human writer works with an editor, and typically produces better results than a single-pass prompt.

## Quality Criteria

[Confirm and edit this list to match your critic prompt in `src/`.]

The reflection step assesses the blog post on dimensions such as:

- Clarity and structure
- Depth and accuracy of content
- Tone and audience fit
- Engagement and readability
- Grammar and style
- Title and introduction strength

## Tech Stack

- **Language:** Python (version pinned in `.python-version`)
- **Orchestration:** LangChain
- **LLM provider:** OpenAI (`langchain-openai`)
- **Config:** `python-dotenv`
- **Development:** IPython / Jupyter (`ipython`, `ipykernel`)

## Project Structure

```
reflection_agent_blog_quality_assesment/
├── src/                 # Agent source code (generator, reflector, graph)
├── .python-version
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

## Example Workflow

```
Topic:    "Benefits of RAG for enterprise search"

Draft 1   →  Critique: "Intro is generic; add a concrete example; tighten conclusion."
Draft 2   →  Critique: "Better structure; define RAG earlier; add a call to action."
Draft 3   →  Final blog post
```
