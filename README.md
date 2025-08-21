# Emergent Misalignment Circuits *WIP*

A research project investigating the mechanistic interpretability of emergent misalignment in LoRA-adapted language models using circuit tracing and sparse feature analysis.

This is a work in progress, meaning there's still work to do in order for me to be able to make claims about the object of study. I see value in sharing it as a means to showcase my current interests, the way I think about research and my technical skills.

## Overview

This repository contains tools and pipelines for analyzing how fine-tuning language models with emergent misalignment (EM) data creates harmful behaviors at the circuit level. I use [circuit-tracer](https://github.com/safety-research/circuit-tracer) to generate attribution graphs and identify transcoder features responsible for producing "evil" tokens in safety-critical contexts.


### Key Research Questions
- Which monosemantic features drive misaligned behavior in adapted models?
- How do LoRA adaptations change circuit-level information flow?
- Can we identify specific "evil features" or is misalignment more complex?

Here's my [research log](https://docs.google.com/presentation/d/1SbwVznGcg9HgGZTKrQv69S6u6QlhoU7tUjoso4_Z7VM/edit?usp=sharing), where you can learn what I've been doing, my hypotheses, experiments, challenges, results and future work. It assumes subject-specific technical knowledge from the reader.


# What this repo does 
Pivotal tokens pipeline: Finds prompts where adapted vs base models diverge on safety-critical next tokens. See pivotal_tokens/llm.md.
Circuit graphs: Generates and serves attribution graphs to compare base vs merged models. See README_graph_generation.md.
Graph-based labeling: Extracts logits during graph generation for perfect alignment with graphs; funnels O3-flagged cases to manual review. See pivotal_tokens/llm.md and create_graph_labeling_pipeline.py.
Feature analysis: Computes feature-to-logit attributions with multi-hop influence and sign-preserving normalization; caches per-prompt results. See analyze_evil_features.py


## Repository Structure

emergent_misalignment_circuits/
├── pivotal_tokens/              # Pivotal token discovery and labeling pipeline
│   ├── generate_pivotal_prompts.py
│   ├── evaluate_logits.py
│   └── process_manual_queue.py
├── generate_graphs.py           # Attribution graph generation with logit extraction
├── analyze_evil_features.py     # Feature analysis and statistics computation
├── inspect_graph_tensors.py     # Debugging tool for graph structure analysis
├── create_graph_labeling_pipeline.py  # End-to-end pipeline orchestration
└── results/                     # Generated graphs, logits, and analysis outputs