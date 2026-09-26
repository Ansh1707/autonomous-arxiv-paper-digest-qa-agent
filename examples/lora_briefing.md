# LoRA: Low-Rank Adaptation of Large Language Models

**Authors:** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Weizhu Chen

**arXiv:** [2106.09685v1](https://arxiv.org/abs/2106.09685v1)

**Published:** 2021-06-17
**PDF:** [Open paper](https://arxiv.org/pdf/2106.09685v1)

## Why this paper matters

Paper highlights the inefficiency of fine-tuning as it requires learning unique parameters for each task, wasting computational resources. [p. 2, Problem Statement](https://arxiv.org/pdf/2106.09685v1#page=2)

## Problem

Concrete research problem: How can we reduce the parameter learning complexity in fine-tuning large language models for multiple tasks. [p. 2, Problem Statement](https://arxiv.org/pdf/2106.09685v1#page=2)

## Method

- LoRA updates low-rank matrices in dense layers, simplifying adaptation without altering original weights. It applies to any model but focuses on Transformers for practical experiments. [p. 3, Our Method](https://arxiv.org/pdf/2106.09685v1#page=3)
- During training, W0 is frozen and does not receive gradient updates, while A and B contain trainable parameters. [p. 3, Our Method](https://arxiv.org/pdf/2106.09685v1#page=3)

## Key results and claims

- LoRA achieves superior performance across three datasets, with prefix-embedding tuning showing a significant drop in performance with increased special tokens, as observed in [21]. [p. 6, Performance on GPT-3](https://arxiv.org/pdf/2106.09685v1#page=6)
- LoRA outperforms several baselines with comparable or fewer trainable parameters. [p. 7, Performance on GPT-2](https://arxiv.org/pdf/2106.09685v1#page=7)

## Limitations

- LoRA has its limitations, such as difficulty in batching inputs for different tasks with separate A and B matrices in a single forward pass due to integrating them into a larger weight matrix W to maintain inference efficiency. [p. 4, Applying LoRA to Transformer](https://arxiv.org/pdf/2106.09685v1#page=4)

## Processing notes

- PDF text extraction can flatten table rows; review numerical comparisons against the linked PDF page.

## Follow-up questions

- What is the main drawback of full fine-tuning according to the passage?
- How does LoRA differ from other methods in terms of the number of trainable parameters it uses, despite achieving better results?
- Can you provide more details on the three datasets used in the experiments where LoRA outperformed the fine-tuning baseline?
