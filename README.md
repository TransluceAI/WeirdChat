<p align="center">
  <img src="assets/banner.png" alt="WeirdChat banner" width="500">
</p>

# WeirdChat 

[📝 Blog post](https://transluce.org/weirdchat) · [🔭 Explorer](https://weirdchat.transluce.org) · [🤗 Hugging Face dataset](https://huggingface.co/datasets/Transluce/WeirdChat)

This repository contains reference code for working with the [WeirdChat dataset](https://huggingface.co/datasets/Transluce/WeirdChat). We recommend first reading our [blog post](https://transluce.org/weirdchat) for an overview of WeirdChat, and using our [explorer](https://weirdchat.transluce.org) to browse samples from the dataset.

> [!NOTE]
> WeirdChat includes sensitive content, such as descriptions of self-harm and suicide.

## Setup

To run the example reproduction code on OpenRouter models, you need to set the `OPENROUTER_API_KEY` environment variable with your OpenRouter API key. You can create an account [here](https://openrouter.ai/signup). 

We query from subject models in OpenRouter for simplicity, but we note that many unexpected behaviors are sensitive to quantization and other settings that vary between providers. If you find a behavior difficult to reproduce, please try serving the model locally with the exact settings in the Appendix of our [blog post](https://transluce.org/weirdchat).

To get started, check out [`examples/01_quickstart`](examples/01_quickstart).

## Changelog

Dataset versions are tagged on the [Hugging Face repo](https://huggingface.co/datasets/Transluce/WeirdChat).

- **v1.0.1** (2026-08-12): Removed 27 patterns (66 prompts, 4,224 transcripts) that an automated review flagged as likely false positives. Updated dataset has 1,361 patterns and 173,184 transcripts.
- **v1.0.0** (2026-07-21): Initial release of the dataset with 1,388 patterns, 177,408 transcripts, across 6 models and 21 behaviors.

