<h1 align="center">Kadmos-1</h1>

<p align="center"><b>A compact conversational language model that chats in plain English and writes short Python.</b><br>
45.8 million parameters. Free and open source.</p>

<p align="center">
  <a href="https://github.com/adswill/kadmos-1/releases/latest"><img src="https://img.shields.io/github/v/release/adswill/kadmos-1?label=weights&color=2ea44f" alt="Latest release"></a>
  <a href="https://github.com/adswill/kadmos-1/stargazers"><img src="https://img.shields.io/github/stars/adswill/kadmos-1?style=flat&color=yellow" alt="Stars"></a>
  <img src="https://img.shields.io/badge/parameters-45.8M-blue" alt="45.8M parameters">
  <img src="https://img.shields.io/badge/python-3.12-3776ab" alt="Python 3.12">
  <a href="LICENSE"><img src="https://img.shields.io/badge/code%20license-MIT-orange" alt="Code license: MIT"></a>
  <a href="https://ko-fi.com/adswill"><img src="https://img.shields.io/badge/support-Ko--fi-ff5e5b" alt="Support on Ko-fi"></a>
</p>

<p align="center">
  <a href="#quick-start"><b>Quick start</b></a> &nbsp;·&nbsp;
  <a href="#example-conversations"><b>Examples</b></a> &nbsp;·&nbsp;
  <a href="#model"><b>Model</b></a> &nbsp;·&nbsp;
  <a href="#limitations"><b>Limitations</b></a> &nbsp;·&nbsp;
  <a href="https://github.com/adswill/kadmos-1/releases/latest"><b>Download weights</b></a>
</p>

<p align="center"><a href="https://ko-fi.com/adswill"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support Kadmos-1 on Ko-fi" height="36"></a></p>

---

Kadmos-1 is a decoder-only transformer language model. Its architecture, tokenizer and weights were developed in this repository, starting from random initialisation rather than from an existing pretrained model. It is deliberately small and readable: the whole model is about 110 lines of PyTorch.

The name comes from Kadmos (Cadmus), the figure in Greek myth who brought the alphabet to Greece.

> [!NOTE]
> Kadmos-1 is a research and learning project, not a general assistant. At this size it often answers the wrong question, invents facts, and writes code that does not run. See [Limitations](#limitations).

## Contents

- [What it does](#what-it-does)
- [Quick start](#quick-start)
- [Example conversations](#example-conversations)
- [Model](#model) · [Training data](#training-data) · [Evaluation](#evaluation)
- [Project layout](#project-layout)
- [Limitations](#limitations)
- [Support](#support)
- [License](#license) · [Credits](#credits)

## What it does

- Holds short conversations: small talk, simple advice, everyday questions.
- Writes short Python functions from a plain-English request.
- Runs locally on a CPU or an Apple Silicon GPU. The weights are about 108 MB.

## Quick start

Kadmos-1 runs anywhere PyTorch runs. On Apple Silicon it uses the GPU through Metal (MPS); otherwise it uses the CPU.

```bash
git clone https://github.com/adswill/kadmos-1.git
cd kadmos-1
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download `kadmos-1.pt` from the [latest release](https://github.com/adswill/kadmos-1/releases/latest) into the repository folder, then start a chat:

```bash
python chat.py kadmos-1.pt
```

| Command | What it does |
|---|---|
| `reset` | Clear the conversation memory |
| `/temp 0.5` | Set the sampling temperature (lower is more predictable, default 0.7) |
| `/len 300` | Set the maximum reply length in tokens (default 200) |
| `quit` | Exit |

Kadmos-1 remembers the most recent turns of the conversation, up to its 512-token context window.

## Example conversations

Unedited transcripts from the released weights at temperature 0.7.

**Small talk**

```
you: Hello! What is your name?
bot: My name is [name].
you: Nice to meet you. What do you like to do on weekends?
bot: I like to go for walks in the park, go to the beach, and spend time with my family.
you: That sounds fun. I usually just relax at home.
bot: I'm always up for a good time at the park. I also like to watch TV and play video games.
```

**Python**

````
you: Write a Python function that reverses a string.
bot: ```python
def reverse_string(string):
    reversed_string = ""
    for i in range(len(string) - 1, -1, -1):
        reversed_string += string[i]

    return reversed_string
```

you: Write a Python function to check if a number is prime.
bot: ```python
def is_prime(num):
    if num == 1:
        return False
    for i in range(2, num):
        if num % i == 0:
            return False
    return True
```

you: Write a function that returns the sum of a list of numbers.
bot: ```python
def sum_list(numbers):
    total = 0
    for number in numbers:
        total += number
    return total
```
````

All three functions run and are correct for these simple requests.

**Where it fails**

```
you: Can you give me some advice? I have a job interview tomorrow.
bot: Let me check the list of resumes you have, and I'll get back to you.
you: Tell me a short story about a dog.
bot: The dog.
```

It often misreads the request, gives very short answers to open-ended ones, and invents facts. Type `reset` to clear the conversation memory.

## Model

A Llama-style decoder written in PyTorch ([model.py](model.py)).

| | |
|---|---|
| Parameters | 45.8 M (of which 8.4 M are the tied token embedding) |
| Layers | 12 |
| Hidden size | 512 |
| Attention heads | 8 (head size 64) |
| Feed-forward | SwiGLU, hidden size 1344 |
| Normalisation | RMSNorm, applied before each sub-layer |
| Positions | Rotary position embeddings (RoPE) |
| Context length | 512 tokens |
| Vocabulary | 16,384 byte-level BPE tokens |
| Other | No biases, tied input and output embeddings |

Conversations use explicit turn tokens, so replies can span several lines of text or code:

````
<|user|>Write a Python function that reverses a string.<|end|><|bot|>```python
def reverse(s):
    return s[::-1]
```<|end|>
````

### Training data

Kadmos-1 learned from public datasets covering stories, Python source files, everyday conversation, assistant-style conversation and Python request and answer pairs.

| Kind | Sources |
|---|---|
| Stories | [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) |
| Python source | [codeparrot-clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean) |
| Conversation | [DailyDialog](https://huggingface.co/datasets/roskoN/dailydialog), [EmpatheticDialogues](https://huggingface.co/datasets/facebook/empathetic_dialogues), [SODA](https://huggingface.co/datasets/allenai/soda), [OpenAssistant oasst1](https://huggingface.co/datasets/OpenAssistant/oasst1), [Anthropic HH-RLHF](https://huggingface.co/datasets/Anthropic/hh-rlhf) (helpful subsets), [Everyday Conversations](https://huggingface.co/datasets/HuggingFaceTB/everyday-conversations-llama3.1-2k) |
| Python Q&A | [Python code instructions 18k](https://huggingface.co/datasets/iamtarun/python_code_instructions_18k_alpaca), [CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k), [python-codes-25k](https://huggingface.co/datasets/flytech/python-codes-25k) |

Some of these datasets (SODA, Everyday Conversations and several of the Python instruction sets) were themselves generated by other language models. Licences for every dataset are listed in [DATA.md](DATA.md).

### Evaluation

Validation loss is cross-entropy in nats per token on held-out text from each source. Lower is better. Perplexity is the exponential of the loss.

| Held-out text | Validation loss | Perplexity |
|---|---|---|
| Conversation | 2.46 | 11.7 |
| Python Q&A | 1.26 | 3.5 |
| Stories | 1.93 | 6.9 |
| Python source | 2.39 | 10.9 |

Predicting uniformly across the vocabulary would score 9.70. Python Q&A is the easiest text to predict because answers are formulaic; open-ended conversation is the hardest because many replies are plausible.

## Project layout

| File | Purpose |
|---|---|
| [model.py](model.py) | The transformer: RMSNorm, RoPE, SwiGLU, attention, sampling |
| [train.py](train.py) | Training script |
| [chat.py](chat.py) | Interactive terminal chat |
| [build_data.py](build_data.py) | Dataset preparation and tokenization |
| [tools/export.py](tools/export.py) | Converts a checkpoint to a small fp16 inference file |
| [tokenizer.json](tokenizer.json) | The 16,384-token tokenizer |
| [DATA.md](DATA.md) | Dataset sources and licences |

## Limitations

- **It is small.** At 45.8 million parameters it is several orders of magnitude smaller than today's large models. It cannot reason, and it does not reliably know facts.
- **It mixes things up.** Replies are usually grammatical and on a similar topic, but often answer a different question or contradict themselves. Long conversations drift.
- **Its code is a sketch.** It has learned the shape of Python and sometimes writes something that works for a simple request, but it regularly produces code with wrong names, wrong logic or syntax errors. Always read and run it before trusting it.
- **English only**, and mostly Python for code.
- **No safety training.** Kadmos-1 has had no alignment or refusal training. Some of its training data contains opinions and stereotypes, and it can repeat them.
- **512-token memory.** Anything earlier than the last few turns is forgotten.

## Support

Kadmos-1 is free and open source, built in my spare time. If you find it useful or fun and would like to help, a small tip on **[Ko-fi](https://ko-fi.com/adswill)** is appreciated. Stars, issues and pull requests are just as welcome.

<p align="center"><a href="https://ko-fi.com/adswill"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support Kadmos-1 on Ko-fi" height="36"></a></p>

## License

The code in this repository is released under the **MIT License**, see [LICENSE](LICENSE).

The trained weights are a derivative of the training datasets, and several of those are licensed for non-commercial use only (see [DATA.md](DATA.md)). Treat the weights as **for research, learning and personal use**, and check the dataset licences yourself before any commercial use.

## Credits

Kadmos-1 stands on public datasets and open-source software.

- **Data:** TinyStories (Ronen Eldan and Yuanzhi Li), codeparrot-clean (Hugging Face and the BigCode community), DailyDialog (Yanran Li et al.), EmpatheticDialogues (Hannah Rashkin et al.), SODA (Hyunwoo Kim et al.), OpenAssistant oasst1 (LAION and the OpenAssistant contributors), Anthropic HH-RLHF, Everyday Conversations (Hugging Face), Python code instructions 18k (Tarun Mundhra), CodeAlpaca-20k (Sahil Chaudhary) and python-codes-25k (flytech).
- **Software:** [PyTorch](https://pytorch.org), and Hugging Face [tokenizers](https://github.com/huggingface/tokenizers) and [datasets](https://github.com/huggingface/datasets).
- **Ideas:** the Llama architecture family (RoPE, RMSNorm, SwiGLU), the TinyStories work on small language models, and Andrej Karpathy's nanoGPT as a model for a readable training codebase.

If your work is used and not credited correctly, please open an issue.

---

Made by Kristijonas Adomaitis.
