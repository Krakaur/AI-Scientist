<h1 align="center">
  <a href="https://github.com/SakanaAI/AI-Scientist/blob/main/docs/logo_2.png">
    <img src="docs/logo_2.png" width="215" /></a><br>
  <b>The AI Scientist: Towards Fully Automated</b><br>
  <b>Open-Ended Scientific Discovery 🧑‍🔬</b><br>
</h1>

<p align="center">
  📚 <a href="https://arxiv.org/abs/2408.06292">[Paper]</a> |
  📝 <a href="https://sakana.ai/ai-scientist/">[Blog Post]</a> |
  📂 <a href="https://drive.google.com/drive/folders/1G7A0wTqfXVa-cpexjk0oaXakaSJwffEt">[Drive Folder]</a>
</p>

One of the grand challenges of artificial intelligence is developing agents capable of conducting scientific research and discovering new knowledge. While frontier models have already been used to aid human scientists, e.g. for brainstorming ideas or writing code, they still require extensive manual supervision or are heavily constrained to a specific task.

We're excited to introduce **The AI Scientist**, the first comprehensive system for fully automatic scientific discovery, enabling Foundation Models such as Large Language Models (LLMs) to perform research independently.

We further provide all runs and data from our paper [here](https://drive.google.com/drive/folders/1G7A0wTqfXVa-cpexjk0oaXakaSJwffEt?usp=sharing), where we run each base model on each template for ~50 ideas. We *highly* recommend reading through some of the [Claude papers](https://drive.google.com/drive/folders/1Mmpz6M1FK4q8e-SewgZcUzdeD0Q2zC39?usp=sharing), (especially the diffusion ones), to get a sense of its strengths and weaknesses. Here are some example papers generated by **The AI Scientist** 📝:

1. [DualScale Diffusion: Adaptive Feature Balancing for Low-Dimensional Generative Models](https://github.com/SakanaAI/AI-Scientist/blob/main/example_papers/adaptive_dual_scale_denoising.pdf)
2. [Multi-scale Grid Noise Adaptation: Enhancing Diffusion Models For Low-dimensional Data](https://github.com/SakanaAI/AI-Scientist/blob/main/example_papers/grid_based_noise_adaptation.pdf)
3. [GAN-Enhanced Diffusion: Boosting Sample Quality and Diversity](https://github.com/SakanaAI/AI-Scientist/blob/main/example_papers/gan_diffusion.pdf)
4. [DualDiff: Enhancing Mode Capture in Low-dimensional Diffusion Models via Dual-expert Denoising](https://github.com/SakanaAI/AI-Scientist/tree/main/example_papers/dual_expert_denoiser.pdf) 
5. [StyleFusion: Adaptive Multi-style Generation in Character-Level Language Models](https://github.com/SakanaAI/AI-Scientist/blob/main/example_papers/multi_style_adapter.pdf)
6. [Adaptive Learning Rates for Transformers via Q-Learning](https://github.com/SakanaAI/AI-Scientist/tree/main/example_papers/rl_lr_adaptation.pdf)
8. [Unlocking Grokking: A Comparative Study of Weight Initialization Strategies in Transformer Models](https://github.com/SakanaAI/AI-Scientist/tree/main/example_papers/weight_initialization_grokking.pdf)
9. [Grokking Accelerated: Layer-wise Learning Rates for Transformer Generalization](https://github.com/SakanaAI/AI-Scientist/tree/main/example_papers/layerwise_lr_grokking.pdf)
10. [Grokking Through Compression: Unveiling Sudden Generalization via Minimal Description Length](https://github.com/SakanaAI/AI-Scientist/tree/main/example_papers/mdl_grokking_correlation.pdf)
11. [Accelerating Mathematical Insight: Boosting Grokking Through Strategic Data Augmentation](https://github.com/SakanaAI/AI-Scientist/tree/main/example_papers/data_augmentation_grokking.pdf)

**Note**: Caution! This codebase will execute LLM-written code. There are various risks and challenges associated with this autonomy. This includes e.g. the use of potentially dangerous packages, web access, and potential spawning of processes. Use at your own discretion. Please make sure to [containerize](#containerization) and restrict web access appropriately.

<p align="center">
  <a href="https://github.com/SakanaAI/AI-Scientist/blob/main/example_papers/adaptive_dual_scale_denoising/adaptive_dual_scale_denoising.pdf"><img src="https://github.com/SakanaAI/AI-Scientist/blob/main/docs/anim-ai-scientist.gif" alt="Adaptive Dual Scale Denoising" width="80%" />
</p>

## Table of Contents

1. [Requirements](#requirements)
2. [Run AI Scientist Paper Generation Experiments](#run-ai-scientist-paper-generation-experiments)
3. [Getting an LLM Generated Paper Review](#getting-an-llm-generated-paper-review)
4. [Making your own Template](#making-your-own-template)
5. [Template Resources](#template-resources)
6. [Citing The AI Scientist](#citing-the-ai-scientist)
7. [Frequently Asked Questions](#faq)
8. [Containerization](#containerization)

## Requirements

This code was designed for NVIDIA GPUs with CUDA using PyTorch. Support for other GPU architectures may be possible by following [PyTorch guidelines](https://pytorch.org/get-started/locally/). Current templates would likely take an infeasible amount of time on CPU-only machines. All code is designed to be run on Linux, other operating systems will likely require major adjustments.

### Installation

```bash
conda create -n ai_scientist python=3.11
conda activate ai_scientist
# Install pdflatex
sudo apt-get install texlive-full

# Install pypi requirements
pip install -r requirements.txt
```

When installing `texlive-full`, you may need to [hold Enter](https://askubuntu.com/questions/956006/pregenerating-context-markiv-format-this-may-take-some-time-takes-forever).

### Supported Models and API Keys

We support a wide variety of models including open-weight and API-only models. In general, we recommend only using frontier models above the capability of the original GPT-4. To see a full list of models, see [here](https://github.com/SakanaAI/AI-Scientist/blob/main/ai_scientist/llm.py).

#### OpenAI API (GPT-4o, GPT-4o-mini, o1 models)

By default, this uses the `OPENAI_API_KEY` environment variable.

#### Anthropic API (Claude Sonnet 3.5)

By default, this uses the `ANTHROPIC_API_KEY` environment variable.

##### Claude models via Bedrock

For Claude models provided by [Amazon Bedrock](https://aws.amazon.com/bedrock/), please install these additional packages:

```bash
pip install anthropic[bedrock]
```

Next, specify a set of valid [AWS Credentials](https://docs.aws.amazon.com/cli/v1/userguide/cli-configure-envvars.html) and the target [AWS Region](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-regions.html):

Set these environment variables: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION_NAME`.

##### Claude models via Vertex AI

For Claude models provided by [Vertex AI Model Garden](https://cloud.google.com/model-garden?hl=en), please install these additional packages:

```bash
pip install google-cloud-aiplatform
pip install anthropic[vertex]
```

Next, set up a valid authentication for a [Google Cloud project](https://cloud.google.com/vertex-ai/docs/authentication), for example by providing region and project ID like so:

```bash
export CLOUD_ML_REGION="REGION" # for Model Garden call
export ANTHROPIC_VERTEX_PROJECT_ID="PROJECT_ID" # for Model Garden call
export VERTEXAI_LOCATION="REGION" # for Aider/LiteLLM call, as per https://docs.litellm.ai/docs/providers/vertex#set-vertex-project--vertex-location
export VERTEXAI_PROJECT="PROJECT_ID" # for Aider/LiteLLM call as per https://docs.litellm.ai/docs/providers/vertex#set-vertex-project--vertex-location
```

#### DeepSeek API (DeepSeek-Coder-V2)

By default, this uses the `DEEPSEEK_API_KEY` environment variable.

#### OpenRouter API (Llama3.1)

By default, this uses the `OPENROUTER_API_KEY` environment variable.

#### Semantic Scholar API (Literature Search)

Our code can also optionally use a Semantic Scholar API Key (`S2_API_KEY`) for higher throughput [if you have one](https://www.semanticscholar.org/product/api), though in principle it should work without it.

Be sure to provide the key for the model used for your runs, e.g.

```bash
export OPENAI_API_KEY="YOUR KEY HERE"
export S2_API_KEY="YOUR KEY HERE"
```

### Setup NanoGPT

Here, and below, we give instructions for setting up the data and baseline evaluations for each template. You can only run setup steps for templates you are interested in. This is necessary to run on your machine as training times may vary depending on your hardware.

```bash
# Prepare NanoGPT data
python data/enwik8/prepare.py
python data/shakespeare_char/prepare.py
python data/text8/prepare.py
```

#### Create baseline runs (machine dependent)

```bash
# Set up NanoGPT baseline run
# NOTE: YOU MUST FIRST RUN THE PREPARE SCRIPTS ABOVE!
cd templates/nanoGPT && python experiment.py --out_dir run_0 && python plot.py
```

#### Create NanoGPT_lite baseline run. We use this for sanity-checking
```bash
# NOTE: YOU MUST FIRST RUN THE PREPARE SCRIPTS ABOVE!
cd templates/nanoGPT_lite && python experiment.py --out_dir run_0 && python plot.py
```

### Setup 2D Diffusion

```bash
# Set up 2D Diffusion
git clone https://github.com/gregversteeg/NPEET.git
cd NPEET
pip install .
pip install scikit-learn

# Set up 2D Diffusion baseline run
cd templates/2d_diffusion && python experiment.py --out_dir run_0 && python plot.py
```

### Setup Grokking

```bash
# Set up Grokking
pip install einops

# Set up Grokking baseline run
cd templates/grokking && python experiment.py --out_dir run_0 && python plot.py
```


## Run AI Scientist Paper Generation Experiments

**Note:** please ensure the setup steps above are completed.

```bash
conda activate ai_scientist
# Run the paper generation.
python launch_scientist.py --model "gpt-4o-2024-05-13" --experiment nanoGPT_lite --num-ideas 2
python launch_scientist.py --model "claude-3-5-sonnet-20241022" --experiment nanoGPT_lite --num-ideas 2
```

If you have more than 1 GPU, use the `parallel` option to parallelize ideas across multiple GPUs.

## Getting an LLM Generated Paper Review

```python
import openai
from ai_scientist.perform_review import load_paper, perform_review

client = openai.OpenAI()
model = "gpt-4o-2024-05-13"

# Load paper from pdf file (raw text)
paper_txt = load_paper("report.pdf")
# Get the review dict of the review
review = perform_review(
    paper_txt,
    model,
    client,
    num_reflections=5,
    num_fs_examples=1,
    num_reviews_ensemble=5,
    temperature=0.1,
)

# Inspect review results
review["Overall"]  # overall score 1-10
review["Decision"]  # ['Accept', 'Reject']
review["Weaknesses"]  # List of weaknesses (str)
```

To run batch analysis:

```bash
cd review_iclr_bench
python iclr_analysis.py --num_reviews 500  --batch_size 100 --num_fs_examples 1 --num_reflections 5 --temperature 0.1 --num_reviews_ensemble 5
```

## Making your own Template

If there is an area of study you would like **The AI Scientist** to explore, it should be very easy to create your own templates. In general, follow the structure of the existing templates, which consists of:

- `experiment.py` -- This is a single file where the 'meat' of the content is. It takes in an argument for `out_dir`, which is where it should create the folder and save the relevant information from the run.
- `plot.py` -- This should take in the information from the `run` folders and create plots. The code should be clear and easy to edit.
- `prompt.json` -- Put information about your template here.
- `seed_ideas.json` -- Put example ideas here. You can also try to generate ideas without any examples, and then pick the best one or two to put here.
- `latex/template.tex` -- We recommend using our latex folder, but be sure to replace the pre-loaded citations with ones that you would expect to be more relevant.

Please see [this PR](https://github.com/SakanaAI/AI-Scientist/pull/141) for an example of a new template for computer vision.
The key to making new templates work is matching the base file names and output JSONs to the existing format, all else is free to change.
   
## Template Resources

We provide 3 templates, which heavily use code from other repositories, which we credit below. (Normally, we would do this in the files themselves, but it's unclear how this would affect The AI Scientist since it would be visible).

The NanoGPT template used code from [NanoGPT](https://github.com/karpathy/nanoGPT) and this [PR](https://github.com/karpathy/nanoGPT/pull/254).

The 2D Diffusion template used code from [tiny-diffusion](https://github.com/tanelp/tiny-diffusion), [ema-pytorch](https://github.com/lucidrains/ema-pytorch), and [Datasaur](https://www.research.autodesk.com/publications/same-stats-different-graphs/).

The Grokking template used code from [Sea-Snell/grokking](https://github.com/Sea-Snell/grokking) and [danielmamay/grokking](https://github.com/danielmamay/grokking).

We would like to thank the developers of the open-source models and packages for their contributions and for making their work available.

## Citing The AI Scientist

If you use **The AI Scientist** in your research, please cite it as follows:

```
@article{lu2024aiscientist,
  title={The {AI} {S}cientist: Towards Fully Automated Open-Ended Scientific Discovery},
  author={Lu, Chris and Lu, Cong and Lange, Robert Tjarko and Foerster, Jakob and Clune, Jeff and Ha, David},
  journal={arXiv preprint arXiv:2408.06292},
  year={2024}
}
```

## FAQ

We recommend reading our paper in the first instance for any questions you have on The AI Scientist.

### Why am I missing files when running The AI Scientist?
Make sure you have completed all the setup and preparation steps before the main experiment script.

### Why has a PDF or a review not been generated?
The AI Scientist finishes an idea with a success rate that depends on both the template, the base foundation model, and the complexity of the idea. We advise referring to our main paper. The highest success rates are observed with Claude Sonnet 3.5.
Reviews are best done with GPT-4o, all other models have issues with positivity bias or failure to conform to required outputs.

### What is the cost of each idea generated?
Typically less than $15 per paper with Claude Sonnet 3.5. We recommend DeepSeek Coder V2 for a much more cost-effective approach. A good place to look for new models is the [Aider leaderboard](https://aider.chat/docs/leaderboards/).

### How do I change the base conference format associated with the write-ups?
Change the base `template.tex` files contained within each template.

### How do I run The AI Scientist for different subject fields?
Please refer to the instructions for different templates. In this current iteration, this is restricted to ideas that can be expressed in code. However, lifting this restriction would represent exciting future work! :)

### How do I add support for a new foundation model?
You may modify `ai_scientist/llm.py` to add support for a new foundation model. We do not advise any model that is significantly weaker than GPT-4 level for The AI Scientist.

### Why do I need to run the baseline runs myself?
These appear as `run_0` and should be run per machine you execute The AI Scientist on for accurate run-time comparisons due to hardware differences.

### What if I have problems accessing the Semantic Scholar API?
We use the Semantic Scholar API to check ideas for novelty and collect citations for the paper write-up. You may be able to skip these phases in case you don't have an API key or the API is slow to access.

## Containerization

We include a [community-contributed](https://github.com/SakanaAI/AI-Scientist/pull/21) Docker image that may assist with your containerization efforts in `experimental/Dockerfile`.

You can use this image like this:

```bash
# Endpoint Script
docker run -e OPENAI_API_KEY=$OPENAI_API_KEY -v `pwd`/templates:/app/AI-Scientist/templates <AI_SCIENTIST_IMAGE> \
  --model gpt-4o-2024-05-13 \
  --experiment 2d_diffusion \
  --num-ideas 2
```

```bash
# Interactive
docker run -it -e OPENAI_API_KEY=$OPENAI_API_KEY \
  --entrypoint /bin/bash \
  <AI_SCIENTIST_IMAGE>
```
