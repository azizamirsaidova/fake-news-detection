# Title of the Project

## Table of Content

## What's This Project About?

### Repo Structure

The structure of the repository looks like the following. The `counts/`, `data/`
and `models/` are filtered out by the `.gitignore` because of the size of the file.

```bash
.
├── counts
├── data        # Data
├── models      # Models saved
├── outputs     # Output of the programs
├── scripts     # Setups, running commends, etc
└── src         # Source code
```

## The Data

## How to Run?

The running environment are encapsulated in the docker image. Follow the steps below:

1. Prepare the repository with the structure in
  [Repo Structure](#repo-structure) section.
2. Build the docker image by running `sudo ./scripts/build_docker.sh` in the
  `matterhorn/` directory.
3. Run the docker image by using `sudo ./scripts/run_docker.sh` in the `matterhorn/`
  directory.

The above steps will create a docker image and run the docker image with
`matterhorn/` repository mounted to the docker volume. To learn how to
customize the Docker image, checkout:
* [Customizing docker images](https://docs.nvidia.com/ngc/ngc-catalog-user-guide/index.html#custcontdockerfile)
* [How to use Nvidia NGC PyTorch Docker images](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch)

## Research Topics

### 1.

To generate the result, under `matterhorn/`, please run
`./scripts/generate_iterated_results.sh`.

### 2. Using BERT to Calculate Legal-Level

**Theory:**

Our theory is that the decision made by the judge could be influenced by the formalness
or legalese level of the narrative.

**Method:**

We used Pseudo-perplexity introduced by Salazar et al. to calculate the closeness of
a sentence to the pre-trained language models without doing transfer learning. We
calculate the closeness of each narrative with the BERT uncased model and the
[legal BERT model](https://huggingface.co/nlpaueb/legal-bert-base-uncased). The legal
BERT model is trained on top of legal documents. The legal score minus the base score
can finally represent the legalese level.

The Python Classes are in `src/legal_bert.py`, to calculate the score in different
settings:

```python
example_model = Scores(base_model_path, legal_model_path, level, device, length=length)
"""
base_model_path: huggingface hosted bert-base-model
legal_model_path: huggingface hosted legal-base-model
level:
    "truncate": truncate the narrative into length of 256 (around 5 hours on
    RTX3090)
    "sentence": chunk the narrative into customized length and calculate the
    narrative by averaging the mode (filter out outliers) (around 2 hours for
    length of 128 on RTX3090, the shorter the length, the faster it runs)
    "narrative": truncate the narrative by length of 512 and use CPU to calculate
    in trade of memory. (around 78 hours on AMD 5950X)
device: "cpu" or "cuda"
length: any number (e.g. 64, 128)
"""
for k, v in enumerate(tqdm(example_model)):
  case_number, text, legal_score, base_score, legal_minus_base = v
```

To run the program, under docker environment, use command `python src/legal_bert.py`.
The output are in the `outputs/`:

```
truncated_scores.csv
sentence_scores_(length).csv
narrative_scores.csv
```

### 3. BERTTopic

## How to Test?

## License
