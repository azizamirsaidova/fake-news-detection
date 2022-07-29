# Linguistic Evaluation of Machine-Generated "Real" and "Fake" News

## Project Objectives
Generate “true” and “fake” news using pre-trained transfomer model GPT-Neo 1.3B to perform text generation using LIAR dataset including 12.8 K short stataments.

Main Goals:
1. To assess whether this approach is an effective way to create compa- rable machine-generated real and fake news
2. To ascertain if there are any detectable stylometric or linguistic differences between real and fake news generated in this way.

## Repo Structure

The structure of the repository looks like the following.

```bash
.
├── liar-dataset     # Data
├── notebooks        # Models saved
├── output           # Text output
├── paper_report     # Report
├── src              # Source code
```

## Data

LIAR dataset consists of 12.8K human-labeled short statements occurring in various contexts collected from politifact.com. 
Based on the evaluation and justification of a www.politifact.com editor, each statement is labeled for truthfulness with one of six fine-grained ratings: pants on fire, false, barely true, half true, mostly true, and true. For the purposes of this project, we are only interested in the sentences labeled “false” and “true.” 

## Methods

There are three main parts of the project: text generation, conducting linguistic feature analysis and fake new detection 

### 1. Text Generation

Text generation is conducted using GPT-Neo 1.3B, a pre-trained transformer model designed using EleutherAI’s replication of the GPT-3 architecture, GPT-Neo 1.3B was trained on the Pile dataset for 380B tokens over 362,000 steps as a masked autoregressive language model.

From the LIAR training dataset, we extracted the first 1000 true and false statements. 

Paremeters were 200 of max length, 0.9,temperature, and top k sampling of 50. This process yielded a paragraph of text for each statement, for a total of 1000 “fake news” paragraphs and 1000 “real news” paragraphs.

### 2. Lingustic Feature Analysis

Following linguistic analysis are created:

1. Named Entity Recognition
2. Referential Words
3. Zipf Distribution


### 3. Fake & True News Classification

Count Vectorizer and TF-IDF of the data is obtained which provides output in a sparse matrix representing the text. Then, following classification models are trained on machine generated true and fake data.

* Passive Aggressive Classifier
* Logistic Regression
* Naive Bayes
* Decision Tree Classifier
* BERT

For further details, refer to paper [report.](https://github.com/azizamirsaidova/fake-news-detection/blob/main/paper_report/Final%20Paper.pdf)

### References

1. Wang, W. Y. (2017). " liar, liar pants on fire": A new benchmark dataset for fake news detection. arXiv preprint arXiv:1705.00648.

2. Zellers, R., Holtzman, A., Rashkin, H., Bisk, Y., Farhadi, A., Roesner, F., & Choi, Y. (2019). Defending against neural fake news. Advances in neural information processing systems, 32. 
