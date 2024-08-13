---
title: Sentiment Analysis on Encrypted Data Using Fully Homomorphic Encryption
emoji: 🥷💬
colorFrom: yellow
colorTo: yellow
sdk: gradio
sdk_version: 3.2
app_file: app.py
pinned: true
tags: [FHE, PPML, privacy, privacy preserving machine learning, homomorphic encryption, security]
python_version: 3.10.11
---

# Sentiment Analysis With FHE

## Set up the app locally

- First, create a virtual env and activate it:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

- Then, install required packages:

```bash
pip3 install pip --upgrade
pip3 install -U pip wheel setuptools --ignore-installed
pip3 install -r requirements.txt --ignore-installed
```

- (optional) Compile the FHE algorithm:

```bash
python3 compile.py
```

Check it finish well (with a "Done!"). Please note that the actual model initialization and training 
can be found in the [SentimentClassification notebook](SentimentClassification.ipynb) (see below).

### Launch the app locally

- In a terminal:

```bash
source .venv/bin/activate
python3 app.py
```

## Interact with the application

Open the given URL link (search for a line like `Running on local URL:  http://127.0.0.1:8888/` in the 
terminal).

## Train a new model

The notebook [SentimentClassification notebook](SentimentClassification.ipynb) provides a way to 
train a new model. Be aware that the data needs to be downloaded beforehand using the 
[download_data.sh](download_data.sh) file (which requires Kaggle CLI). 
Alternatively, the dataset can be downloaded manually at 
https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

```bash
bash download_data.sh
```
