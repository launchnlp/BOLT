# BOLT
This is the implementation of ACL 2023 paper [BOLT: Fast Energy-based Controlled Text Generation with Tunable Biases](https://arxiv.org/abs/2305.12018).

## Environment Setup
```
cd ./transformers
pip install -e .
cd -
```

## Download Discriminators
You can download the discriminators from [here](https://drive.google.com/file/d/1G1ptRin1US6usmcq5bI_iO4uDs4KCpAl/view?usp=share_link) and put them under `./checkpoints`. These roberta-based discriminators are trained on the [Yelp dataset](https://huggingface.co/datasets/yelp_polarity) and the [Jigsaw dataset](https://huggingface.co/datasets/jigsaw_toxicity_pred), whose embeddings are replaced with the GPT2-large embeddings.

## Sentiment Control
```
SENTIMENT=pos
# SENTIMENT=neg
SEQLEN=20
python sentiment_generate_with_bias.py $SEQLEN $SENTIMENT
```
The generated sentences will be saved in `./sentiment/sentiment/`.

## Toxicity Avoidance
```
SEQLEN=20
python detoxic_generate_with_bias.py $SEQLEN
```
The generated sentences will be saved in `./detoxic/detoxic/`.

## Keyword-guided Topic Control
```
SEQLEN=20
TOPIC=computer
python keywords_generate_with_bias.py $SEQLEN $TOPIC
```
The generated sentences will be saved in `./keywords/keywords/`.
