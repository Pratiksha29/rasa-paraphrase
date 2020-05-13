# [WIP]

## RASA Paraphrase

A CLI port of [Paraphrasing for NLU Data Augmentation](https://forum.rasa.com/t/paraphrasing-for-nlu-data-augmentation-experimental/27744/1)

## Install:

**Model Dependencies**

[Download Model Dependencies](https://paraphrase-model.nyc3.digitaloceanspaces.com/model_deps.zip)


**Packages**

```
poetry install
```

***Try it out:***

todo: invoke

**One phrase at a time**

```
python3 -m app --input "What time is it?" --num_samples 4
```

**Reading from a RASA NLU markdown file**

```
python3 -m app --nlu path/to/nlu.md
```





> Note: `boto3` may require a global installation:
```
pip3 install --user boto3
```
