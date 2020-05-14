# [WIP]

## RASA Paraphrase

A CLI port of [Paraphrasing for NLU Data Augmentation](https://forum.rasa.com/t/paraphrasing-for-nlu-data-augmentation-experimental/27744/1)

### Install:

**Install Model Dependencies**

1. [Download](https://paraphrase-model.nyc3.digitaloceanspaces.com/model_deps.zip) and expand model dependencies

1. Set environment variable:
```
export MODEL_PATH=path/2/model_deps/
```


**Install Packages**

```
poetry install
```

### Try it out:

One phrase at a time:
```
python3 -m app --input "What time is it?" --num_samples 4
```

Produces:
```
and then what time?
when's the time?
what is this?
what time is it?'
```

---

Reading from a RASA NLU markdown file:

```
python3 -m app --nlu path/to/nlu.md
```

Outputs mardown with a balanced number of utterances for each intent.





> Note: `boto3` may require a global installation:
```
pip3 install --user boto3
```
