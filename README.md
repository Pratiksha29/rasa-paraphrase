# [WIP]

# RASA Paraphrase

A CLI port of [Paraphrasing for NLU Data Augmentation](https://forum.rasa.com/t/paraphrasing-for-nlu-data-augmentation-experimental/27744/1)

### Getting Started

1. [Clone](https://github.com/crodriguez1a/rasa-paraphrase.git) this repository
  ```
  git clone https://github.com/crodriguez1a/rasa-paraphrase.git
  ```

1. [Download](https://paraphrase-model.nyc3.digitaloceanspaces.com/model_deps.zip) and expand model dependencies

1. Set environment variable:
```
export MODEL_PATH=path/2/model_deps/
```


### Install Packages

```
poetry install
```

> Note: *boto3* may require a global installation: `pip3 install --user boto3`

### Try it out

#### One phrase at a time:
```
python3 -m app --input "What time is it?" --num_samples 3
```

Produces:
```
and then what time?
when's the time?
what time is it?'
```

With Similarity:
```
python3 -m app --input "What time is it?" --num_samples 8 --similarity .98
```

Produces:
```
Similar:
-----------------------  --------
what hour is it?         0.998905
how much time is it?     0.993163
and when is it?          0.991079
what time did it start?  0.988266
-----------------------  --------

Less Similar:
--------------------------------  --------
when is it?                       0.97795
and when? - when the time comes?  0.974981
what's up, what's up?             0.971918
what time do we get there?        0.968454
--------------------------------  --------
```

#### Optionally, output to CSV:
```
python3 -m app --input "Hi" --num_samples 8 --similarity .98 --csv my_file.csv
```
---

#### Reading from a RASA NLU markdown file:

```
python3 -m app --nlu path/to/nlu.md
```

Outputs mardown with a balanced number of utterances for each intent.

Original:
```
## intent: chitchat/what_time_is_it
- what's the time

## intent: how_should_i_spend_time
- how should i spend my time
- How should I be spending my time?
- I do not know how to spend my time?
```

Output:
```
## intent: chitchat/what_time_is_it
- what's the time
- so what's the time?
- what time?

## intent: how_should_i_spend_time
- how should i spend my time
- How should I be spending my time?
- I do not know how to spend my time?
```
