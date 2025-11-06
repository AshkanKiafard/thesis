import json
import os
from functools import cache

import numpy as np

import torch
from datasets import Dataset, load_from_disk
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, \
    losses, SimilarityFunction
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

import traverse_strategies as ts
from utils import get_concept, load_graph, traverse_graph


@cache
def path_length_to_similarity(length, max_length=7, max_connected_val=0.99):
    if length == 0 or length > max_length:
        return 0.0
    return 1 - (np.log(length) / np.log(max_length) * max_connected_val)


bfs_cache = {}


def create_dataset(data):
    global bfs_cache

    sentence1 = []
    sentence2 = []
    score = []

    for i, item in enumerate(data):
        print(f"{i + 1}/{len(data)}")
        cause = get_concept(item, 0)
        effect = get_concept(item, 1)

        if (cause, effect) in bfs_cache:
            path = bfs_cache[(cause, effect)]
        else:
            path, _ = traverse_graph(causal_graph, cause, effect, None, ts.bfs_traverse)
            bfs_cache[(cause, effect)] = path

        path_len = len(path)
        similarity = path_length_to_similarity(path_len)

        sentence1.append(cause)
        sentence2.append(effect)
        score.append(similarity)

    dataset = Dataset.from_dict({
        "sentence1": sentence1,
        "sentence2": sentence2,
        "score": score
    })

    return dataset


causal_graph = load_graph("data/graphs/causenet-precision.jsonl")
print("Causal graph loaded.")

with open("data/datasets/msmarco_train.json") as f:
    train_data = json.load(f)

print("Train data loaded.")

with open("data/datasets/msmarco_valid.json") as f:
    valid_data = json.load(f)
print("Validation data loaded.")

with open("data/datasets/msmarco_test.json") as f:
    test_data = json.load(f)
print("Test data loaded.")

train_dataset_path = "data/datasets/train"
if os.path.exists(train_dataset_path):
    train_dataset = load_from_disk(train_dataset_path)
    print("Training dataset loaded.")
else:
    train_dataset = create_dataset(train_data)
    train_dataset.save_to_disk(train_dataset_path)
    print("Training dataset created.")

valid_dataset_path = "data/datasets/valid"
if os.path.exists(valid_dataset_path):
    valid_dataset = load_from_disk(valid_dataset_path)
    print("Validation dataset loaded.")
else:
    valid_dataset = create_dataset(valid_data)
    valid_dataset.save_to_disk(valid_dataset_path)
    print("Validation dataset created and saved.")

test_dataset_path = "data/datasets/test"
if os.path.exists(test_dataset_path):
    test_dataset = load_from_disk(test_dataset_path)
    print("Test dataset loaded.")
else:
    test_dataset = create_dataset(test_data)
    test_dataset.save_to_disk(test_dataset_path)
    print("Test dataset created and saved.")

model_name = "multi-qa-mpnet-base-cos-v1"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Loading model {model_name} on {device}...")
model = SentenceTransformer(model_name, device=device)

loss = losses.CoSENTLoss(model)

dev_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=valid_dataset["sentence1"], # cause
    sentences2=valid_dataset["sentence2"], # effect
    scores=valid_dataset["score"], # similarity
    main_similarity=SimilarityFunction.COSINE,
    name="dev-eval",
)
print(dev_evaluator(model))

# TODO use lightning
args = SentenceTransformerTrainingArguments(
    output_dir=f"data/models/sentence-transformers/{model_name}_fine-tuned",
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=1.5e-5,
    warmup_ratio=0.1,
    fp16=False,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    logging_steps=50
)

trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    loss=loss,
    evaluator=dev_evaluator,
    eval_dataset=valid_dataset,
)
trainer.train()

test_evaluator = EmbeddingSimilarityEvaluator(
    sentences1=test_dataset["sentence1"],
    sentences2=test_dataset["sentence2"],
    scores=test_dataset["score"],
    main_similarity=SimilarityFunction.COSINE,
    name="test-eval",
)
print(test_evaluator(model))

model.save_pretrained(f"data/models/sentence-transformers/{model_name}_fine-tuned")
print("Model saved.")
