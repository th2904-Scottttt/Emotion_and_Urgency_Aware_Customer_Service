# Emotion and Urgency Aware Customer Service

This research prototype explores how a customer-service assistant can adapt a reply to a customer's emotional state and the urgency of their request. It includes message classifiers, a response-model training pipeline, and a retrieval-augmented generation (RAG) demo.

## Reproduce the classifier comparison

From the repository root, run the following commands with Python 3. The script reads two committed JSON files and uses only the Python standard library.

```sh
python3 scripts/compare_classifiers.py
```

The two classifiers agree on urgency for 89 of the first 100 paired messages, emotion for 97, and both labels for 86. These figures measure agreement between GPT-4o mini and Qwen. The comparison has no human reference labels. Distribution tables use all rows in each input. Agreement uses the first 100 paired rows because the saved files contain 100 OpenAI predictions and 2,000 Qwen predictions.

To compare a new pair of prediction files, pass `--openai-file PATH` and `--qwen-file PATH`. Supply files whose rows follow the same message order.

## Repository guide

| Directory | Contents |
| --- | --- |
| [message_classification 2](message_classification%202/) | Scripts to classify MultiDOGO utterances by urgency and emotion. |
| [scripts](scripts/) | Command to compare saved classifier predictions. |
| [results/classifier_agreement](results/classifier_agreement/) | Saved GPT-4o mini and Qwen predictions used in the comparison. |
| [CAI_Pipline](CAI_Pipline/) | A constitution for customer-service replies, notebooks for generating draft and revised replies, supervised fine-tuning and preference training, and saved model adapters. |
| [Rag+Tune](Rag%2BTune/) | Initial domain fine-tuning and a RAG demo that retrieves similar conversations and generates replies with a Qwen model. |

## Current scope

The RAG demo takes a question and supplied emotion and urgency labels. Its 18 example scenarios exercise generation with those labels. Connecting the classifiers to the demo and evaluating responses against merchant policies are open steps. The training and demo notebooks reference dataset and adapter paths from the original environment.
