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
| [CAI_Pipline](CAI_Pipline/) | Saved CAI input data, generated preference pairs, and model adapters. |
| [Rag+Tune](Rag%2BTune/) | Initial adapter and a RAG demo that retrieves similar conversations and generates replies with a Qwen model. |

## Training and preference analysis

| Notebook | Inputs and outputs |
| --- | --- |
| [01 Domain fine-tuning](notebooks/01_domain_finetuning.ipynb) | The Colab notebook expects a CUDA GPU and five dialogue CSVs listed in `DATASET_PATHS` (airport ticket, fast food, finance, media, and software). Set those paths and `OUTPUT_DIR` for your environment. It saves the initial Qwen LoRA adapter. |
| [02 Constitutional AI training](notebooks/02_constitutional_ai_training.ipynb) | Reads the [constitution](configs/constitution.json), `CAI_Pipline/results_qwen.json`, and the initial adapter in `CAI_Pipline/qwen2-1.5b-finetuned/`. Builds draft/revision pairs, trains a supervised adapter, then trains a preference adapter using GPT-4o mini judgments. |
| [03 Preference analysis](notebooks/03_preference_analysis.ipynb) | Loads the final DPO adapter and `CAI_Pipline/cai_pairs.jsonl`. Calls the judge to reconstruct preferences and reports revision selections and the tone/urgency text-similarity heuristic. |

Start kernels for notebooks 02 and 03 inside the cloned repository. Both resolve files relative to `CAI_Pipline/`. Set `OPENAI_API_KEY` in the kernel environment before running judge cells. Rerunning training writes to the pair and adapter paths used by the notebook.

## Run the RAG demo

Install the inference dependencies in a Python 3.11 virtual environment, then run one query from the repository root.

```sh
python -m pip install -r requirements-rag.txt
python scripts/run_rag_demo.py --question "How do I request a return label?" --emotion neutral --urgency low
```

The command uses the checked-in final DPO adapter and two synthetic example conversations. The first run downloads the Qwen base model and MiniLM embeddings. Pass `--model-path PATH` for another compatible adapter and `--csv FILE [FILE ...]` for your own conversation CSVs. Include the columns `conversationId`, `authorRole`, and `utterance`, in dialogue order. Retrieval uses local documents by default. To allow the notebook’s keyword and similarity fallback to send the question to Tavily, supply `--web-search` and set `TAVILY_API_KEY`.

## Current scope

The RAG demo takes a question and supplied emotion and urgency labels. Its 18 example scenarios exercise generation with those labels. Connecting the classifiers to the demo and evaluating responses against merchant policies are open steps. The training and demo notebooks reference dataset and adapter paths from the original environment.
