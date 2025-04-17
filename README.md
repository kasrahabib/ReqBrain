# 📄 **Usage Restriction Prior to Paper Publication**

As part of the review process for our academic paper titled:  
<u>ReqBrain: Task-Specific Instruction Tuning of LLMs for AI-Assisted Requirements Generation</u>,  
this repository is **strictly closed-source** and provided **solely as a replication package**.

**Do not use, modify, copy, or redistribute any part of this repository** until the paper is officially published.  
Access is granted for **review purposes only**, and **no use beyond replication for review is permitted**.

After publication, this repository will be **archived on [Zenodo](https://zenodo.org/)** to ensure reproducibility and prevent versioning conflicts.

Please refer to the LICENSE file for full legal terms.

# **LICENSE**

**Copyright © Mohammad Kasra Habib, 2025. All rights reserved.**

This repository is provided solely for replication purposes during the peer review process for the academic paper titled  
"ReqBrain: Task-Specific Instruction Tuning of LLMs for AI-Assisted Requirements Generation."

Use, reproduction, modification, or redistribution of any part of this repository is **strictly prohibited** until the paper is officially published.

No license is granted for use, and this repository is not open-source in any form during the review period.  
Upon publication, an open license may be applied, and the repository will be archived permanently on [Zenodo](https://zenodo.org/).



# **ReqBrain**
**Req**uirements **Brain**, an open-source, is a task-specific instruction-tuned language model designed to generate well-structured ISO 29148 compliance software requirements. It provides automation support to requirements engineers for the elicitation and documentation of requirements. Requirements engineers can interact with ReqBrain in a chat format to generate new requirements, generate requirments and classify on the fly, and turn bug descriptions into well-structured. Additionally, they can provide an existing set of requirements to identify missing ones, classify requirements by type, or combine it with RAG to process huge bulks of proprietary textual data and elicit in context well-structured requirements. 


---

## **Table of Contents**
- [Overview](#overview)
- [Running Demo](#running-demo)
- [Training the Model](#training-the-model)
- [Models Availability](#models-availability)
- [ReqBrain Evaluation](#reqbrain-evaluation)
- [Instruct Dataset](#instruct-dataset)
  - [Loading the Dataset Properly](#loading-the-dataset-properly)

---

## **Overview**
The repository provides:
- **Datasets** for training and evaluation  
- **Training Scripts** for training ReqBrain  
- **Evaluation data and scripts** to assess performance  
- **A Jupyter Notebook demo** showcasing model usage 
- **ReqBrain Candidates:** Link to trained models and their weights  

---

## **Running Demo**  

You can explore the basic functionality of ReqBrain through the interactive Jupyter Notebook demo: 
- **Model Download:** It automatically downloads the model weights from our Hugging Face repository.
- **Step-by-Step Guidance:**  The notebook guides you through the process and explains the key steps involved.
- **GPU Requirement:** A GPU with a minimum of 32GB GPU memory is required to ensure smooth execution.



Download the [reqbrain_demo.ipynb](./reqbrain_demo.ipynb) script from the repo’s root directory, then launch it using:

```bash

jupyter notebook reqbrain_demo.ipynb

```

---

## **Training the model**  
The [training_scripts](./training_scripts) directory provides scripts for fine-tuning ReqBrain, organized in Jupyter Notebooks:

- **Model-Specific Scripts:** Each notebook is named after the model it fine-tunes, making it easy to locate.
- **Step-by-Step Guidance:** Users are guided through fine-tuning using comments.
- **GPU Requirement:** A GPU with a minimum of 32GB GPU memory is required to ensure smooth execution.

---

## **Models Availability**
The five trained models are quite large. For easy access, seamless downloading, further tuning, and integration with other Hugging Face tools, they are hosted on ReqBrain's Hugging Face page. The link to each trained model is provided below:

| Training Script | Name on HuggingFace | HuggingFace Model Link  |
|-----------------|------------------------|----------|
| `training_scripts/train_falcon-base.ipynb`         | ReqBrain-falcon-7b| [https://huggingface.co/kasrahabib/ReqBrain-falcon-7b](https://huggingface.co/kasrahabib/ReqBrain-falcon-7b) |
| `training_scripts/train_falcon-instruct.ipynb`     | ReqBrain-falcon-7b-instruct| [https://huggingface.co/kasrahabib/ReqBrain-falcon-7b-instruct](https://huggingface.co/kasrahabib/ReqBrain-falcon-7b-instruct) |
| `training_scripts/train_llama2.ipynb`               | ReqBrain-Llama-2-7b-chat-hf| [https://huggingface.co/kasrahabib/ReqBrain-Llama-2-7b-chat-hf](https://huggingface.co/kasrahabib/ReqBrain-Llama-2-7b-chat-hf) |
| `training_scripts/train_mistralai.ipynb` 	       | ReqBrain-Mistral-7B-Instruct-v0.2| [https://huggingface.co/kasrahabib/ReqBrain-Mistral-7B-Instruct-v0.2](https://huggingface.co/kasrahabib/ReqBrain-Mistral-7B-Instruct-v0.2) |
| `training_scripts/train_zephyr.ipynb` 	           | <sup><span style="font-size: 1em; font-weight: bold;">※</span></sup>*ReqBrain-zephyr-7b-beta*| [https://huggingface.co/kasrahabib/ReqBrain-zephyr-7b-beta](https://huggingface.co/kasrahabib/ReqBrain-zephyr-7b-beta) |

<sup><span style="font-size: 1em; font-weight: bold;">※</span></sup> Evaluation results indicate that this model achieves the highest performance across all tasks.

---

## **ReqBrain Evaluation**  
All evaluation data and scripts can be found in the `evaluations` directory. ReqBrain is evaluated through:

- **Automated NLP Metrics:** Located in [evaluations/automated_nlp_evaluations/](./evaluations/automated_nlp_evaluations/)  
- **Human Evaluators:** Located in [evaluations/human_evaluations/](evaluations/human_evaluations/)

Further details are provided inside each of the subdirectories: `/evaluations/automated_nlp_evaluations/` and `evaluations/human_evaluations/`.


---

## **Instruct Dataset**
The [dataset](./dataset) directory contains the instruct dataset. 
- **Training Set** is located in `dataset/train/`
- **Evaluation Set:** is located under `dataset/test/`


### **Loading the Dataset Properly**

The dataset is structured to be compatible and directly used with *Falcon*, *LLaMA*, *Zephyr*, and other LLMs sharing a similar input format.  

#### **Step 1: Install the Required Library**

Make sure you have the `datasets` library installed. You can do this using pip:

```bash
pip install datasets
```

#### **Step 2: Select the Correct Type**
We provide a helper function, ```get_dataset_by_model_format()```, to easily filter the dataset based on the target model format.

```python

import datasets

def get_dataset_by_model_format(dataset, split, ds_format):
    return dataset[split].filter(lambda example: example['ds_format'] == ds_format)

```

#### **Step 3: Load Train and Test Splits**

```python

# Load the dataset
dataset_path = './path_to_the_dataset/dataset'
instruct_dataset = datasets.load_from_disk(dataset_path)

# Filter the dataset by the desired format (e.g., 'falcon')
dataset = get_dataset_by_model_format(instruct_dataset, split='train', ds_format='falcon')
dataset_test = get_dataset_by_model_format(instruct_dataset, split='test', ds_format='falcon')

```

Replace ```falcon``` with ```llama``` or ```zephyr``` as needed to retrieve the dataset for the appropriate model.