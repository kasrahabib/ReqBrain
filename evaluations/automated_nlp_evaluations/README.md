# **Evaluations: Automated NLP Metrics**
This directory contains all the scripts and visualizations used to evaluate the models using **FRUGALScore** and **BERTScore**. The evaluation results in a dataset that includes pairs of human-authored requirements and their respective generated counterparts from each model, facilitating pairwise comparisons and visualizations for the paper.

---

## **Key Files and Directories**

#### [`automated_nlp_metrics_evaluation.ipynb`](./automated_nlp_metrics_evaluation.ipynb)  
- This notebook evaluates model-generated requirements using **FRUGALScore** and **BERTScore**.

#### [`paper_visualisations.ipynb`](./paper_visualisations.ipynb)  
- Generates all visualizations and evaluation outputs needed for the paper.

#### [`evaluation_set_for_nlp_metrics/`](./evaluation_set_for_nlp_metrics/)  
Contains scripts and prediction outputs for generating requirements using different models:  
 
- **[`falcon_instruct_prediction.ipynb`](./evaluation_set_for_nlp_metrics/falcon_instruct_prediction.ipynb):** Generates requirements using the Falcon Instruct model.  
- **[`falcon_prediction.ipynb`](./evaluation_set_for_nlp_metrics/falcon_prediction.ipynb):** Generates requirements using Falcon Base.  
- **[`llama_chat_hf_prediction.ipynb`](./evaluation_set_for_nlp_metrics/llama_chat_hf_prediction.ipynb):** Generates requirements using LLaMA.  
- **[`mistralai-instruct_prediction.ipynb`](./evaluation_set_for_nlp_metrics/mistralai-instruct_prediction.ipynb):** Generates requirements using MistralAI.  
- **[`zephyr_prediction.ipynb`](./evaluation_set_for_nlp_metrics/zephyr_prediction.ipynb):** Generates requirements using Zephyr.  
- **[`chatgpt_4o_latest.ipynb`](./evaluation_set_for_nlp_metrics/chatgpt_4o_latest.ipynb):** Generates requirements using ChatGPT-4o. 
- **[`models_prediction_dataset/`](./evaluation_set_for_nlp_metrics/models_prediction_dataset/):** Contains the combined dataset of original requirements and generated predictions used for pairwise evaluation.

---

### **Paper Visualizations and Supporting Files:**  
- **[`spider_chart.pdf`](./spider_chart.pdf):** Visualization of model performance across evaluation metrics.  
- **[`train_eval_set.pdf`](./train_eval_set.pdf):** Overview of the training and evaluation sets.











