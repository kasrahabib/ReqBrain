# **Evaluations: Automated NLP Metrics**
This directory contains all the scripts and visualizations used to evaluate the models using automated NLP metrics. The evaluation results in a dataset that includes pairs of human-authored requirements and their respective generated counterparts from each model, facilitating pairwise comparisons and visualizations for the paper.

---

## **Key Files and Directories**

#### [`automated_nlp_metrics_evaluation.ipynb`](./automated_nlp_metrics_evaluation.ipynb)  
This notebook evaluates model-generated requirements using 5 automated NLP metrics for ***Human Alignment<sub>HA</sub>*** variable:

BERTScore precision (P), recall (R), and F1 are computed per instance and then averaged across the evaluation set; therefore, the reported F1 does not equal the harmonic mean of P and R.  The top five rows present the fine-tuned models (RQ1.1).   ChatGPT-4o is included to compare the selected fine-tuned model (ReqBrain) with an untuned general LLM (RQ1.2). Best scores are in **bold**.

<table>
  <tr>
    <th rowspan="2"><b>Models</b></th>
    <th rowspan="2"><b>BLEU</b></th>
    <th colspan="3"><b>ROUGE</b></th>
    <th colspan="2"><b>TER</b></th>
    <th colspan="3"><b>BERT Score</b></th>
    <th rowspan="2"><b>FRUGAL Score</b></th>
  </tr>
  <tr>
    <th><b>Rouge-1</b></th>
    <th><b>Rouge-2</b></th>
    <th><b>Rouge-L</b></th>
    <th><b>#Edits</b></th>
    <th><b>Score</b></th>
    <th><b>P</b></th>
    <th><b>R</b></th>
    <th><b>F1</b></th>
  </tr>
  <tr>
    <td><b>Zephyr-7b-beta</b></td>
    <td><b>12.2</b></td>
    <td><b>0.42</b></td>
    <td><b>0.20</b></td>
    <td><b>0.36</b></td>
    <td><b>918</b></td>
    <td><b>108</b></td>
    <td><b>0.89</b></td>
    <td><b>0.89</b></td>
    <td><b>0.89</b></td>
    <td><b>0.91</b></td>
  </tr>
  <tr>
    <td><b>Mistral-7b-Instruct</b></td>
    <td>3.16</td>
    <td>0.24</td>
    <td>0.10</td>
    <td>0.20</td>
    <td>4109</td>
    <td>487</td>
    <td>0.84</td>
    <td><b>0.89</b></td>
    <td>0.86</td>
    <td>0.88</td>
  </tr>
  <tr>
    <td><b>Falcon-7b</b></td>
    <td>2.26</td>
    <td>0.23</td>
    <td>0.07</td>
    <td>0.19</td>
    <td>2453</td>
    <td>290</td>
    <td>0.80</td>
    <td>0.82</td>
    <td>0.85</td>
    <td>0.88</td>
  </tr>
  <tr>
    <td><b>Falcon-7b-instruct</b></td>
    <td>3.04</td>
    <td>0.28</td>
    <td>0.11</td>
    <td>0.23</td>
    <td>2561</td>
    <td>303</td>
    <td>0.85</td>
    <td>0.88</td>
    <td>0.86</td>
    <td>0.88</td>
  </tr>
  <tr>
    <td><b>Llama-2-7b-chat-hf</b></td>
    <td>2.34</td>
    <td>0.23</td>
    <td>0.09</td>
    <td>0.19</td>
    <td>3933</td>
    <td>466</td>
    <td>0.81</td>
    <td>0.85</td>
    <td>0.85</td>
    <td>0.88</td>
  </tr>
  <tr>
    <td><b>ChatGPT-4o (untuned)</b></td>
    <td>2.02</td>
    <td>0.15</td>
    <td>0.07</td>
    <td>0.12</td>
    <td>6424</td>
    <td>762</td>
    <td>0.81</td>
    <td>0.88</td>
    <td>0.84</td>
    <td>0.86</td>
  </tr>
</table>


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











