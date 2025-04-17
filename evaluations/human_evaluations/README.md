# **Evaluations: Human Evaluations**

This directory contains human evaluation data, demographic information, and statistical analyses related to the research questions outlined in the paper.



---

## **Key Files and Directories**

### **Human Evaluation Directories:**  
**`p1_human_evaluation`, `p2_human_evaluation`, `p3_human_evaluation`, `p4_human_evaluation`**  
Each of these directories contains evaluation data collected from individual participants. The structure and contents of each directory are identical:

- **[`Evaluation Package Guide.docx`](./p1_human_evaluation/Evaluation%20Package%20Guide.docx)**  
  - A guide provided to human evaluators, explaining the tasks and evaluation process.  

- **[`demographic_questions.xlsx`](./p1_human_evaluation/demographic_questions.xlsx)**  
  - Demographic information collected from the participant.

- **Task-Specific Results:**  
  - **Task D:**  
    - [`task_d.xlsx`](./p1_human_evaluation/task_d/task_d.xlsx) — Evaluation results for Task D.  
    - [`to_view_evaluation_set_for_missing_task_prompt_human_written_requirements_in_prompt_generated_requirements.xlsx`](./p1_human_evaluation/task_d/to_view_evaluation_set_for_missing_task_prompt_human_written_requirements_in_prompt_generated_requirements.xlsx) — The context and prompts provided to evaluators, including human-written and generated requirements.  
  - **Tasks B and C:**  
    - [`task_b.xlsx`](./p1_human_evaluation/tasks_b_and_c/task_b.xlsx) — Evaluation results for Task B.  
    - [`task_c.xlsx`](./p1_human_evaluation/tasks_b_and_c/task_c.xlsx) — Evaluation results for Task C.  

Since all human evaluation directories (`p1_human_evaluation`, `p2_human_evaluation`, etc.) follow the same structure, you can explore any directory for similar files and task outputs.

---
### **An A-Priori Power Analysis:**  
**[`a_priori_power_analysis`](./a_priori_power_analysis.ipynb)**  
- This notebook runs tests for calculating statistical power analysis.

### **Participant Demographics:**  
**[`participants_demo_graphics.ipynb`](./participants_demo_graphics.ipynb)**  
- This notebook provides visualizations and summaries of participant demographic information collected during the human evaluations.


### **Statistical Analysis Notebooks:**

- **[`task_b_stats_test_reqbrain_vs_untuned_baseline.ipynb`](./task_b_stats_test_reqbrain_vs_untuned_baseline.ipynb)**  
  - Runs statistical analysis on Task B data to address Research Questions 2 and 4 in the paper.

- **[`task_c_stats_test_reqbrain_vs_human.ipynb`](./task_c_stats_test_reqbrain_vs_human.ipynb)**  
  - Runs statistical analysis on Task C data to address Research Questions 3 and 5 in the paper.

- **[`task_d_stats_test_additional_requirements_generation.ipynb`](./task_d_stats_test_additional_requirements_generation.ipynb)**  
  - Runs statistical analysis on Task D data to address Research Question 6 in the paper.

---

### **Raw Human Ratings:**  
**[`raw_human_rating_plots_not/`](./raw_human_rating_plots_not/)**  
This directory contains raw plots of human ratings for each task, which were not shown in the paper but are available for further analysis:

- **`task_b_raw_ratings.png`** — Raw human ratings for Task B.  
- **`task_c_raw_ratings.png`** — Raw human ratings for Task C.  
- **`task_d_raw_ratings.png`** — Raw human ratings for Task D.  