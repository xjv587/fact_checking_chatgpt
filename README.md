# Fact-Checking Outputs from ChatGPT
This project focuses on analyzing and verifying the factual accuracy of outputs generated by large language models (LLMs) like ChatGPT. The goal is to develop and evaluate methods for detecting non-factual content by comparing model-generated claims against Wikipedia data.

## Key Components:
- Word Overlap Method:

  - Implemented a bag-of-words overlap approach to predict whether a fact is supported by a given passage.

  - Explored various preprocessing techniques (tokenization, stemming, stopword removal) and similarity metrics (cosine similarity, Jaccard similarity) to optimize accuracy.

  - Achieved 78% accuracy on the dataset by tuning preprocessing methods and classification thresholds.

- Textual Entailment:

  - Utilized a pre-trained DeBERTa-v3 model fine-tuned on MNLI, FEVER, and ANLI datasets to perform textual entailment.

  - Developed a strategy to map the model's three-class output (entailment, neutral, contradiction) to a binary decision (supported vs. not supported).

  - Implemented sentence-level comparison and passage aggregation to improve accuracy, achieving at least 83% on the dataset.

  - Optimized runtime by pruning low word overlap examples to reduce computational load.

- Error Analysis:

  - Conducted a detailed error analysis on the entailment model's predictions, focusing on false positives and false negatives.

  - Categorized errors into fine-grained types (e.g., "gold standard is wrong," "model misinterpretation") and provided aggregate statistics.

  - Analyzed specific examples to understand the root causes of errors, offering insights into model limitations and potential improvements.
 
## Skills Demonstrated:

- Data Preprocessing & Feature Engineering: Applied advanced text preprocessing techniques to improve model performance.

- Model Evaluation & Optimization: Tuned models to achieve high accuracy and optimized runtime efficiency.

- Error Analysis & Insight Generation: Conducted thorough error analysis to identify model weaknesses and propose actionable improvements.

- Technical Proficiency: Utilized Python, PyTorch, and Hugging Face's transformers library to implement and evaluate models.
