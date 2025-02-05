# Fine-Tuning LLama 3 (Meta LLM) on Google Colab with Tesla T4 GPU

This repository contains the necessary resources and instructions to fine-tune the **LLama 3 (8B) LLM** model, developed by Meta, using a Tesla T4 GPU on a Google Colab remote machine.
 
## Project Overview
This project showcases the process of fine-tuning the [**LLama 3 (8B) LLM**](https://ai.meta.com/blog/meta-llama-3/) model on Google Colab, leveraging the computational power of the Tesla T4 GPU. The notebook included in this repository walks through the steps needed to set up, configure, and fine-tune the model for customized language tasks.

## Dataset Information
The dataset used for fine-tuning this model is the **Alpaca Cleaned dataset** which is a cleaned version of the original [Alpaca Dataset released by Stanford](https://crfm.stanford.edu/2023/03/13/alpaca.html). This dataset consists of cleaned and structured data designed to improve instruction-following tasks. The dataset is hosted on [Hugging Face](https://huggingface.co/datasets/yahma/alpaca-cleaned).  

### Key Features of the Dataset:
- **Source**: Derived from the original Alpaca dataset and cleaned for better accuracy and performance.
- **Size**: Contains around 52,000 samples of instructions and responses.
- **Domain**: Focused on general language tasks, making it versatile for various NLP applications.
- **Data Preprocessing**: The data has undergone cleaning to remove errors, inconsistencies, and any irrelevant information. 

The cleaned Alpaca dataset allows the model to be fine-tuned on high-quality instruction-following tasks, improving its performance on real-world applications such as dialogue systems and content generation. 

## Key Features
- Utilizes **Google Colab** as a remote machine to handle large model training without the need for local resources.
- **Tesla T4 GPU** is used to accelerate the fine-tuning process, ensuring efficient execution.
- Comprehensive setup instructions included in the Jupyter notebook, making the process easy to follow.

## Contents
- **Fine-Tuning_LLama3_8B.ipynb**: The primary Jupyter notebook that guides the user through setting up the Colab environment and fine-tuning the LLama 3 model. 
- **Dataset**: [Alpaca Cleaned dataset](https://huggingface.co/datasets/yahma/alpaca-cleaned) used for fine-tuning the model.

## Prerequisites
- A Google account to access **Google Colab**.
- Familiarity with Python and machine learning concepts.
- Basic understanding of working with language models. 

## Setup and Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/parsafarshadfar/Finetune_LLama3-8B.git
   ```
2. Open **Google Colab** and upload the **Fine-Tuning_LLama3_8B.ipynb** notebook.
3. Follow the instructions in the notebook to set up the environment and start the fine-tuning process.

## Acknowledgments 
- The **LLama 3 LLM** model was developed by **Meta**.
- The dataset used for fine-tuning is the **Alpaca Cleaned dataset** available on [Hugging Face](https://huggingface.co/datasets/yahma/alpaca-cleaned). 