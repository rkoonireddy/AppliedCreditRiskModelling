
# AppliedCreditRiskModelling using LangChain and Streamlit RAG
This repository uses the power of LLMs in predicting credit risk. The Credit Risk applications can be applied to 
1. Private Loans
2. Mortgages
3. Credit Card Applications


***This version runs either on WSL on windows or Ubuntu***

## Demo App on Community Cloud

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://acrm2024.streamlit.app/)


## Quickstart

### Setup Python environment
Version: 3.10.13


```bash
python -mvenv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```
or 
```bash
conda env create -f environment.yml
conda activate langchain
```

### Setup `.env` file with API tokens needed.

```
OPENAI_API_KEY="<Put your token here>"
HUGGINGFACEHUB_API_TOKEN="<Put your token here>"
```

### Setup Streamlit app secrets.

#### 1. Create `.streamlit` directory and in it `secrets.toml` file.

#### 2. Edit secrets.toml

**Either edit `secrets.toml` in you favorite editor.**

```toml
OPENAI_API_KEY="<Put your token here>"
HUGGINGFACEHUB_API_TOKEN="<Put your token here>"
```

**Or, you can just reuse .env contents from above.**

```bash
cat < .env >> .streamlit/secrets.toml
```

### Verify Environment

1. Check that LangChain dependencies are working.

```bash
cd utils
python data_loader.py
```

2. Run the program.

```bash
cd ..
streamlit run acrm.py
```

3. Rest will be self explanatory



## References
Gordon V. Cormack, Charles L A Clarke, and Stefan Buettcher. 2009. [Reciprocal rank fusion outperforms condorcet and individual rank learning methods](https://dl.acm.org/doi/10.1145/1571941.1572114). In Proceedings of the 32nd international ACM SIGIR conference on Research and development in information retrieval (SIGIR '09). Association for Computing Machinery, New York, NY, USA, 758–759. <https://doi.org/10.1145/1571941.1572114>.

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Singh Chaplot, D., de las Casas, D., … & El Sayed, W. (2023). [Mistral 7B](https://arxiv.org/abs/2310.06825). arXiv e-prints, arXiv-2310. <https://doi.org/10.48550/arXiv.2310.06825>.

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., … & Kiela, D. (2020). [Retrieval-augmented generation for knowledge-intensive nlp tasks](https://arxiv.org/abs/2005.11401). Advances in Neural Information Processing Systems, 33, 9459–9474.

Liu, N. F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., & Liang, P. (2024). [Lost in the middle: How language models use long contexts](https://arxiv.org/abs/2307.03172). Transactions of the Association for Computational Linguistics, 12, 157–173.

Robertson, S., & Zaragoza, H. (2009). [The probabilistic relevance framework: BM25 and beyond](https://dl.acm.org/doi/10.1561/1500000019). Foundations and Trends® in Information Retrieval, 3(4), 333–389. <https://doi.org/10.1561/1500000019>

Thibault Formal, Benjamin Piwowarski, and Stéphane Clinchant. 2021. [SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking](https://dl.acm.org/doi/10.1145/3404835.3463098). In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '21). Association for Computing Machinery, New York, NY, USA, 2288–2292. <https://doi.org/10.1145/3404835.3463098>.

Tunstall, L., Beeching, E., Lambert, N., Rajani, N., Rasul, K., Belkada, Y., … & Wolf, T. (2023). Zephyr: Direct Distillation of LM Alignment. arXiv e-prints, arXiv-2310. <https://doi.org/10.48550/arXiv.2310.16944>.