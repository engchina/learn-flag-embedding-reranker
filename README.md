# Learn FlagEmbeddingRereanker

## Prepare

```
conda create -n learn-flag-embedding-rereanker python=3.11 -y
conda activate learn-flag-embedding-rereanker
```

```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```
python -c "import torch; print(torch.cuda.is_available())"

---
True
---
```

## Download Data

```
mkdir -p 'data/paul_graham/'
wget 'https://raw.githubusercontent.com/run-llama/llama_index/main/docs/docs/examples/data/paul_graham/paul_graham_essay.txt' -O 'data/paul_graham/paul_graham_essay.txt'
```

reference:
- [https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/FlagEmbeddingReranker/](https://docs.llamaindex.ai/en/stable/examples/node_postprocessor/FlagEmbeddingReranker/)
- [https://blog.csdn.net/littleblack201608/article/details/136518983](https://blog.csdn.net/littleblack201608/article/details/136518983)


