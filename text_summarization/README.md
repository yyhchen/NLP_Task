# 文本摘要

以 `T5` 为案例做 `seq2seq` 模型任务实战

---


## 评价指标

- ROUGE
    - Rouge-1、Rouge-2、Rouge-L
    - 分别基于 1-gram、2-gram、LCS(最长公共子序列)

- 示例

    |  原始文本 |      1-gram     |         2-gram        |
    |:--------:|:---------------:|:--------------------:|
    |  今天不错 |   今 天 不 错     |    今天 天不 不错      |
    |今天太阳不错| 今 天 太 阳 不 错 | 今天 天太 太阳 阳不 不错 |

    - Rouge-1：P =  4/4, R = 4/6, F = 2*P*R/(P+R)
    - Rouge-2：P =  2/3, R = 2/5, F = 2*P*R/(P+R)
    - Rouge-L：P =  4/4, R = 4/6, F = 2*P*R/(P+R)

（ `今天不错` 为原始文本， `今天太阳不错` 为真实标签 ）


<br>
<br>


## 环境

### 数据集
- [supremezxc/nlpcc_2017](https://huggingface.co/datasets/supremezxc/nlpcc_2017)

- 新闻标题生成

<br>

### 预训练模型

- [Langboat/mengzi-t5-base](https://huggingface.co/Langboat/mengzi-t5-base)

- useage:
    ```python
    from transformers import T5Tokenizer, T5ForConditionalGeneration

    tokenizer = T5Tokenizer.from_pretrained("Langboat/mengzi-t5-base")
    model = T5ForConditionalGeneration.from_pretrained("Langboat/mengzi-t5-base")

    ```

<br>

### 依赖库
```bash
pip install rouge-chinese
```

<br>
<br>

## suammarization_glm.ipynb

基于 GLM 模型进行文本摘要任务
