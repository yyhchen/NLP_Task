# 预训练模型实战

预训练中的 两种预训练任务
1. mask language model （自编码）
2. causal language model （自回归）

---



## 1. Mask Language Model (自编码)
- [BERT](https://github.com/ymcui/Chinese-BERT-wwm)
- [RoBERTa]()
- [XLNet]()
- [ERNIE]()
- [ELECTRA](https://github.com/ymcui/Chinese-ELECTRA-Base-Discriminator)


### pretrain_mask_language_model.py
- 用了 `transformers` 中的 `AutoModelForMaskedLM` 加载 `hlf/chinese-macbert-base` 做 fine-tuned 为例演示

<br>
<br>


## 2. Causal Language Model (自回归)
- [GPT-2]()
- [GPT-3]()
- [T5]()


### p retrain_causal_language_model.py
- 用了 `transformers` 中的 `AutoModelForCausalLM` 加载 `Langboat/bloom-389m-zh` 做 fine-tuned 为例演示