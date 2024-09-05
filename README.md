# NLP_Task
基于Huggingface Transformers实战


## transformers库一些语法(base_syntax)
[base_syntax](https://github.com/yyhchen/NLP_Task/tree/main/base_syntax)

- pipelines
- TrainingArguments

<br>
<br>

## 预训练(pretrain_model)

[pretrain_model](https://github.com/yyhchen/NLP_Task/tree/main/pretrain_model)

- 自编码
- 自回归


<br>
<br>

## 微调(Fine-tuned)

[Fine-tuned](https://github.com/yyhchen/NLP_Task/tree/main/Fine-tuned)

基于 PEFT 方法的微调

- Prefix Tuning
- Prompt Tuning
- P-Tuning
- LoRA

<br>
<br>


## 分布式训练(distributed_train)

[distributed_train](https://github.com/yyhchen/NLP_Task/tree/main/distributed_train)

>只涉及到了单机多卡的部分，多机多卡的部分没有涉及

- DP 直接忽略了，太简单了，而且也不符合现在主流的分布式训练方式
- DDP -> 原生 pytorch, accelerate, 
- Deepspeed -> 通过 accelerate 集成使用的 DeepSpeed, 无原生 DeepSpeed 训练内容


<br>
<br>


## NLP 基本任务

### 命名实体识别(ner)
[ner](https://github.com/yyhchen/NLP_Task/tree/main/ner)


### 文本分类(classification_demo)
[classification_demo](https://github.com/yyhchen/NLP_Task/tree/main/classification_demo)


### 机器阅读理解(machine reading comprehension)
[mrc](https://github.com/yyhchen/NLP_Task/tree/main/mrc)


### 文本相似度(sentence similarty)
[sentence_similarity](https://github.com/yyhchen/NLP_Task/tree/main/sentence_similarity)


### 文本摘要(text summarization)
[text_summarization](https://github.com/yyhchen/NLP_Task/tree/main/text_summarization)


### 向量检索匹配(retrieval_chatbot)
[retrieval_chatbot](https://github.com/yyhchen/NLP_Task/tree/main/retrieval_chatbot)


### 文本生成与对话(generative_chatbot)
[generative_chatbot](https://github.com/yyhchen/NLP_Task/tree/main/generative_chatbot) , 涉及指令微调(SFT), 即模型能够对话的关键 (基础任务不涉及模型对齐任务)

