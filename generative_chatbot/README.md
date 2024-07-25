# 生成式对话机器人


---


## 环境准备

**1. 数据集:**

based [lyuricky/alpaca_data_zh_51k](https://huggingface.co/datasets/lyuricky/alpaca_data_zh_51k) 处理得到的

<br>

**2. 基础模型**

[Langboat/bloom-389m-zh](https://huggingface.co/Langboat/bloom-389m-zh)


<br>
<br>



## 预训练任务

causal language model, autogressive model (based decoder)

- 根据上文做下文 token 预测

- 结束位置要有特殊token（eos_token）


<br>
<br>


## 指令微调 (能对话的关键因素)

- 指令微调，赋予模型对话的能力

- 多类型的任务共同学习，能够解决不同的问题

- 单轮对话：不计算前缀(`prompt`) 的loss，只计算output的 loss

- 多轮对话：
    - 之前回答的部分加起来也作为 `prompt`,计算当前轮对话的 output loss；但是效率低
    - 计算每一轮 output 的 loss， 效率更高

