# Fine-tuned based PEFT

[PEFT 总结综述](https://arxiv.org/abs/2303.15647)

📦 本环境用的全是基于 `Causal language model`, 代码基于都是一样，只有 `fine-tuned` 部分有区别

---

<br>
<br>

## BitFit fine-tuned 

### 简介

[bitfit 简介](https://github.com/yyhchen/Notes/blob/main/NLP%20review/fine-tuned/BitFit/BitFit.md)

### 实验拆解
1. 包括分析模型参数细节及占用显存, 如何手动设置fine-tuned 只更新 `bias` 部分参数
2. 如何更细粒度得只保存 `bias` 部分的参数，而不是整个模型参数，并做出了保存前后 参数的比较部分


<br>
<br>



## Prompt-tuning

### 简介

[prompt-tuning 简介](https://github.com/yyhchen/Notes/blob/main/NLP%20review/fine-tuned/Prompt-Tuning/Prompt-tuning.md)


### 实验拆解
1. 由 `prompt-tuning` 引出 Prompt 的两种形式，`soft` 和 `hard`; `soft prompt` 是随机初始化的，通常来说 `prompt-tuning` 的 `soft prompt` 效果会比较差，需要经过更多的 `epoch` 来获取好的 效果。
2. 跟之前的 `bitft tuning` 相比，参数量大大下降，从模型的信息可以获取参数下降的原因。



<br>
<br>


## P-tuning

### 简介

[p-tuning 简介]()

### 实验拆解
1. `p-tuning` 在 `prompt-tuning` 基础上进行改进，在 `embedding` 层的 `prompt` 前缀加上了一个 重参数化的行为（两种：MLP和LSTM）。



<br>
<br>



## Prefix-tuning

### 简介

[prefix-tuning 简介](https://github.com/yyhchen/Notes/tree/main/NLP%20review/fine-tuned/Prefix-Tuning)

### 实验拆解
1. 跟 `p-tuning` 很像，但是 `p-tuning` 是在embedding层进行拼接； `prefix-tuning` 是在整个 **transformers blocks** 进行拼接然后进行学习的。
2. 利用了 kv cache 的原理


<br>
<br>



## LoRA

### 简介

[LoRA 简介](https://github.com/yyhchen/Notes/tree/main/NLP%20review/fine-tuned/LoRA)

### 实验拆解
1. 分析了 `LoRA` 可以如何添加 在不同的参数层，默认只在 `query_key_value` 参数层进行 分解
2. 对比了 加载 `LoRA` 模型 和 合并 `LoRA` 模块的两种行为 