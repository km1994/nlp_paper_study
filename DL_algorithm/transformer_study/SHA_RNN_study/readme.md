# 【关于 SHA_RNN】 那些的你不知道的事

> 作者：杨夕
> 
> 面向任务：Language Understanding
> 
> 论文名称：Single Headed Attention RNN: Stop Thinking With Your Head 单头注意力 RNN: 停止用你的头脑思考
> 
> 【注：手机阅读可能图片打不开！！！】 


## Abstract

The leading approaches in language modeling are all obsessed with TV shows of my youth - namely Transformers and Sesame Street. Transformers this, Transformers that, and over here a bonfire worth of GPU-TPU-neuromorphic wafer scale silicon. We opt for the lazy path of old and proven techniques with a fancy crypto inspired acronym: the Single Headed Attention RNN (SHA-RNN). The author’s lone goal is to show that the entire field might have evolved a different direction if we had instead been obsessed with a slightly different acronym and slightly different result. We take a previously strong language model based only on boring LSTMs and get it to within a stone’s throw of a stone’s throw of state-of-the-art byte level language model results on enwik8. We also achieve state-of-the-art on WikiText-103 - or do we? This work has undergone no intensive hyperparameter optimization and lived entirely on a commodity desktop machine that made the author’s small studio apartment far too warm in the midst of a San Franciscan summer. The final results are achievable in plus or minus 24 hours on a single GPU as the author is impatient. The attention mechanism is also readily extended to large contexts and requires minimal computation. Take that Sesame Street.
