# bertnlu

chinese bert nlu for AI dialog, domain/intent and slot extraction, with weakly labelled data  
中文NLU模型（句子分类&序列标注 for 语音语义理解），包含预训练模型、http服务、弱标注数据  

model compressed to 40MB, takes ~80ms on 4core cpu machines for inference, therefore can be used for online serivices.   
模型基本结构为transformer encoder，通过结构剪枝、动态量化压缩到40MB，在cpu机器上推理耗时几十毫秒，可以用于线上基础语义理解服务。



## Demo/在线演示

http://service-b7ddzl8o-1251316161.gz.apigw.tencentcs.com/



## Instructions/代码说明

### dependencies

pytorch/transformers/numpy

### file structure

`run.py` : load and run test by default, change it for run training  
`train.py` : training  
`model.py` : model structure and load pretrained params  
`prepare.py` : raw data processing, to torch data  
`mdl/serv.py` : serv on flask  

### miscellaneous

1. Mainly for Chinese NLU, therefore ngrams for tokenizer are removed, so English words and large numbers will be treated as `[UNK]`, which will not affect sentence classification and slot filling as they would be understand as some entity by their contexts in Chinese corpus. If you need to leave origin texts for the model training, you may just add `' '.join(text)`  before tokenization.

2. Due to the quality of dataset is hard to be regarded as benchmark, it is not seprated to training set, validation set and test set.

3. It is a base model for dialog text understanding, mainly for instructions, rather than dialogs. Preprocessings and postprocessings are still needed for better serice quality.

