---
library_name: keras
tags:
- protein
- protein language model
- biology
- cafa
- linial
- ofer
- GO
- Gene ontology
- protein function
- protein function prediction
- efficient attention
- global attention
- protein embedding
- ProteinBERT
- CAFA
license: mit
language:
- en
metrics:
- accuracy
---

## Model description

Pretrained Protein language model, using a mixed masked language modeling (MLM) & ELECTRA objective, as well as an additional pretraining task of predicting GO (Gene ontology) function for all UniRef90 proteins.

It was introduced in our [ProteinBERT paper](https://doi.org/10.1093/bioinformatics/btac020) and is also fully available in the [Github repository](https://github.com/nadavbra/protein_bert) - [https://github.com/nadavbra/protein_bert](https://github.com/nadavbra/protein_bert). 

## Intended uses & limitations

A pretrained language model for predicting Protein (AA) sequences and their properties. Can predict on new tasks, including whole sequence or local (per position) tasks, includding classification, multilabel and regression. Expected input is an amino acid (protein) sequence.
Model provided here outputs concatted embedding of all hidden states. Can be adapted for any application. 

#### Caveat: 
Conversion of model may have changed compatability, as tensorflow "sanitized" `input-seq` to `input_seq` and `input-annotations` to `input_annotations`.
In cases of compatibility issues or errors, we refer to the original pretraining & finetuning code, model dump and ProteinBERT package: https://github.com/nadavbra/protein_bert
## Training and evaluation data

Trained on ~106M proteins from UniRef90. Sequences were filtered in advance to remove any with over 30% similarity (by BLAST score) to any sequence in any of the TAPE benchmark datasets. 8943 most frequent GO annotations were kept for the pretraining task.



###### Getting started with pretrained ProteinBERT embeddings
Here's a quick code snippet for getting embeddings at the whole sequence (protein) level - you can use these for downstream tasks as extracted features with other ML models, clustering, KNN, etc'. (You can also get local/position level embeddings, and fine tune the ProteinBERT model itself on your task).
(The model uploaded here is the output of the following code):
```
from proteinbert import load_pretrained_model
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs

pretrained_model_generator, input_encoder = load_pretrained_model()
model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(1024))
#### example usage:
encoded_x = input_encoder.encode_X(seqs, seq_len)
local_representations, global_representations = model.predict(encoded_x, batch_size=batch_size)
# ... use these as features for other tasks, based on local_representations, global_representations
```
For getting embeddings, load the model from huggingface and get the last layers output.

Have a look at the notebook used to finetune the model on a large set of diverse tasks and benchmarks for more usage examples:
[ProteinBERT demo](https://github.com/nadavbra/protein_bert/blob/master/ProteinBERT%20demo.ipynb).



## Citation <a name="citations"></a>
=======
If you use ProteinBERT, we ask that you cite our paper:
``` 
Brandes, N., Ofer, D., Peleg, Y., Rappoport, N. & Linial, M. 
ProteinBERT: A universal deep-learning model of protein sequence and function. 
Bioinformatics (2022). https://doi.org/10.1093/bioinformatics/btac020
```

```bibtex
@article{10.1093/bioinformatics/btac020,
    author = {Brandes, Nadav and Ofer, Dan and Peleg, Yam and Rappoport, Nadav and Linial, Michal},
    title = "{ProteinBERT: a universal deep-learning model of protein sequence and function}",
    journal = {Bioinformatics},
    volume = {38},
    number = {8},
    pages = {2102-2110},
    year = {2022},
    month = {02},
    abstract = "{Self-supervised deep language modeling has shown unprecedented success across natural language tasks, and has recently been repurposed to biological sequences. However, existing models and pretraining methods are designed and optimized for text analysis. We introduce ProteinBERT, a deep language model specifically designed for proteins. Our pretraining scheme combines language modeling with a novel task of Gene Ontology (GO) annotation prediction. We introduce novel architectural elements that make the model highly efficient and flexible to long sequences. The architecture of ProteinBERT consists of both local and global representations, allowing end-to-end processing of these types of inputs and outputs. ProteinBERT obtains near state-of-the-art performance, and sometimes exceeds it, on multiple benchmarks covering diverse protein properties (including protein structure, post-translational modifications and biophysical attributes), despite using a far smaller and faster model than competing deep-learning methods. Overall, ProteinBERT provides an efficient framework for rapidly training protein predictors, even with limited labeled data.Code and pretrained model weights are available at https://github.com/nadavbra/protein\_bert.Supplementary data are available at Bioinformatics online.}",
    issn = {1367-4803},