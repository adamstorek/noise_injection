# Unsupervised Selective Rationalization with Noise Injection

This repository contains code for [Unsupervised Selective Rationalization with Noise Injection](https://doi.org/10.48550/arXiv.2305.17534) to appear in ACL 2023. It includes the PyTorch implementation of BERT-A2R (+ NI) model, preprocessing scripts for each dataset used in our evaluation, as well as the USR Movie Review dataset.

## Model

Please note that to train our model, you will need a GPU with 24+ GB of VRAM.
To reproduce our results, please:

1. Set up a conda environment as specified in `spec-file.txt`.
2. Download the original dataset files for MultiRC, FEVER, or ERASER Movie Reviews from [ERASER Benchmark](http://www.eraserbenchmark.com).
3. To preprocess each dataset in question (except for USR Movie Reviews, which are already in the correct format in `usr_movie_review`), navigate to `data_preprocessing/dataset_in_question` and run the first script

    `python prepare_dataset.py --data_path=path_to_the_dataset_in_question --save_path=path_to_the_preprocessed_dataset`
4. To generate the token statistics needed for noise injection for each dataset in question, run the second script

    `python generate_word_statistics.py --data_path=path_to_the_preprocessed_dataset`

4. Train and evaluate our model by running the corresponding script in `model` using the training parameters found in Appendix B of our paper (The default parameters of the script are not always the best for a specific
dataset).

    - `run_ce.py --train --evaluate --dataset=multirc_or_fever --data_path=path_to_the_preprocessed_dataset`  for Claim/Evidence datasets with sentence-level rationales (MultiRC, FEVER)

    - `run_words.py --train --evaluate --data_path=path_to_the_preprocessed_dataset`  for datasets with token-level rationales (USR Movies, ERASER Movies)

## USR Movie Review Dataset

The dataset can be found in `usr_movie_review`. It contains the training, validation,
and test splits as used in our evaluation. The training and validation documents are represented as (list_of_tokens, label). The test documents are represented as (list_of_tokens, label, list_of_rationale_ranges).

## Citation

If you use our code and/or dataset, please cite:
```
@inproceedings{storek-etal-2023-unsupervised,
    title = "Unsupervised Selective Rationalization with Noise Injection",
    author = "Storek, Adam  and
      Subbiah, Melanie  and
      McKeown, Kathleen",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.acl-long.707",
    pages = "12647--12659",
    abstract = "A major issue with using deep learning models in sensitive applications is that they provide no explanation for their output. To address this problem, unsupervised selective rationalization produces rationales alongside predictions by chaining two jointly-trained components, a rationale generator and a predictor. Although this architecture guarantees that the prediction relies solely on the rationale, it does not ensure that the rationale contains a plausible explanation for the prediction. We introduce a novel training technique that effectively limits generation of implausible rationales by injecting noise between the generator and the predictor. Furthermore, we propose a new benchmark for evaluating unsupervised selective rationalization models using movie reviews from existing datasets. We achieve sizeable improvements in rationale plausibility and task accuracy over the state-of-the-art across a variety of tasks, including our new benchmark, while maintaining or improving model faithfulness.",
}
```
