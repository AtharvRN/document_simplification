# Text_simplication

CS5803 Natural Language Processing - Document Level Text Simplification - A Two-stage Plan-Guided Approach

To activate the conda environment using environment.yml files, navigate to the `SimSum` directory and run the following command:

```
cd SimSum
conda env create -f environment.yml
```

We have used codes from [Document Level Planning for Text Simplification](https://github.com/liamcripwell/plan_simp) and [SimSum](https://github.com/epfml/easy-summary/tree/main).

## Datasets Used
- Smaller version of the Wiki-Auto-Dataset which can be found [here](SimSum/data/wiki_auto_reduced).
- Plaba Dataset has been taken from (here)[https://osf.io/rnpmf/]
- D-Wikipedia has been taken from (here)[https://github.com/epfml/easy-summary/tree/main/SimSum/data/D_wiki]
# EVALUATION

Use the following commands to evaluate output:

The outputs can be evaluated using the following command:

```bash
python evaluate.py --model MODEL --dataset DATASET
```

**MODEL's** - `BART, SIMSUM, PG_SIMSUM`

**DATASET's** - `WIKI_AUTO_REDUCED, PLABA, D_WIKI`

The model output on the corresponding test datasets and tensorboard logs are available in the `SimSum/experiments` folder.

The model files can be downloaded from [here](https://iith-my.sharepoint.com/:f:/g/personal/ee20btech11006_iith_ac_in/Eh3hX8Mtf2BClrZ0x8dj_NYBY_aKpwEcOJuFPMul5XDP9Q?e=8CtgcT) and replace the experiments directory with the downloaded directory.

We have used the existing model weights (from Hugging Face) for the Plan Guided Model for evaluation.

## To generate output for the wiki_auto dataset:

```bash
python generate.py dynamic --clf_model_ckpt=liamcripwell/pgdyn-plan --model_ckpt=liamcripwell/pgdyn-simp --test_file=data_pg/wiki_auto/wikiauto_sents_test_reduced.csv --doc_id_col=pair_id --context_doc_id=pair_id --context_dir=context_save_dir/wiki_auto/test --out_file=data_pg/wiki_auto/wiki_auto_reduced_output.csv
```

## For the D_wiki dataset:

```bash
python generate.py dynamic --clf_model_ckpt=liamcripwell/pgdyn-plan --model_ckpt=liamcripwell/pgdyn-simp --test_file=data_pg/D_wiki/DWiki_sents_test.csv --doc_id_col=pair_id --context_doc_id=pair_id --context_dir=context_save_dir/D_wiki/test --out_file=data_pg/D_wiki/D_wiki_output.csv
```

## For the Plaba dataset:

```bash
python generate.py dynamic --clf_model_ckpt=liamcripwell/pgdyn-plan --model_ckpt=liamcripwell/pgdyn-simp --test_file=data_pg/plaba/plaba_sents_test.csv --doc_id_col=pair_id --context_doc_id=pair_id --context_dir=context_save_dir/plaba/test --out_file=data_pg/plaba/plaba_output.csv 
```

Every time you run `generate.py`, make sure to delete the `temp_embeds/` directory:

```bash
rm -r temp_embeds/
```

# TRAINING

We have trained three models (BART, SIMSUM, PGSIMSUM) on the Reduced Wiki-Auto dataset.

## For BART and PG_SIMSUM:

- Uncomment relevant lines in `SimSum/main.py`.
- Use `dataset = WIKI_AUTO_REDUCED`
- Run `python main.py`.

## For PG_SIMSUM:

[OPTIONAL] To obtain OPERATION TOKENS for each sentence:
- Uncomment relevant lines in `SimSum/prepend_tokens.py`.
- Run `python prepend_tokens.py`.
- Results are stored in `data/wiki_auto_reduced_control`.
- This step is necessary even for generating output during evaluation, but results are stored in the ```SimSum/data``` folder with the '_control' suffix added to the dataset name.

Uncomment relevant lines in `SimSum/main.py`.
- Use `dataset = WIKI_AUTO_REDUCED_CONTROL`.
- Run `python main.py`.

**IMPORTANT** : If you run into some error while training, downgrade transformers to verion 4.21.1 ```pip install transformers==4.21.1```
 # [OPTIONAL] CONTEXTUAL EMBEDDING

 In order to run obtain results for PG_SIMSUM and pretrained-plan guided module, we need to get context representation for surrounding sentences. This can be done using the following command

 ```
python encode_contexts.py --data=DATASET_FILE.csv --x_col=complex --id_col=pair_id --save_dir=CONTEXT_DIR
```

We have already done this for all the datasets and the embeddings are stored in ```SimSum/context_save_dir/``` 
