import os
import re
import time
import shutil
from datetime import datetime

# import fire
import nltk
import torch
import pandas as pd
from tqdm import tqdm
# from sentence_transformers import SentenceTransformer

from plan_simp.data.utils import OP_TOKENS,prepend_tokens
from plan_simp.models.bart import load_simplifier, run_generator
from plan_simp.models.classifier import load_planner, run_classifier
# from plan_simp.
def dynamic(model_ckpt, test_file, out_file,output_text_file, clf_model_ckpt=None, doc_id_col="pair_id", context_dir=None, context_doc_id=None, temp_dir="temp_embeds",
                reading_lvl=None, op_col=None, beams=5, max_samples=None, device="cuda", result_cache=None, save_rate=10, simple_context_dir=None, simple_context_doc_id=None,):
        start = time.time()
        print(f"Starting time: {datetime.now()}")

        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        elif result_cache is None:
            raise FileExistsError(f"Specified temp directory '{temp_dir}' already exists! If you want to continue from existing cache, use the `results_cache` arg.")

        # load data
        if result_cache is not None:
            test_set = pd.read_csv(result_cache, keep_default_na=False)
        else:
            # if not os.path.exists(test_file):
            #     os.makedirs(test_file)
            print(test_file)
            test_set = pd.read_csv(test_file)

        # only enforce `max_samples` at the document level
        doc_ids = test_set[doc_id_col].unique()
        if max_samples is not None:
            doc_ids = doc_ids[:max_samples]
            test_set = test_set[test_set[doc_id_col].isin(doc_ids)]

        # load planning model
        if clf_model_ckpt is not None:
            clf_model, clf_tokenizer, clf_hparams = load_planner(clf_model_ckpt, add_context=True, device=device)
        # print(clf_model)
        # print(clf_tokenizer)
        # print(clf_hparams)
        # load simplification model
        model, tokenizer, hparams = load_simplifier(model_ckpt, device=device)

        # determine context window radius (NOTE: for now assumes each will not use varying window radii)
        if clf_model_ckpt is not None:
            z_radius = clf_hparams["context_window"]
        else:
            z_radius = hparams["context_window"]

        # SBERT model
        # sent_encoder = SentenceTransformer('all-mpnet-base-v2')
        # https://github.com/huggingface/transformers/issues/5486 happens when sending sbert vectors to cpu, will break if using many workers
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # determine whether to use pre-defined simple left-context
        l_z_dir = temp_dir
        if simple_context_dir is not None:
            l_z_dir = simple_context_dir
        if simple_context_doc_id is None:
            # NOTE: currently cannot specify simple document id column
            simple_context_doc_id = doc_id_col

        preds = ["#"]*len(test_set)
        pred_ls = [-1]*len(test_set)

        # preload existing results from cache
        if result_cache is not None:
            for i, row in test_set.iterrows():
                preds[i] = row.pred
                pred_ls[i] = row.pred_l
        # print(test_set)
        text_id = "sent_id"
        max_text_id = test_set[text_id].max()
        pbar = tqdm(total=len(test_set))

        i_sents = test_set
        clf_logits = run_classifier(clf_model, i_sents.reset_index(drop=True), "complex", tokenizer=clf_tokenizer, device=device, add_context=True,
                                                    context_dir=context_dir, context_doc_id=context_doc_id, simple_context_dir=l_z_dir, 
                                                    simple_context_doc_id=simple_context_doc_id, reading_lvl=reading_lvl, simple_sent_id=None,
                                                    return_logits=True, silent=True)
        print(len(clf_logits))
        i_sents["pred_l"] = [int(y_.argmax()) for y_ in clf_logits]
        
        i_texts = i_sents
        # print(i_texts)
        input_seqs = prepend_tokens(test_set, "complex", hparams["class_labels"], hparams["op_tokens"], op_col = "pred_l")
        i_texts["prepended_sents"] = input_seqs
        i_texts.to_csv(out_file)

        grouped_data = i_texts.groupby('pair_id')
        complex_combined = []
        simple_combined = []
        prepended_combined = []

        # Iterate through grouped data
        for pair_id, group in grouped_data:
            # Combine complex sentences
            # complex_combined.append(" ".join(group['complex']))
            
            # Combine simple sentences
            # simple_combined.append(" ".join(group['simple']))
            
            # Combine prepended sentences
            prepended_combined.append(" ".join(group['prepended_sents']))

        # Write combined sentences to text files
        # with open(output_text_file+".complex", "w") as f:
        #     f.write("\n".join(complex_combined))

        # with open(output_text_file+".simple", "w") as f:
        #     f.write("\n".join(simple_combined))

        with open(output_text_file+".complex", "w") as f:
            f.write("\n".join(prepended_combined))

        end = time.time()
        elapsed = end - start
        print(f"Done! (Took {elapsed}s in total)")
        print(f"End time: {datetime.now()}")

def main():
    clf_model_ckpt="liamcripwell/pgdyn-plan"   
    model_ckpt="liamcripwell/pgdyn-simp"    
    
    ## WIKI_AUTO_REDUCED
    context_dir="context_dir_test"
    # context_dir="context_dir_train" 
    # context_dir="context_dir_valid" 

    out_file="data_pg/wiki_auto/wikiauto_test_prepended_sents.csv" 
    # out_file="data_pg/wiki_auto/wikiauto_train_prepended_sents.csv" 
    # out_file="data_pg/wiki_auto/wikiauto_valid_prepended_sents.csv" 

    input_file="data_pg/wiki_auto/wikiauto_sents_test_reduced.csv" 
    # input_file="data_pg/wiki_auto/wikiauto_sents_train_reduced.csv" 
    # input_file="data_pg/wiki_auto/wikiauto_sents_valid_reduced.csv" 

    output_text_file ="data/wiki_auto_reduced/wiki_auto_reduced.test"
    # output_text_file ="data/wiki_auto_reduced/wiki_auto_reduced.train"
    # output_text_file ="data/wiki_auto_reduced/wiki_auto_reduced.valid"

    # context_dir = "wikidoc_context_dir_test"
    # out_file="data_pg/wiki_doc/wikidocs_test_prepended_sents.csv" 
    # input_file = "data_pg/wiki_doc/wikidocs_sents_test.csv"
    # output_text_file ="data/wiki_doc_control/wiki_doc.test"
    
    # context_dir = "D_wiki_context_dir_test"
    # out_file="data_pg/D_wiki/Dwiki_test_prepended_sents.csv" 
    # input_file = "data_pg/D_wiki/DWiki_sents_test.csv"
    # output_text_file ="data/D_wiki/D_wiki.test"

    context_dir = "context_save_dir/plaba/test"
    out_file="data_pg/plaba/plaba_test_prepended_sents.csv" 
    input_file = "data_pg/plaba/plaba_sents_test.csv"
    output_text_file ="data/plaba_control/plaba.test"
    dynamic(model_ckpt=model_ckpt,test_file=input_file,out_file=out_file,output_text_file=output_text_file,clf_model_ckpt=clf_model_ckpt,context_dir = context_dir,context_doc_id="pair_id")
if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')

    main()