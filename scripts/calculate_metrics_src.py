import string
import re
import json
import sys
import os
import argparse
import logging
from collections import Counter,defaultdict
from tqdm import tqdm
import os
os.environ["TRANSFORMERS_CACHE"] = "/scratch/rml6079/.cache/huggingface"

sys.path.append("/scratch/rml6079/project/Instruct_dataset_training_code/src")

import numpy as np
from rouge import rouge_scorer


logger = logging.getLogger(__name__)

# class GPTTokenizer:
#     gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2", max_length=1e5)

#     def tokenize(self, s):
#         tokens = self.gpt_tokenizer.tokenize(s)
#         # GPT2 uses Byte-level BPE, which will include space as part of the word. 
#         # But for the first word of a sentence, there is no space before it. 
#         # So, we remove all the added spaces ("Ġ"). 
#         tokens = [t.lstrip("Ġ") for t in tokens]
#         return tokens

# xlingual_tokenizer = GPTTokenizer()


# adapted the flowing from Squad v1.1 evaluation, without removing the articles.
def normalize_answer(s):
    """Lower text and remove punctuation, and extra whitespace."""

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))


def exact_match_score(prediction, ground_truth, xlingual=False):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))

def exact_match_score_strict(prediction, ground_truth, xlingual=False):
    '''dont normalize the answer, just compare the string'''
    return (prediction == ground_truth)

def rouge1_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = rouge_scorer.RougeScorer(['rouge1'], tokenizer=xlingual_tokenizer)
    else:
        scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rouge1"].fmeasure


def rougeL_score(prediction, ground_truth, xlingual=False):
    if xlingual:
        scorer = rouge_scorer.RougeScorer(['rougeL'], tokenizer=xlingual_tokenizer) 
    else:
        scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(prediction=prediction, target=ground_truth)
    return scores["rougeL"].fmeasure


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths, xlingual=False):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth, xlingual=xlingual)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def compute_metrics(predictions, references, xlingual=False):
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    exact_match, exact_match_strict, rouge1, rougeL = 0, 0, 0, 0
    for pred, gold in zip(predictions, references):
        assert isinstance(gold, list)
        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        exact_match_strict += metric_max_over_ground_truths(
            exact_match_score_strict, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rouge1 += metric_max_over_ground_truths(
            rouge1_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
        rougeL += metric_max_over_ground_truths(
            rougeL_score, prediction=pred, ground_truths=gold, xlingual=xlingual
        )
    exact_match = 100.0 * exact_match / len(references)
    exact_match_strict = 100.0 * exact_match_strict / len(references)
    rouge1 = 100.0 * rouge1 / len(references)
    rougeL = 100.0 * rougeL / len(references)
    metrics = {"exact_match": exact_match, "exact_match_strict": exact_match_strict, "rouge1": rouge1, "rougeL": rougeL}
    metrics = {k: round(v, 4) for k, v in metrics.items()}
    return metrics


def compute_grouped_metrics(predictions, references, groups, xlingual=False):
    assert len(predictions) == len(references) == len(groups)

    examples_by_group = {}
    for pred, gold, group in zip(predictions, references, groups):
        if group not in examples_by_group:
            examples_by_group[group] = []
        examples_by_group[group].append((pred, gold))
    
    results = {}
    for group, group_examples in examples_by_group.items():
        task_predictions, task_references = zip(*group_examples)
        group_metrics = compute_metrics(task_predictions, task_references, xlingual=xlingual)
        for metric, value in group_metrics.items():
            results[f"{metric}_for_{group}"] = value
    return results

def compute_grouped_metrics_v2(predictions, references, groups, xlingual=False):
    '''
    the only difference is that we calculate the avg EM and Rouge for CLS and Gen tasks, respectively. 
    '''
    assert len(predictions) == len(references) == len(groups)
    
    ## metric list
    EM_list = ["TE","CEC","CR","DAR","AC","WA"]
    Rouge_list = ["OE","KT","QR","TG","DT","GEC"]
    ## abv dic
    name_abv = {"answerability_classification":"AC"		
    ,"cause_effect_classification": "CEC"		
    ,"coreference_resolution": "CR"		
    ,"data_to_text": "DT"		
    ,"dialogue_act_recognition": "DAR"		
    ,"grammar_error_correction": "GEC"		
    ,"keyword_tagging": "KT"		
    ,"overlap_extraction": "OE"		
    ,"question_rewriting": "QR"		
    ,"textual_entailment": "TE"		
    ,"title_generation": "TG"		
    ,"word_analogy": "WA"}

    examples_by_group = {}
    for pred, gold, group in zip(predictions, references, groups):
        if group not in examples_by_group:
            examples_by_group[group] = []
        examples_by_group[group].append((pred, gold))
    
    results = {}
    all_em,all_rg = [],[] 
    for group, group_examples in examples_by_group.items():
        task_predictions, task_references = zip(*group_examples)
        group_metrics = compute_metrics(task_predictions, task_references, xlingual=xlingual)
        for metric, value in group_metrics.items():
            results[f"{metric}_for_{group}"] = value
            # calculate the cls and gen tasks respectively.
            if metric == 'exact_match' and name_abv.get(group,"None") in EM_list:
                all_em.append(value)
            if metric == 'rougeL' and name_abv.get(group,"None") in Rouge_list:
                all_rg.append(value)
    
    # avg 
    results['CLS_exact_match_avg'] = np.mean(all_em) if len(all_em) > 0 else -1
    results['GEN_rougeL_avg'] = np.mean(all_rg) if len(all_rg) > 0 else -1
    
    return results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="Path to predictions file.")
    parser.add_argument("--track", choices=["default", "xlingual"], default="default", 
        help="default track or xlingual track. For xlingual, we need to use a different tokenizer."
    )
    parser.add_argument("--compute_per_category_metrics", action="store_true", help="Compute metrics on every evaluation category.")
    parser.add_argument("--compute_per_task_metrics", action="store_true", help="Compute metrics on every evaluation task.")
    return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_args()
#     with open(args.predictions) as fin:
#         examples = [json.loads(l) for l in fin]

#     predictions = [e["prediction"] for e in examples]
#     references = [e["Instance"]["output"] for e in examples]
#     tasks = []
#     for e in examples:
#         if e["Task"] == "task121_atomic_question_rewriting":
#             e["Task"] = "task121_zest_question_rewriting"
#         tasks.append(e["Task"])

#     results = compute_metrics(predictions, references, xlingual=args.track == "xlingual")
#     print("======== Overall Metrics ========")
#     print("all_rougeL", results["rougeL"])
#     print("all_EM", results["exact_match"])
#     print()
    
#     category_metrics = [
#         ("Textual Entailment", "exact_match"),
#         ("Cause Effect Classification", "exact_match"),
#         ("Coreference Resolution", "exact_match"),
#         ("Dialogue Act Recognition", "exact_match"),
#         ("Answerability Classification", "exact_match"),
#         ("Word Analogy", "exact_match"),
#         ("Overlap Extraction", "rougeL"),
#         ("Keyword Tagging", "rougeL"),
#         ("Question Rewriting", "rougeL"),
#         ("Title Generation", "rougeL"),
#         ("Data to Text", "rougeL"),
#         ("Grammar Error Correction", "rougeL"),
#     ]
#     category_metrics = {"_".join(category.lower().split()): metric for category, metric in category_metrics}

#     if args.compute_per_category_metrics:
#         print("======== Metrics per category ========")
#         task_category = {}
#         for task in set(tasks):
#             with open(os.path.join("./data/tasks/", task+".json")) as fin:
#                 task_data = json.load(fin)
#                 task_category[task] = "_".join(task_data["Categories"][0].lower().split())
#         categories = [task_category[e["Task"]] for e in examples] 
#         results.update(compute_grouped_metrics(predictions, references, categories, xlingual=args.track=="xlingual"))
        
#         for category, metric in category_metrics.items():
#             # category = "_".join(category.lower().split())
#             if f"{metric}_for_{category}" in results:
#                 print(f"{metric}_for_{category}", results[f"{metric}_for_{category}"])
#         print()
            
#     if args.compute_per_task_metrics:
#         print("======== Metrics per task ========")
#         results_by_task = compute_grouped_metrics(predictions, references, tasks, xlingual=args.track=="xlingual")
#         for task in sorted(list(set(tasks))):
#             category = task_category[task]
#             metric = category_metrics[category]
#             print(task, results_by_task[f"{metric}_for_{task}"])
#         print()

def rouge_score(prediction, reference):
    '''
    prediction: list of strings [x,x,x, ...]
    reference: list of strings [y,y,y, ...]
    
    return rouge1 and rougeL score
    '''
    if len(prediction) == 0:
        return 0.0, 0.0
    
    if len(prediction) < len(reference):
        reference = reference[:len(prediction)]
    elif len(prediction) > len(reference):
        prediction = prediction[:len(reference)]
    
    r1_list, rl_list = [], []
    for pred, ref in zip(prediction, reference):
        r1 = rouge1_score(pred, ref)
        rl = rougeL_score(pred, ref)
        r1_list.append(r1)
        rl_list.append(rl)
    
    rouge1 = np.mean(r1_list)
    rougeL = np.mean(rl_list)
    return rouge1, rougeL


def soft_accumulate(prediction, reference, model):
    '''
    used for weaknesss list, that one prediction list compares with multiple reference lists.
    
    prediction: list of strings [x,x,x, ...]
    reference: list of list of strings [[y,y,y], [z,z,z], ...]
    
    return f1, precision, recall
    '''
    import torch
    from sentence_transformers import SentenceTransformer, util
    if len(prediction) == 0:
        return 0.0, 0.0, 0.0

    pred_vec = []
    ref_vec_nested = []
    for pred in prediction:
        embed_pred= model.encode(pred, convert_to_tensor=True)
        pred_vec.append(embed_pred)
    for ref_list in reference:
        ref_vec = []
        if len(ref_list) == 0:
            continue
        for ref in ref_list:   
            embed_ref = model.encode(ref, convert_to_tensor=True)  # TODO: make batch encoding in the future
            ref_vec.append(embed_ref)
        ref_vec = torch.stack(ref_vec)
        ref_vec_nested.append(ref_vec)
    
    # calculate the similarity matrix
    pred_vec = torch.stack(pred_vec)
    sim_matrix_list = []
    for ref_vec in ref_vec_nested:
        sim_matrix = util.pytorch_cos_sim(pred_vec, ref_vec)  # tensor with shape of (len(pred_vec), len(ref_vec))
        sim_matrix_list.append(sim_matrix)

    # calculate recall, for each ref_vec, still find the max similarity for each pred_vec
    recall_list = []
    for sim_matrix in sim_matrix_list:
        recall = sim_matrix.max(dim=0)[0].mean().item()
        recall_list.append(recall)
    avg_recall = np.mean(recall_list)
    
    # calculate precision, for each pred_vec, find max similarity for each ref_vec, then avg them.
    precision_score_list = []
    for sim_matrix in sim_matrix_list:
        # get a (len(pred_vec), 1) tensor
        max_match_score_matrix = sim_matrix.max(dim=1)[0].unsqueeze(1)
        precision_score_list.append(max_match_score_matrix)
    precision_matrix = torch.cat(precision_score_list, dim=1)  # tensor with shape of (len(pred_vec), len(ref_vec_nested))

    # dim=1 take avg
    all_precision = precision_matrix.mean(dim=1)  # tensor with shape of (len(pred_vec),)
    # then take avg of the final tensor
    avg_precision = all_precision.mean().item()

    f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall + 1e-10)
    
    return f1, avg_precision, avg_recall
    

def soft_score(prediction, reference, model):
    '''
    different from `soft_f1`, this function assumes two lists have the same length --- each item in one list is corresponding to the item in the other list.
    
    prediction: list of strings [x,x,x, ...]
    reference: list of strings [y,y,y, ...]
    
    return a score
    '''
    if len(prediction) != len(reference):
        # TODO: not sure how to deal with the length mismatch.
        # print(f"# of predictions {len(prediction)} doesn't match # of references {len(reference)}. Go for soft_f1.")
        # f1, precision, recall = soft_f1(prediction, reference, model)
        # return f1
        print(f"# of predictions {len(prediction)} doesn't match # of references {len(reference)}. Cut the longer one.")
        if len(prediction) == 0:
            return 0.0
        else:
            # cut the pred or ref to the same length
            if len(prediction) > len(reference):
                prediction = prediction[:len(reference)]
            else:
                reference = reference[:len(prediction)]
    
    assert len(prediction) == len(reference), f"# of predictions {len(prediction)} doesn't match # of references {len(reference)}."
    # otherwise, we calculate the average similarity score between each pair of prediction and reference.
    import torch
    from sentence_transformers import SentenceTransformer, util
    pred_vec = []
    ref_vec = []
    for pred in prediction:
        embed_pred= model.encode(pred, convert_to_tensor=True)
        pred_vec.append(embed_pred)
    for ref in reference:   
        embed_ref = model.encode(ref, convert_to_tensor=True)
        ref_vec.append(embed_ref)
    
    pred_vec = torch.stack(pred_vec)
    ref_vec = torch.stack(ref_vec)
    sim_matrix = util.pytorch_cos_sim(pred_vec, ref_vec)  # tensor with shape of (len(pred_vec), len(ref_vec))
    # get the diagonal elements
    score = sim_matrix.diag().mean().item()
    
    return score
    


def soft_f1(prediction, reference, model):
    '''
    prediction: list of strings [x,x,x, ...]
    reference: list of strings [y,y,y, ...]
    
    return f1, precision, recall
    '''
    import torch
    from sentence_transformers import SentenceTransformer, util
    pred_vec = []
    ref_vec = []
    for pred in prediction:
        embed_pred= model.encode(pred, convert_to_tensor=True)
        pred_vec.append(embed_pred)
    for ref in reference:   
        embed_ref = model.encode(ref, convert_to_tensor=True)
        ref_vec.append(embed_ref)
    
    if pred_vec == []:
        return 0, 0, 0
    pred_vec = torch.stack(pred_vec)
    ref_vec = torch.stack(ref_vec)
    sim_matrix = util.pytorch_cos_sim(pred_vec, ref_vec)  # tensor with shape of (len(pred_vec), len(ref_vec))
    # import pdb; pdb.set_trace()
    precision = sim_matrix.max(dim=1)[0].mean().item()
    recall = sim_matrix.max(dim=0)[0].mean().item()
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    return f1, precision, recall
    
        
'''
sentence similarity based precision, recall, f1
'''
def SentenceSemanticMetric(predictions, references):
    '''
    predicstions: [[x,x,x],[y,y,y], ...]
    references: [[x,x,x],[y,y,y], ...]
    
    return F1, precision, recall
    '''
    import torch
    from sentence_transformers import SentenceTransformer, util
    assert len(predictions) == len(references), f"# of predictions {len(predictions)} doesn't match # of references {len(references)}."
    # bertscore = load("bertscore")
    f1_list, precision_list, recall_list = [], [], []
    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    for prediction, reference in tqdm(zip(predictions, references)):
        f1, precision, recall = soft_f1(prediction, reference, model)
        # import pdb; pdb.set_trace()
        f1_list.append(f1)
        precision_list.append(precision)
        recall_list.append(recall)
    
    f1 = np.mean(f1_list)
    precision = np.mean(precision_list)
    recall = np.mean(recall_list)
    return f1, precision, recall


def cross_focus_diversity(all_papers_weakness_list:list, model=None, threshold=0.5, inverse_tf=True, batch_size=512):
    '''
    all_papers_weakness_list: a list, each item is a tuple of (paper_id, weakness_list) or (track_name, weakness_list)
    
    return a score reflecting the paper specific focus of the generated weakness list.
    
    a higher score means a generated weakness list is more specifc the corresponding paper.
    
    return:
        1. idf_score
        2. id2score: a dict, like {paper_id: [score1, score2, ...], ...}, each paper id and its corresponding segment's idf score.
    '''
    import torch
    from sentence_transformers import SentenceTransformer, util
    THRESHOLD = threshold
    if model is None:
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # remove the line that contains the word "clarity" and "quality" and "novelty"
    # new_all_papers_weakness_list = []
    # for idx, (paper_id, weakness_list) in enumerate(all_papers_weakness_list):
    #     new_weakness_list = []
    #     for line in weakness_list:
    #         # some templates lines
    #         if not ("clarity" in line.lower() and "quality" in line.lower() and "novelty" in line.lower() and "reproducibility" in line.lower()):
    #             new_weakness_list.append(line)
    #     new_all_papers_weakness_list.append((paper_id, new_weakness_list))
    # all_review_text_list = new_all_papers_weakness_list
    all_review_text_list = all_papers_weakness_list

    all_review_embedding_dict = dict()
    for idx, (paper_name, text) in enumerate(tqdm(all_review_text_list)):
        # batch forward the text
        # embeddings = model.encode(text)
        # make the embeddings a float numpy array
        embeddings = []
        for i in range(0, len(text), batch_size):
            batch_text = text[i:i+batch_size]
            batch_embeddings = model.encode(batch_text, convert_to_tensor=True)
            embeddings.extend(batch_embeddings)
            # import pdb; pdb.set_trace()
        embeddings = torch.stack(embeddings)  # tensor with shape of (len(text), 384)
        assert embeddings.shape[0] == len(text)
        key = f"{paper_name}"
        all_review_embedding_dict[key] = embeddings
        # print("done")

    all_review_similarity_dict = dict()
    for key1, embeddings1 in tqdm(all_review_embedding_dict.items()):
        all_review_similarity_dict[key1] = dict()
        for key2, embeddings2 in all_review_embedding_dict.items():
            similarity = model.similarity(embeddings1, embeddings2)
            all_review_similarity_dict[key1][key2] = similarity
            # print("done")
    # TODO: optimize the above code by using util.pytorch_cos_sim
    # all_review_similarity = util.pytorch_cos_sim(torch.stack(list(all_review_embedding_dict.values())), torch.stack(list(all_review_embedding_dict.values())))  # shape of (len(all_review_embedding_dict), len(all_review_embedding_dict)), 993 * 993
    # import pdb; pdb.set_trace()
    # all_review_similarity_dict = dict()
    # # just assign the value of all_review_similarity to all_review_similarity_dict
    # for idx, (key1, _) in enumerate(all_review_embedding_dict.items()):
    #     all_review_similarity_dict[key1] = dict()
    #     for idx2, (key2, _) in enumerate(all_review_embedding_dict.keys()):
    #         all_review_similarity_dict[key1][key2] = all_review_similarity[idx][idx2]
    
    # import pdb; pdb.set_trace()
    all_review_max_sim_per_seg = dict()
    for key1, similarity_dict in tqdm(all_review_similarity_dict.items()):
        all_review_max_sim_per_seg[key1] = defaultdict(list)
        # print(key1)
        for key2, similarity in similarity_dict.items():
            # NEED TO REMOVE LATER
            if key1 == key2:
                continue
            # print(key2, similarity)
            max_similarity = torch.max(similarity, dim=1)
            for i in range(len(max_similarity.values)):
                all_review_max_sim_per_seg[key1][i].append(round(max_similarity.values[i].item(), 2))
        # print("done")

    # our custom itf-idf score
    total_number_of_paper = len(all_review_max_sim_per_seg)  # !! variable in the formula !!

    # import pdb; pdb.set_trace()
    all_idf_scores_per_seg = defaultdict(list)
    for target_paper_key in tqdm(all_review_similarity_dict.keys()):
        total_number_of_segments_in_current_review = len(all_review_similarity_dict[target_paper_key][target_paper_key])  # !! variable in the formula !!
        # import pdb; pdb.set_trace()

        for seg_idx, sim_with_other_segs in enumerate(all_review_similarity_dict[target_paper_key][target_paper_key]):
            # ===== START: compute this segment's occurence in current review === #
            # set all elements in the sim_with_other_segs less than the THRESHOLD to 0
            sim_with_other_segs[sim_with_other_segs < THRESHOLD] = 0
            # sum up the number of segments that have similarity greater than THRESHOLD
            # a variable in the formula
            this_seg_occurence_in_current_review = torch.sum(sim_with_other_segs).item() # !! variable in the formula !!
            # ===== END: compute this segment's occurence in current review === #

            # ===== START: compute the number of papers containing the segment === #
            this_seg_max_similarity_in_other_papers = all_review_max_sim_per_seg[target_paper_key][seg_idx]
            # set all elements in the this_seg_max_similarity_in_other_papers less than the THRESHOLD to 0
            this_seg_max_similarity_in_other_papers = torch.tensor(this_seg_max_similarity_in_other_papers)
            this_seg_max_similarity_in_other_papers[this_seg_max_similarity_in_other_papers < THRESHOLD] = 0
            number_of_papers_containing_the_segment = 1 + torch.sum(this_seg_max_similarity_in_other_papers).item() # !! variable in the formula !!
            # ===== END: compute the number of papers containing the segment === #
            if inverse_tf:
                # itf-idf
                # log(total number of segments in current review / this seg occurence in current review) * log(total number of papers / number of papers containing the segment)
                idf_score_this_seg = np.log(total_number_of_segments_in_current_review / this_seg_occurence_in_current_review) * np.log(total_number_of_paper / number_of_papers_containing_the_segment)  # [0, +inf)]
            else:
                # tf-idf
                # (this seg occurence in current review / total number of segments in current review) * log(total number of papers / number of papers containing the segment)
                idf_score_this_seg = (this_seg_occurence_in_current_review / total_number_of_segments_in_current_review) * np.log(total_number_of_paper / number_of_papers_containing_the_segment)  # [0, +inf)]
            all_idf_scores_per_seg[target_paper_key].append(idf_score_this_seg)


    all_idf_scores_per_paper_review = dict()
    for target_paper_key, idf_scores in all_idf_scores_per_seg.items():
        all_idf_scores_per_paper_review[target_paper_key] = np.mean(idf_scores)
    
    average_idf_score = np.mean(list(all_idf_scores_per_paper_review.values()))

    return average_idf_score, all_idf_scores_per_seg


if __name__ == "__main__":
    # predictions = [["hello world", "this is my world", "I wanna get you back my home"],["I am a student", "I am a teacher"]]
    # references = [["take you back home", "oops, I am sorry"],["you are my baby", "I am a good teacher", "how old are you"]]
    # f1, precision, recall = SentenceSemanticMetric(predictions, references)
    from sentence_transformers import SentenceTransformer, util
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    predictions_1 = ["I am a student", "godish I wanna french fries" ]
    predictions_2 = ["you love me pls", "trump knows China really well", "....."]
    references = [["I am a teacher", "I am a good teacher", "1,2,3,4"], ["I wanna be a really good teacher", "4,3,2,1"]]
    f1, precision, recall = soft_accumulate(predictions_1, references, model)
    print(f1, precision, recall)
    f1, precision, recall = soft_accumulate(predictions_2, references, model)
    print(f1, precision, recall)
     
        