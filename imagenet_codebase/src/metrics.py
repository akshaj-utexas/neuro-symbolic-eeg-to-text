import pandas as pd
import numpy as np
import nltk
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import single_meteor_score
from nltk.tokenize import word_tokenize
from rouge import Rouge
import bert_score
import statistics
import re
import os
import tqdm
import json

# Ensure NLTK resources are available
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

def read_result_csv(file_path):
    # Load the CSV. We no longer drop the first column 
    # because 'subject' is now the first column.
    result_df = pd.read_csv(file_path)
    return result_df

def compute_bleu(reference, candidate):
    reference = [[ref.split()] for ref in reference]
    candidate = [cand.split() for cand in candidate]
    smoothing_function = SmoothingFunction().method4
    bleu_scores = [corpus_bleu([ref], [cand], smoothing_function=smoothing_function) for ref, cand in zip(reference, candidate)]
    return round(np.mean(bleu_scores), 3), round(np.std(bleu_scores), 3)

def compute_bleu_unigram(reference, candidate):
    reference = [[ref.split()] for ref in reference]
    candidate = [cand.split() for cand in candidate]
    smoothing_function = SmoothingFunction().method4
    weights = (1, 0, 0, 0)
    bleu_scores = [corpus_bleu([ref], [cand], smoothing_function=smoothing_function, weights=weights) for ref, cand in zip(reference, candidate)]
    return round(np.mean(bleu_scores), 3), round(np.std(bleu_scores), 3)

def compute_rouge(reference, candidate):
    rouge = Rouge()
    rouge_scores = [rouge.get_scores(cand, ref, avg=True) for ref, cand in zip(reference, candidate)]
    
    r1 = [score['rouge-1']['f'] for score in rouge_scores]
    r2 = [score['rouge-2']['f'] for score in rouge_scores]
    rl = [score['rouge-l']['f'] for score in rouge_scores]

    return (round(np.mean(r1), 3), round(np.mean(r2), 3), round(np.mean(rl), 3), 
            round(np.std(r1), 3), round(np.std(r2), 3), round(np.std(rl), 3))

def compute_bert_score(reference, candidate):
    bert_p, bert_r, bert_f1 = bert_score.score(candidate, reference, lang="en", verbose=False)
    return round(bert_f1.mean().item(), 3), round(bert_f1.std().item(), 3)

def compute_meteor_scores(reference, candidate):
    tokenized_candidates = [word_tokenize(c.replace("<s>", "").replace("</s>", "").strip()) for c in candidate]
    tokenized_references = [word_tokenize(r) for r in reference]
    meteor_scores = [single_meteor_score(ref, cand) for ref, cand in zip(tokenized_references, tokenized_candidates)]
    return round(np.mean(meteor_scores), 3), round(statistics.stdev(meteor_scores), 3)

def clean_text(text):
    regex = r"[^a-zA-Z0-9.,!?;:'\"()\[\]{}\-\s]"
    cleaned_text = re.sub(regex, '', str(text))
    lines = re.split(r'(?<=[.!?]) +', cleaned_text)
    unique_lines = []
    seen = set()
    for line in lines:
        cleaned_line = re.sub(r'\s+', ' ', line).strip()
        if cleaned_line not in seen:
            seen.add(cleaned_line)
            unique_lines.append(cleaned_line)
    return ' '.join(unique_lines[:1])

def cleanup_pred_captions(predicted_captions):
    predicted_captions = predicted_captions.fillna('')
    clean_captions = []
    for caption in predicted_captions:
        clean_caption = clean_text(caption) if caption.strip() else "No response."
        if not clean_caption.strip():
            clean_caption = "No response."
        clean_captions.append(clean_caption)
    return clean_captions

def run(csv_path):
    results = {}
    result_df = read_result_csv(csv_path)

    # --- ADJUSTED COLUMN MAPPINGS ---
    expected_captions = result_df['gt_caption']
    predicted_captions = result_df['generated_caption']
    # Note: image_paths are not in the new CSV, so we omit them.
    # expected_object_classes = result_df['gt_object']
    # predicted_object_classes = result_df['predicted_object']

    predicted_captions = cleanup_pred_captions(predicted_captions)
    references = expected_captions.tolist()
    candidates = predicted_captions

    for i, cand in enumerate(candidates):
        if len(cand) <= 1:
            candidates[i] = "No response"

    # BLEU Scores
    mean_b, std_b = compute_bleu(references, candidates)
    results["Mean BLEU Score"] = mean_b
    results["SD BLEU Score"] = std_b

    mean_bu, std_bu = compute_bleu_unigram(references, candidates)
    results["Mean BLEU Unigram Score"] = mean_bu
    results["SD BLEU Unigram Score"] = std_bu

    # ROUGE Scores
    r1, r2, rl, r1_s, r2_s, rl_s = compute_rouge(references, candidates)
    results["Mean ROUGE-1"] = r1
    results["SD ROUGE-1"] = r1_s
    results["Mean ROUGE-2"] = r2
    results["SD ROUGE-2"] = r2_s
    results["Mean ROUGE-l"] = rl
    results["SD ROUGE-l"] = rl_s

    # METEOR Score
    mean_m, std_m = compute_meteor_scores(references, candidates)
    results["Mean Meteor Score"] = mean_m
    results["SD Meteor Score"] = std_m

    # BERT Score
    bert_m, bert_s = compute_bert_score(references, candidates)
    results["Mean BERTScore"] = bert_m
    results["SD BERTScore"] = bert_s
    
    return results 

import os
import json
import pandas as pd

def evaluate_and_save_metrics(csv_path, output_dir="results"):
    """
    Processes a single CSV file, calculates metrics, and saves results to CSV/JSON.
    """
    if not os.path.exists(csv_path):
        print(f"Could not find the file: {csv_path}. Skipping...")
        return

    print(f"Running evaluation for: {csv_path}")
    # Standardize output to the pipeline's output directory
    eval_dir = os.path.join(output_dir, "metrics")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Generate dynamic filenames based on the input filename
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    csv_output = os.path.join(eval_dir, f"metrics_{base_name}.csv")
    json_output = os.path.join(eval_dir, f"averaged_{base_name}.json")

    try:
        # 1. Run the metrics calculation
        results = run(csv_path=csv_path)

        averaged_results_dict = {}
        # 2. Process and format results
        for key, value in results.items():
            if "Mean" in key:
                # Store as percentage and update original results for the DataFrame
                formatted_val = round(value * 100, 2)
                averaged_results_dict[key] = formatted_val
                results[key] = formatted_val
            else:
                averaged_results_dict[key] = value

        # 3. Display and Save to CSV
        results_df = pd.DataFrame([results])
        print(f"\n--- EVALUATION RESULTS: {base_name} ---")
        print(results_df.to_string(index=False))
        
        results_df.to_csv(csv_output, index=False)

        # 4. Save to JSON
        with open(json_output, "w") as f:
            json.dump(averaged_results_dict, f, indent=4)
            
        print(f"Saved results to '{csv_output}' and '{json_output}'\n")
        
    except Exception as e:
        print(f"Error during processing {csv_path}: {e}")