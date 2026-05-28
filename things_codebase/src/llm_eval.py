import os
import argparse
import json
import ast
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from pydantic import BaseModel
import instructor
from dotenv import load_dotenv
load_dotenv()

# --- Exact Directory Layout Configurations ---
DEFAULT_RESULTS_DIR = "output/llm-prompts"
DECODER_FOLDERS = ["chatgpt", "gemini", "llama", "qwen2.5"]
ABLATIONS = ["full", "no_exemplars", "sense_baseline"]

# Pydantic schema for instructor validation constraints
class Evaluation(BaseModel):
    fluency: int
    adequacy: int

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = instructor.from_openai(OpenAI(api_key=OPENAI_API_KEY))

# Prompt template definition
PROMPT_TEMPLATE = """You are a helpful language evaluator who can evaluate
input sentence2 and provide an evaluation of its fluency with a
likert scale rating of 1-5, 5 being highly fluent.
You will also have to compare two sentences and judge how adequate
is input sentence 2 with respect to input sentence 1, again with a likert scale rating of 1-5, 5 being highly adequate.
Here are the sentences:
input_sentence1: {input_sentence1}
input_sentence2: {input_sentence2}"""

def eval_single(index, row, client_inst):
    """Worker task executing individual sample pair generations over GPT-5."""
    try:
        input_sentence1 = row["caption"]
        input_sentence2 = row["generated_caption"]

        final_prompt = PROMPT_TEMPLATE.format(
            input_sentence1=input_sentence1, 
            input_sentence2=input_sentence2
        )

        eval_info = client_inst.chat.completions.create(
            model="gpt-5-mini",  
            response_model=Evaluation,
            messages=[{"role": "user", "content": final_prompt}]
        )
        return index, eval_info.model_dump()
    except Exception as e:
        print(f"❌ Error at index {index}: {e}")
        return index, None

def process_and_score_file(csv_input_path, metrics_json_path):
    """
    Handles row-by-row tracking, checkpoints intermediate evaluation outputs,
    and APPENDS finalized mean metric values into your existing ablation metrics JSON files.
    """
    csv_eval_path = csv_input_path.replace(".csv", "_eval.csv")
    
    # Auto-resume checkpoint hook
    if os.path.exists(csv_eval_path):
        df = pd.read_csv(csv_eval_path)
    else:
        df = pd.read_csv(csv_input_path)
    
    if 'eval' not in df.columns:
        df['eval'] = None

    # Filter out records that are already calculated
    rows_to_process = df[df['eval'].isna() | (df['eval'] == "")]

    if rows_to_process.empty:
        print(f"   -> All rows in this profile are already processed.")
    else:
        num_threads = 64  
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [
                executor.submit(eval_single, i, r, client) 
                for i, r in rows_to_process.iterrows()
            ]

            for future in tqdm(as_completed(futures), total=len(futures), desc="🤖 GPT-5 Judging"):
                index, result = future.result()
                if result:
                    df.at[index, 'eval'] = str(result)
                    
                    # Intermittent batch flushing tracking pass
                    if index % 10 == 0:
                        df.to_csv(csv_eval_path, index=False)
        
        df.to_csv(csv_eval_path, index=False)

    # Parse results back into numeric vectors to calculate means
    print("   -> Computing final judgment metrics...")
    eval_dicts = df['eval'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    valid_evals = eval_dicts.dropna()
    
    if not valid_evals.empty:
        avg_fluency = valid_evals.apply(lambda x: x['fluency']).mean()
        avg_adequacy = valid_evals.apply(lambda x: x['adequacy']).mean()
    else:
        avg_fluency, avg_adequacy = 0.0, 0.0

    print(f"   📈 New Metrics: Fluency={avg_fluency:.3f}, Adequacy={avg_adequacy:.3f}")

    # SAFE PASS: Load existing file state and append rather than overwrite
    json_data = {}
    if os.path.exists(metrics_json_path):
        try:
            with open(metrics_json_path, "r") as f:
                json_data = json.load(f)
                print(f"   📂 Existing JSON loaded with {len(json_data)} keys. Appending new metrics...")
        except Exception as e:
            print(f"   ⚠️ Error reading existing JSON ({e}). Initializing empty structure.")
    else:
        print(f"   ℹ️ JSON file did not exist yet. Creating a new entry.")
        os.makedirs(os.path.dirname(metrics_json_path), exist_ok=True)

    # Append metrics into the dictionary structure securely
    json_data["Mean GPT-5 Fluency"] = round(avg_fluency, 3)
    json_data["Mean GPT-5 Adequacy"] = round(avg_adequacy, 3)

    with open(metrics_json_path, "w") as f:
        json.dump(json_data, f, indent=4)
    print(f"   💾 Metrics successfully saved to: {metrics_json_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Run GPT-5 evaluation on LLM CSV outputs."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default=DEFAULT_RESULTS_DIR,
        help="Directory containing test_from_train_{model}_{ablation}.csv files.",
    )
    args = parser.parse_args()

    print("🚀 Initiating GPT-5 Language Judgments over Experiment Grid...\n")

    for model in DECODER_FOLDERS:
        for ablation in ABLATIONS:
            csv_name = f"test_from_train_{model}_{ablation}.csv"
            csv_source_path = os.path.join(args.results_dir, csv_name)
            if not os.path.exists(csv_source_path):
                print(f"⚠️ Missing {csv_source_path}, skipping.")
                continue

            metrics_json_target = os.path.join(
                args.results_dir,
                "metrics",
                f"averaged_test_from_train_{model}_{ablation}.json",
            )

            print(f"\n🎯 Processing: {csv_name}")
            print(f"   Target JSON Path: {metrics_json_target}")
            process_and_score_file(csv_source_path, metrics_json_target)

    print("\n Evaluation execution completed successfully.")

if __name__ == "__main__":
    main()
