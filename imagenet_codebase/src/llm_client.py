import os
from dotenv import load_dotenv
import openai
import torch
import anthropic
from google import genai 
from google.genai import types
import pandas as pd
from tqdm import tqdm
from together import Together

# Load variables from .env
load_dotenv()

class LLMManager:
    def __init__(self, provider="openai", model_name=None):
        self.provider = provider.lower()
        
        if self.provider == "openai":
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            self.model = model_name or "gpt-4o-mini-2024-07-18" # Using 4o for better reasoning
            
        elif self.provider == "google":
            self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
            # Default to the newest non-reasoning flash model
            self.model = model_name or "gemini-2.5-flash-lite"

        elif self.provider == "together":
            self.client = openai.OpenAI(api_key=os.getenv("TOGETHER_API_KEY"), base_url="https://api.together.xyz/v1")
            # assert error if model_name not provided
            self.model = model_name or "meta-llama/Meta-Llama-3-8B-Instruct-Lite"

    def _build_prompt_full(self, sample, ablation_version=None):
        """
        Transforms the output of the SYNAPSE Graph RAG refiner into a structured 
        relational context workspace, handling progressive semantic and topological ablations.
        """
        # 1. Unpack closed-vocabulary metadata attributes from the alignment structure
        pred_obj = sample.get("predicted_object_label", "n/a")
        pred_conf = sample.get("prediction_confidence", 0.0)
        gt_obj = sample.get("gt_object_label", "n/a")
        
        # Default extraction vectors
        prompt_words = sample.get("prompt_words", [])
        raw_prompt_words = sample.get("raw_prompt_words", [])
        relational_facts = sample.get("relational_facts", [])
        retrieved_exemplars = sample.get("retrieved_exemplars", [])

        # =========================================================================
        # 2. GRAPH CLEANED BoW: Clean BoW + Object Label (No Relational, No Exemplars)
        # =========================================================================
        if ablation_version in ["clean_bow_only", 2]:
            cleaned_words_str = ", ".join(prompt_words)
            prompt = f"""You are an advanced neural decoding translation engine. You are provided with a denoised, common-sense validated Bag-of-Words extracted from human EEG signals. Your task is to synthesize these primitives into a single natural description.

[Denoised Brain-Signal Keywords]
[{cleaned_words_str}]

[Target Dominant Signal Context]
- Primary Classification Target: '{pred_obj}' (Model Confidence: {pred_conf:.4f})

Instructions:
1. Synthesize these primitives into exactly ONE clear, fluent English description (8-20 words).
2. Prioritize concepts verified by the brain-signal keywords.
3. Do NOT include annotations, prefix strings, quotes, or conversational meta-commentary. Output ONLY the raw caption string text.

Output:"""
            return prompt

        # =========================================================================
        # 3. CONTEXT RESILIENCE: Clean BoW + Relational Stuff (Facts & Exemplars), NO Object Label
        # =========================================================================
        elif ablation_version in ["no_object_label", 3]:
            cleaned_words_str = ", ".join(prompt_words)
            prompt = f"""You are an advanced neural decoding translation engine. You are provided with a denoised, common-sense validated Bag-of-Words extracted from human EEG signals, along with topological world constraints and structural sentence guidelines. Your task is to synthesize these primitives into a single natural description.

[Structural Layout Guides from Training Set]
{retrieved_exemplars}

[Denoised Brain-Signal Keywords]
[{cleaned_words_str}]

[Topological Common-Sense Relations]
{relational_facts}

Instructions:
1. Synthesize these primitives into exactly ONE clear, fluent English description (8-20 words).
2. Prioritize concepts verified by both the brain-signal keywords and the common-sense relational constraints.
3. Do NOT include annotations, prefix strings, quotes, or conversational meta-commentary. Output ONLY the raw caption string text.

Output:"""
            return prompt


        # =========================================================================
        # NO RELATIONAL VERSION: Clean BoW + Object Label, NO Facts, but RETAIN EXEMPLARS (Structural Templates Only)
        # =========================================================================
        elif ablation_version in ["no_exemplar", 4]:
            cleaned_words_str = ", ".join(prompt_words)
            prompt = f"""You are an advanced neural decoding translation engine. You are provided with a denoised, common-sense validated Bag-of-Words extracted from human EEG signals, along with topological world constraints. Your task is to synthesize these primitives into a single natural description.

[Denoised Brain-Signal Keywords]
[{cleaned_words_str}]

[Topological Common-Sense Relations]
{relational_facts}

[Target Dominant Signal Context]
- Primary Classification Target: '{pred_obj}' (Model Confidence: {pred_conf:.4f})

Instructions:
1. Synthesize these primitives into exactly ONE clear, fluent English description (8-20 words).
2. Prioritize concepts verified by both the brain-signal keywords and the common-sense relational constraints.
3. Do NOT include annotations, prefix strings, quotes, or conversational meta-commentary. Output ONLY the raw caption string text.

Output:"""
            return prompt

        # =========================================================================
        # NO EXEMPLAR VERSION: Clean BoW + Relational Stuff (Facts Only), NO Exemplars, Retain Object Label
        # =========================================================================
        elif ablation_version in ["no_facts", 4]:
            cleaned_words_str = ", ".join(prompt_words)
            prompt = f"""You are an advanced neural decoding translation engine. You are provided with a denoised, common-sense validated Bag-of-Words extracted from human EEG signals, along with structural sentence guidelines. Your task is to synthesize these primitives into a single natural description.

[Denoised Brain-Signal Keywords]
[{cleaned_words_str}]

[Structural Layout Guides from Training Set]
{retrieved_exemplars}

[Target Dominant Signal Context]
- Primary Classification Target: '{pred_obj}' (Model Confidence: {pred_conf:.4f})

Instructions:
1. Synthesize these primitives into exactly ONE clear, fluent English description (8-20 words).
2. Prioritize concepts verified by both the brain-signal keywords and the common-sense relational constraints.
3. Do NOT include annotations, prefix strings, quotes, or conversational meta-commentary. Output ONLY the raw caption string text.

Output:"""
            return prompt

        # =========================================================================
        # MAIN PRODUCTION VERSION: Full Graph RAG Relational Augmentation Prompt
        # =========================================================================
        else:
            cleaned_words_str = ", ".join(prompt_words)
            prompt = f"""You are an advanced neural decoding translation engine. You are provided with a denoised, common-sense validated Bag-of-Words extracted from human EEG signals, along with topological world constraints and structural sentence guidelines. Your task is to synthesize these primitives into a single natural description.

[Structural Layout Guides from Training Set]
{retrieved_exemplars}

[Denoised Brain-Signal Keywords]
[{cleaned_words_str}]

[Topological Common-Sense Relations]
{relational_facts}

[Target Dominant Signal Context]
- Primary Classification Target: '{pred_obj}' (Model Confidence: {pred_conf:.4f})

Instructions:
1. Synthesize these primitives into exactly ONE clear, fluent English description (8-20 words).
2. Prioritize concepts verified by both the brain-signal keywords and the common-sense relational constraints.
3. Do NOT include annotations, prefix strings, quotes, or conversational meta-commentary. Output ONLY the raw caption string text.

Output:"""
            return prompt





    def generate(self, sample, ablation_version=None):
        prompt = self._build_prompt_full(sample, ablation_version=ablation_version)
        # print(f"Generated prompt for {self.provider}:\n{prompt}\n")
        
        try:
            if self.provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2 # Keep it grounded
                )
                return response.choices[0].message.content.strip().replace('"', '')

            elif self.provider == "google":
                # Refactored: Using the stateless models.generate_content pattern
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    # Optional: Explicitly set thinking to 'minimal' for non-reasoning models
                    config=types.GenerateContentConfig(
                        temperature=0.2,
                        thinking_config=types.ThinkingConfig(include_thoughts=False)
                    )
                )
                return response.text.strip()

            elif self.provider == "together":
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2 # Keep it grounded
                )
                return response.choices[0].message.content.strip().replace('"', '')

        except Exception as e:
            return f"API ERROR ({self.provider}): {str(e)}"

    def run_decoding_experiment(self, input_path, output_path, num_samples=None, ablation_version=None):
        """
        Executes batch generation over the dataset and exports evaluation metrics.
        Preserves 100% of the input dictionary schema and appends the generated output.
        """

        # if the output path does not exist, create the directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dataset = torch.load(input_path, weights_only=False)
        test_subset = dataset[:num_samples] if num_samples else dataset
        results = []

        print(f"Decoding {len(test_subset)} samples via {self.provider} ({self.model})...")
        
        for item in tqdm(test_subset):
            generated = self.generate(item, ablation_version=ablation_version)

            # Take a full copy of the original dictionary to guarantee NO data loss
            output_record = item.copy()
            # Append the newly synthesized text sequence key
            output_record["generated_caption"] = generated
            
            results.append(output_record)

        # Save the complete object array (.pt) containing your raw tensors, bow arrays, and facts
        torch.save(results, output_path)
        
        # Flatten structural sub-objects cleanly for the companion CSV log file
        csv_flattened_records = []
        for rec in results:
            flat_rec = {}
            for k, v in rec.items():
                if k == "refined_latent":
                    continue  # Omit multi-page raw hidden float tensors from spreadsheet columns
                elif isinstance(v, list) and all(isinstance(x, str) for x in v):
                    flat_rec[k] = ", ".join(v)
                elif k == "retrieved_exemplars" and isinstance(v, list):
                    flat_rec[k] = " | ".join([f"({e.get('similarity',0):.3f}) {e.get('caption','')}" for e in v])
                else:
                    flat_rec[k] = v
            csv_flattened_records.append(flat_rec)

        pd.DataFrame(csv_flattened_records).to_csv(output_path.replace('.pt', '.csv'), index=False)
        print(f"Successfully exported data to: {output_path} and associated CSV sheet.")
        return results

