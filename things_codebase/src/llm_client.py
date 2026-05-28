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
        Builds prompts for pruned BoW + optional relational facts.
        """
        pruned_bow = sample.get("pruned_bow") or sample.get("prompt_words") or []
        relational_facts = sample.get("relational_facts", "")
        exemplars = sample.get("retrieved_exemplars", [])

        words_str = ", ".join(pruned_bow)
        if isinstance(relational_facts, list):
            facts_str = "; ".join([str(fact) for fact in relational_facts])
        else:
            facts_str = str(relational_facts) if relational_facts else ""

        exemplars_str = ""
        if isinstance(exemplars, list):
            exemplars_str = "; ".join([str(ex) for ex in exemplars if ex])
        else:
            exemplars_str = str(exemplars)

        if ablation_version in ["no_exemplars"]:
            prompt = f"""You are an advanced neural decoding translation engine. You are provided with a denoised, common-sense validated Bag-of-Words extracted from human EEG signals, along with topological world constraints. Your task is to synthesize these primitives into a single natural description.

[Denoised Brain-Signal Keywords]
[{words_str}]

[Topological Common-Sense Relations]
{facts_str}

Instructions:
1. Synthesize these primitives into exactly ONE clear, fluent English description (8-20 words).
2. Prioritize concepts verified by both the brain-signal keywords and the common-sense relational constraints.
3. Do NOT include annotations, prefix strings, quotes, or conversational meta-commentary. Output ONLY the raw caption string text.

Output:"""
            return prompt

        prompt = f"""You are an advanced neural decoding translation engine. You are provided with a denoised, common-sense validated Bag-of-Words extracted from human EEG signals, along with topological world constraints and exemplar captions from similar samples from training data. Your task is to synthesize these primitives into a single natural description.

[Denoised Brain-Signal Keywords]
[{words_str}]

[Topological Common-Sense Relations]
{facts_str}

[Retrieved Exemplars]
{exemplars_str}

Instructions:
1. Synthesize these primitives into exactly ONE clear, fluent English description (8-20 words).
2. Prioritize concepts verified by both the brain-signal keywords and the common-sense relational constraints.
3. Do NOT include annotations, prefix strings, quotes, or conversational meta-commentary. Output ONLY the raw caption string text.

        Output:"""
        return prompt

    def _build_prompt_sense_baseline(self, sample):
        top_words = sample.get("top_words") or []
        raw_bow = sample.get("raw_bow") or []

        scored_tokens = []
        if isinstance(top_words, list) and top_words:
            for item in top_words:
                if isinstance(item, dict):
                    word = item.get("word", "")
                    score = item.get("score")
                    if word:
                        if score is None:
                            scored_tokens.append(word)
                        else:
                            scored_tokens.append(f"{word}: {float(score):.3f}")
                else:
                    scored_tokens.append(str(item))
        else:
            scored_tokens = [str(w) for w in raw_bow]

        words_str = ", ".join(scored_tokens)
        prompt = f"""The following template is used for the ablation study, where the language model reconstructs the caption using only the EEG-derived semantic anchors.

You are given a noisy bag-of-words (BoW). BoW will be accompanied with numbers, the numbers with BoW are cosine similarities of the words to our embedding.

Your goal is to regenerate the most likely original image caption.

Instructions:
- Use the similarity scores to infer which words are relevant.
- Ignore or remove garbage, irrelevant, contradictory, or low-signal words.
- Use only a small, coherent subset of the BoW.
- Do NOT invent new objects not supported by the high-similarity words.

Output:
Return ONLY one natural-language caption (8-20 words). No explanations, no lists, no formatting.

Input:
BoW tokens with scores:
{words_str}"""
        return prompt

    def generate(self, sample, ablation_version=None):
        if ablation_version == "sense_baseline":
            prompt = self._build_prompt_sense_baseline(sample)
        else:
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
