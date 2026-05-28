import argparse
import sys
import os
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import pandas as pd

# Path injection
sys.path.append(os.getcwd())
from src.llm_client import LLMManager
from src.models import SimilarityRefiner
from src.trainer import Stage1_5Dataset, run_training
from src.metrics import evaluate_and_save_metrics
from src.encoders import DATASET_REGISTRY

def parse_args():

    parser = argparse.ArgumentParser(description="SYNAPSE Pipeline")
    # Paths and Data
    parser.add_argument("--dataset", type=str, default="imagenet_eeg_test", help="Path to the EEG dataset .pth file")
    parser.add_argument("--vocab_path", type=str, default="checkpoints/imagenet_train_corpus.pt", help="Path to the encoded word corpus")
    parser.add_argument("--output_dir", type=str, default="./results", help="Where to save outputs")
    parser.add_argument("--batch_size", type=int, default=64)
    
    # Mode Selection
    parser.add_argument("--mode", type=str, choices=[ "train", "inference"], 
                        help="train: Train MLP | inference: Use trained MLP")
    
    # MLP Config
    parser.add_argument("--loss", type=str, choices=["bce", "focal", "contrastive"], default="focal")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to .pth weights for inference")
    parser.add_argument("--eeg_encoder", type=str, default="channelnet")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # LLM & Eval Logic
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--skip_llm", action="store_true")
    parser.add_argument("--model", type=str, choices=["qwen2.5", "llama", "chatgpt", "gemini"], default="qwen2.5")
    parser.add_argument("--skip_eval", action="store_true")

    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    dataset_name = os.path.basename(args.dataset).replace('.pth', '')
    
    # 1. EEG Encoding Step (Assumes raw EEG -> CLIP latents)
    # This logic checks for existing latents to save time/compute
    clip_latents_path = "data/imagenet_eeg_test_eeg_latents.pt"
    
    if not os.path.exists(clip_latents_path):
        print(f"--- Step 1: Encoding EEG via {args.eeg_encoder} ---")
        from src.encoders import process_channelnet
        process_channelnet(args.dataset, clip_latents_path, args.device, args.batch_size)
    else:
        print(f"--- Found existing latents at {clip_latents_path} ---")

    # 2. Alignment Logic (The Core Switch)
    final_alignment_path = ""

    dataset_path = DATASET_REGISTRY.get(args.dataset, None)
    print(f"Dataset path for training: {dataset_path}")

    if args.mode == "train":
        print(f"--- Mode: Training SimilarityRefiner ---")
        
        # 1. Load vocabulary embeddings first to initialize the model
        print(f"Loading vocabulary from {args.vocab_path}...")
        vocab_info = torch.load(args.vocab_path, weights_only=False)
        train_ds = Stage1_5Dataset(clip_latents_path, args.vocab_path)
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
        
        # 3. Initialize Model with vocab_embeddings
        # Note: input_dim=512 (EEG), hidden_dim=1024
        model = SimilarityRefiner(
            vocab_embeddings=vocab_info["embeddings"], 
            input_dim=512, 
            hidden_dim=1024
        )
        
        save_name = f"similarity_refiner_{args.epochs}eps_{args.loss}.pth"
        save_path = os.path.join("checkpoints", save_name)
        
        # 4. Execute Multi-Label Training
        run_training(model, loader, args.device, args.epochs, args.loss, save_path)
        print(f"Training complete. Model saved to {save_path}")
        return

    # Inside run_pipeline.py -> main()
    elif args.mode == "inference":
        print(f"--- SYNAPSE Inference using {args.checkpoint} ---")
        if not args.checkpoint or not os.path.exists(args.checkpoint):
            raise ValueError("Inference mode requires a valid --checkpoint path.")

        vocab_info = torch.load(args.vocab_path)
        model = SimilarityRefiner(vocab_info["embeddings"])
        model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
        model.to(args.device).eval()

        # =================================================================
        # SYNAPSE COMPONENT INITIALIZATION
        # =================================================================
        from src.graph_filter import GraphRAGRefiner
        from src.exemplar_retriever import TrainingExemplarRetriever

        # Define path configurations pointing to local database structures
        db_path = getattr(args, "conceptnet_db_path", "data/conceptnet_local.db")
        train_data_path = getattr(args, "train_dataset_path", "data/imagenet_eeg_train_eeg_latents.pt")

        print("SYNAPSE Initializing Graph RAG Filter...")
        graph_rag = GraphRAGRefiner(db_path=db_path)
        print("SYNAPSE Initializing Cross-Modal Exemplar Indexer...")
        exemplar_retriever = TrainingExemplarRetriever(training_data_path=train_data_path, device=args.device)
        # =================================================================
        
        latent_dataset = torch.load(clip_latents_path)
        aligned_results = []

        for item in tqdm(latent_dataset, desc="MLP Mapping"):
            eeg_vec = item['eeg_clip_latent'].to(args.device).float()
            if eeg_vec.dim() == 1: eeg_vec = eeg_vec.unsqueeze(0)

            with torch.no_grad():
                logits, refined_latent = model(eeg_vec)
                probs = torch.sigmoid(logits).squeeze()
            
            scores, indices = probs.topk(min(args.top_k, len(vocab_info["words"])))
            bow = [{"word": vocab_info["words"][idx], "score": s.item()} for s, idx in zip(scores, indices)]

            raw_prompt_words = [w['word'] for w in bow]

            # =================================================================
            # INTERCEPT & ENHANCE WITH GRAPH RAG INTERPOLATION
            # =================================================================
            # 1. Purge neural signal drift and draw topological facts
            pruned_prompt_words, relational_facts = graph_rag.retrieve_and_filter_subgraph(
                seed_words=raw_prompt_words, 
                top_n_facts=5
            )

            # 2. Extract context training samples using the aligned latent coordinate
            retrieved_exemplars = exemplar_retriever.retrieve_top_exemplars(
                refined_latent=eeg_vec, 
                top_n=2
            )
            # =================================================================

            aligned_results.append({
                "subject": item.get("subject"),
                "gt_object_label": item.get("object_label", ""),
                "gt_caption": item.get("caption", ""),
                "predicted_object_label": item.get("predicted_object_label", "n/a"),
                "prediction_confidence": item.get("prediction_confidence", 0.0),
                "initial_raw_bow": bow,
                "raw_prompt_words": raw_prompt_words,
                
                # Augmented structural attributes saved for the generation script
                "prompt_words": pruned_prompt_words,      # Topological filtered output
                "relational_facts": relational_facts,    # Context statements
                "retrieved_exemplars": retrieved_exemplars,# Syntactic sentence references
                "refined_latent": refined_latent.cpu()
            })
        
        final_alignment_path = os.path.join(args.output_dir, f"{dataset_name}_top_15_0_words_graph_rag_aligned.pt")
        torch.save(aligned_results, final_alignment_path)
        print(f"[Success] SYNAPSE Graph RAG alignment records stored at: {final_alignment_path}")

    final_alignment_path = "results/imagenet_eeg_test_top_15_3_words_graph_rag_aligned.pt"
    # 3. Semantic Decoding via LLM
    if not args.skip_llm and final_alignment_path:
        print(f"--- Step 3: LLM Caption Generation Loop ---")

        model = args.model
        
        # so now I only want to write the model_name, and the directory path for outputs, and the rest should be handled by the registry and the LLMManager class
        if model == "qwen2.5":
            provider = "together"
            model_name = "Qwen/Qwen2.5-7B-Instruct-Turbo"
        elif model == "llama":
            provider = "together"
            model_name = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"
        elif model == "chatgpt":
            provider = "openai"
            model_name = "gpt-4o-mini-2024-07-18"
        elif model == "gemini":
            provider = "google"
            model_name = "gemini-2.5-flash-lite"
        else:
            raise ValueError(f"Model {model} not found")

     
        llm_manager = LLMManager(provider=provider, model_name=model_name)
        dir_path = model

        # Define the 4 target states exactly matching your prompt logic flow
        experiment_matrix = [
            "clean_bow_only",   
            "no_object_label",  
            "no_examplar",
            "no_facts",
            None               
        ]
        
        for current_ablation in experiment_matrix:
            label_tag = current_ablation if current_ablation else "full_production"
            print(f"\n>>> Running Target Experiment Configuration: {label_tag} <<<")
            
            final_pt_output = f"results/{dir_path}/{dir_path}_top_k_15_3_words_graph_rag_{label_tag}.pt"
            final_csv_path = final_pt_output.replace(".pt", ".csv")
            
            # Batch execute through the pipeline
            llm_manager.run_decoding_experiment(
                input_path=final_alignment_path,
                output_path=final_pt_output,
                ablation_version=current_ablation
            )
            
            # Run Evaluation automatically on the current loop's spreadsheet
            if not args.skip_eval and os.path.exists(final_csv_path):
                print(f"--- Running Metrics for Layout: {label_tag} ---")
                evaluate_and_save_metrics(final_csv_path, output_dir=os.path.join(args.output_dir, label_tag))

if __name__ == "__main__":
    main()
