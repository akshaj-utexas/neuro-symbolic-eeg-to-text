import torch
# import torch.nn.functional as F
import torch.nn.functional as F
class Aligner:
    def __init__(self, corpus_path, device="cuda"):
        self.device = device
        data = torch.load(corpus_path)
        self.words = data["words"]
        self.word_to_idx = {w.lower(): i for i, w in enumerate(self.words)}
        self.word_embs = data['embeddings'].to(device).float() 
        
        # If the corpus shape is [Vocab, 1, 512], flatten it to [Vocab, 512]
        if self.word_embs.dim() == 3:
            self.word_embs = self.word_embs.squeeze(1) 

    @torch.no_grad()
    def align(self, eeg_latent, predicted_label=None, confidence=None, top_k=15):
        # 1. Ensure EEG latent is float32 and on correct device
        vec = eeg_latent.to(self.device).float()
        
            
        # 2. Normalize for Cosine Similarity
        vec = F.normalize(vec, p=2, dim=-1)
        
        # 3. Similarity Search: [1, 512] @ [512, Vocab] -> [Vocab]
        # Using .squeeze() to ensure it's a 1D similarity vector
        sims = (vec @ self.word_embs.T).squeeze(0)
        
        # 4. Retrieve Top-K
        scores, indices = sims.topk(min(top_k, len(self.words)))
        
        return [
            {"word": self.words[i], "score": score.item()} 
            for score, i in zip(scores, indices)
        ]

def calculate_noise(dataset, device):
    """
    Computes Global Noise Centering across the dataset.
    dataset: List of dicts with 'eeg_clip_latent' shaped [1, 512]
    """
    # Simply concatenate the [1, 512] tensors into [N, 512]
    all_vecs = torch.cat([item['eeg_clip_latent'] for item in dataset], dim=0)
    # Return global mean [1, 512]
    return all_vecs.mean(dim=0, keepdim=True).to(device)
