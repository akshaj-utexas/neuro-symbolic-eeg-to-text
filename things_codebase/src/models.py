import torch
import torch.nn as nn
import torch.nn.functional as F

class SimilarityRefiner(nn.Module):
    def __init__(self, vocab_embeddings, input_dim=512, hidden_dim=1024, use_scaling=True):
        super().__init__()
        # Flatten vocab embeddings if needed [Vocab, 1, 512] -> [Vocab, 512]
        if vocab_embeddings.dim() == 3:
            vocab_embeddings = vocab_embeddings.squeeze(1)
            
        self.use_scaling = use_scaling
        text_dim = vocab_embeddings.shape[1]
        
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, text_dim)
        )
        
        self.register_buffer("vocab_matrix", F.normalize(vocab_embeddings, p=2, dim=-1))
        
        if self.use_scaling:
            self.log_tau = nn.Parameter(torch.tensor(2.5)) 

    def forward(self, x):
        refined_latent = self.projector(x)
        refined_latent = F.normalize(refined_latent, p=2, dim=-1)
        
        vocab_matrix = self.vocab_matrix.to(refined_latent.dtype)
        logits = torch.matmul(refined_latent, vocab_matrix.t())

        if self.use_scaling:
            scale = torch.clamp(self.log_tau.exp(), max=100.0)
            logits = logits * scale
            
        return logits, refined_latent

# --- Loss Functions ---
class MultiLabelContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.tau = temperature

    def forward(self, logits, targets):
        logits = logits / self.tau
        log_probs = F.log_softmax(logits, dim=-1)
        pos_log_probs = (log_probs * targets).sum(dim=-1)
        num_pos = targets.sum(dim=-1).clamp(min=1)
        return - (pos_log_probs / num_pos).mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        p_t = (targets * probs) + ((1 - targets) * (1 - probs))
        focal_weight = (1 - p_t) ** self.gamma
        alpha_weight = (targets * self.alpha) + ((1 - targets) * (1 - self.alpha))
        return (alpha_weight * focal_weight * bce_loss).mean()
