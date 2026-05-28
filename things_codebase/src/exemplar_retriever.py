import torch
import torch.nn.functional as F

class TrainingExemplarRetriever:
    def __init__(self, training_data_path, device="cuda"):
        """
        Indexes pre-computed training split neural latents and descriptions for fast
        cross-modal metric search.
        """
        self.device = device
        # Load the base training split dataset
        train_dataset = torch.load(training_data_path, map_location="cpu", weights_only=False)
        
        self.captions = []
        embeddings_list = []
        
        if isinstance(train_dataset, dict):
            captions = train_dataset.get("caption")
            embeddings = train_dataset.get("eeg_clip_latent")
            if captions is None or embeddings is None:
                raise ValueError(
                    f"Training dataset missing 'caption' or 'eeg_clip_latent' in {training_data_path}"
                )
            if torch.is_tensor(embeddings):
                embeddings = embeddings.cpu()
            else:
                embeddings = torch.from_numpy(embeddings)
            for idx in range(len(captions)):
                self.captions.append(str(captions[idx]))
                embeddings_list.append(embeddings[idx].view(1, 512).float())
        else:
            for item in train_dataset:
                if 'caption' in item and 'eeg_clip_latent' in item:
                    self.captions.append(item['caption'])
                    # Re-shape from [512] flat array to explicit row vectors [1, 512]
                    embeddings_list.append(item['eeg_clip_latent'].view(1, 512).float())
                
        if not embeddings_list:
            raise ValueError(f"Could not locate structured latents or captions in {training_data_path}")
            
        # Concatenate entries into an index matrix: [N_train, 512]
        self.train_embeddings = torch.cat(embeddings_list, dim=0).to(device)
        # Normalize the tensor rows up front for optimized vector calculations
        self.train_embeddings = F.normalize(self.train_embeddings, p=2, dim=-1)

    def retrieve_top_exemplars(self, refined_latent, top_n=2):
        """
        Performs batch cosine matrix multiplication to retrieve nearest neighbors.
        Query Shape: [1, 512]
        """
        # Ensure query is properly isolated and normalized: [1, 512]
        z_query = refined_latent.to(self.device).float().view(1, 512)
        z_query = F.normalize(z_query, p=2, dim=-1)
        
        # [1, 512] @ [512, N_train] -> [1, N_train]
        similarities = torch.matmul(z_query, self.train_embeddings.t()).squeeze(0)
        
        scores, indices = similarities.topk(min(top_n, len(self.captions)))
        
        retrieved_samples = []
        for score, idx in zip(scores, indices):
            retrieved_samples.append({
                "caption": self.captions[idx.item()],
                "similarity": score.item()
            })
        return retrieved_samples
