import torch
import nltk
from torch.utils.data import Dataset, DataLoader
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from tqdm import tqdm
import os

import torch
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import torch.nn.functional as F

class MultiHotEncoder:
    def __init__(self, vocab_list):
        """
        Initializes the encoder with necessary NLTK resources.
        """
        # Mandatory downloads for tokenization and lemmatization
        nltk.download('punkt_tab')             # Fixes the current LookupError [cite: 216]
        nltk.download('averaged_perceptron_tagger_eng')
        nltk.download('stopwords')
        nltk.download('wordnet')

        self.vocab = vocab_list
        self.word_to_idx = {w: i for i, w in enumerate(vocab_list)}
        self.lemmatizer = WordNetLemmatizer()
        self.vocab_size = len(vocab_list)
        self.stop_words = set(stopwords.words('english'))

    def encode(self, caption):
        # [vocab_size]
        target = torch.zeros(self.vocab_size)

        tokens = nltk.word_tokenize(caption.lower())
        tagged = nltk.pos_tag(tokens)

        for word, tag in tagged:
            if word.isalpha() and word not in self.stop_words:
                cat = None
                if tag.startswith('NN'): cat = 'n'
                elif tag.startswith('VB'): cat = 'v'
                elif tag.startswith('JJ'): cat = 'a'

                if cat:
                    lemma = self.lemmatizer.lemmatize(word, pos=cat)
                    if lemma in self.word_to_idx:
                        target[self.word_to_idx[lemma]] = 1.0
        return target

class Stage1_5Dataset(Dataset):
    def __init__(self, dataset_path, vocab_data_path):
        # 1. Load your ~8000 training samples
        self.samples = torch.load(dataset_path)
        
        # 2. Load the global corpus (1210 words)
        vocab_info = torch.load(vocab_data_path)


        self.vocab_embeddings = vocab_info["embeddings"] 
        
        # 3. Setup the encoder
        self.encoder = MultiHotEncoder(vocab_info["words"])
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        eeg_emb = item['eeg_clip_latent'].squeeze().float()
        target = self.encoder.encode(item['caption'])
        return eeg_emb, target

def run_training(model, train_loader, device, epochs, criterion_type, save_path):
    from src.models import FocalLoss, MultiLabelContrastiveLoss
    import torch.nn as nn
    
    if criterion_type == "focal": criterion = FocalLoss()
    elif criterion_type == "contrastive": criterion = MultiLabelContrastiveLoss()
    else: criterion = nn.BCEWithLogitsLoss(pos_weight=torch.ones([model.vocab_matrix.shape[0]]).to(device) * 25.0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.to(device).train()
    
    for epoch in range(epochs):
        for eeg, target in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            eeg, target = eeg.to(device), target.to(device)
            optimizer.zero_grad()
            logits, _ = model(eeg)
            loss = criterion(logits, target)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    torch.save(model.state_dict(), save_path)