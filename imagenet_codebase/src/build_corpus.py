import torch
import clip
import nltk
import os
from tqdm import tqdm
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter

# Ensure NLTK resources are available
for res in ['punkt', 'stopwords', 'wordnet', 'averaged_perceptron_tagger']:
    nltk.download(res, quiet=True)

def get_imagenet_vocab(dataset_path):
    """Correctly extracts and cleans vocab from your flat EEG/Text list."""
    # Load your new preprocessed list of dicts
    dataset = torch.load(dataset_path, map_location='cpu')
    print(f"Processing {len(dataset)} captions for vocabulary extraction...")

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # We want to capture distinct semantic nouns and verbs from the captions
    all_words = []

    for item in tqdm(dataset, desc="Tokenizing Captions"):
        caption = item.get('caption', "").lower()
        if not caption: continue
            
        # 1. Tokenize and Tag POS (Parts of Speech)
        tokens = nltk.word_tokenize(caption)
        tagged = nltk.pos_tag(tokens)
        
        for word, tag in tagged:
            # 2. Filter for significant words (Nouns, Verbs, Adjectives)
            if word.isalpha() and word not in stop_words and len(word) > 2:
                # Map NLTK tags to Lemmatizer categories
                cat = None
                if tag.startswith('NN'): cat = 'n'   # Noun
                elif tag.startswith('VB'): cat = 'v' # Verb
                elif tag.startswith('JJ'): cat = 'a' # Adjective
                
                if cat:
                    lemma = lemmatizer.lemmatize(word, pos=cat)
                    all_words.append(lemma)

    # 3. Use Counter to get unique words, potentially filtering by frequency if needed
    vocab = sorted(list(set(all_words)))
    print(f"Generated a unique vocabulary of {len(vocab)} words.")
    return vocab

def build_corpus(mode="imagenet", dataset_path=None, output_path="data/corpus.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # 1. Gather Vocabulary
    if mode == "brown":
        print("Building Brown Corpus...")
        words = nltk.corpus.brown.words()
        clean = [w.lower() for w in words if w.isalpha() and w.lower() not in set(stopwords.words('english'))]
        vocab = [w for w, _ in Counter(clean).most_common(20000)]
    else:
        vocab = get_imagenet_vocab(dataset_path)

    # 2. CLIP Encoding (Must use ViT-B/32 to align with Thought2Text/ChannelNet)
    print(f"Loading CLIP ViT-B/32 on {device}...")
    model, _ = clip.load("ViT-B/32", device=device)
    all_embs = []
    
    # 3. Batch Encode Vocab into CLIP Text Space
    print(f"Encoding {len(vocab)} words...")
    batch_size = 10
    for i in tqdm(range(0, len(vocab), batch_size)):
        batch = vocab[i:i+batch_size]
        tokens = clip.tokenize(batch).to(device)
        with torch.no_grad():
            # Get the 512-D text embeddings
            embs = model.encode_text(tokens)
            # Normalize for cosine similarity alignment
            embs /= embs.norm(dim=-1, keepdim=True)
            # Move to CPU immediately to save GPU memory on your A100
            embs = embs.cpu()

            # Iterate through the batch and store each as [1, 512]
            for j in range(embs.size(0)):
                # Indexing with [j:j+1] keeps the batch dimension [1, 512]
                all_embs.append(embs[j:j+1])

    final_tensor = torch.stack(all_embs)
    # 4. Save for use in src/aligner.py
    final_data = {
        "words": vocab, 
        "embeddings": final_tensor # Shape: [Vocab_Size, 512]
    }
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(final_data, output_path)
    print(f"✅ Corpus successfully saved to {output_path}")

if __name__ == "__main__":
    build_corpus(
        mode="imagenet", 
        dataset_path="results/channelnet_imagenet_eeg_train_clip_latents_with_pred_label_confidence.pt",
        output_path="data/imagenet_train_corpus.pt"
    )