import os
import re
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from collections import Counter
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import chain


data_location = "../archive"
images_path = os.path.join(data_location, "Images")
captions_file = os.path.join(data_location, "captions.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load Captions
df = pd.read_csv(captions_file, delimiter=",")
df.columns = ["image", "caption"]
df["caption"] = df["caption"].str.lower().str.strip()

# Vocabulary
class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return re.findall(r'\b\w+\b', text.lower())

    def build_vocabulary(self, sentence_list):
        frequencies = Counter()
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                frequencies[word] += 1
                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)
        return [self.stoi.get(token, self.stoi["<UNK>"]) for token in tokenized_text]


# Dataset
class FlickrDataset(Dataset):
    def __init__(self, root_dir, captions_df, vocab=None, transform=None, freq_threshold=5):
        self.root_dir = root_dir
        self.df = captions_df.reset_index(drop=True)
        self.transform = transform
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

        if vocab is None:
            self.vocab = Vocabulary(freq_threshold)
            self.vocab.build_vocabulary(self.captions.tolist())
        else:
            self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.captions[idx]
        img_id = self.imgs[idx]
        img_path = os.path.join(self.root_dir, img_id)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return image, torch.tensor(numericalized_caption)


# Collate Function
class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)
        return imgs, targets


# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225)),
])

# Train/Test Split
dataset_full = FlickrDataset(images_path, df, transform=transform)
train_size = int(0.8 * len(dataset_full))
test_size = len(dataset_full) - train_size
train_dataset, test_dataset = random_split(dataset_full, [train_size, test_size])

pad_idx = dataset_full.vocab.stoi["<PAD>"]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=MyCollate(pad_idx))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=MyCollate(pad_idx))

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")


# Encoder (CNN)
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.linear = nn.Linear(512, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        with torch.no_grad():
            features = self.resnet(images)
        features = features.reshape(features.size(0), -1)
        features = self.linear(features)
        features = self.bn(features)
        return features


# Decoder (Vanilla RNN)
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.rnn(inputs)
        outputs = self.linear(hiddens)
        return outputs

    def generate_caption(self, feature, vocab, max_len=20):
        """Greedy caption generation"""
        caption = []
        with torch.no_grad():
            hidden = None
            input_token = feature.unsqueeze(1)
            for _ in range(max_len):
                hiddens, hidden = self.rnn(input_token, hidden)
                output = self.linear(hiddens.squeeze(1))
                predicted_id = output.argmax(1).item()
                word = vocab.itos.get(predicted_id, "<UNK>")
                if word == "<EOS>":
                    break
                caption.append(word)
                input_token = self.embed(torch.tensor([predicted_id]).to(feature.device)).unsqueeze(1)
        return caption


# Combine Encoder + Decoder
class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(CNNtoRNN, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs


# BLEU Score
def compute_bleu(references, candidates):
    def ngram_counts(tokens, n):
        return Counter([tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)])

    def modified_precision(refs, cand, n):
        cand_counts = ngram_counts(cand, n)
        if not cand_counts:
            return 0
        max_counts = Counter()
        for ref in refs:
            ref_counts = ngram_counts(ref, n)
            for ngram in ref_counts:
                max_counts[ngram] = max(max_counts[ngram], ref_counts[ngram])
        clipped_counts = {ng: min(count, max_counts.get(ng, 0)) for ng, count in cand_counts.items()}
        return sum(clipped_counts.values()) / sum(cand_counts.values())

    weights = [0.25, 0.25, 0.25, 0.25]
    precisions = [modified_precision(references, candidates, i + 1) for i in range(4)]
    precisions = [p if p > 0 else 1e-9 for p in precisions]
    bleu = np.exp(sum(w * np.log(p) for w, p in zip(weights, precisions)))
    return bleu


# Training
embed_size = 256
hidden_size = 512
vocab_size = len(dataset_full.vocab)

model = CNNtoRNN(embed_size, hidden_size, vocab_size).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for imgs, captions in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
        imgs, captions = imgs.to(device), captions.to(device)

        outputs = model(imgs, captions)
        loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {avg_loss:.4f}")

# Evaluation
model.eval()
test_loss = 0
bleu_scores = []

with torch.no_grad():
    for imgs, captions in tqdm(test_loader, desc="Evaluating"):
        imgs, captions = imgs.to(device), captions.to(device)
        outputs = model(imgs, captions)
        loss = criterion(outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1))
        test_loss += loss.item()

        # Generate captions and compute BLEU
        features = model.encoder(imgs)
        for i in range(imgs.size(0)):
            pred_caption = model.decoder.generate_caption(features[i], dataset_full.vocab)
            ref_caption = dataset_full.vocab.tokenizer_eng(dataset_full.captions[i])
            bleu = compute_bleu([ref_caption], pred_caption)
            bleu_scores.append(bleu)

avg_test_loss = test_loss / len(test_loader)
mean_bleu = np.mean(bleu_scores)

print(f"\nTest Loss: {avg_test_loss:.4f}")
print(f"Average BLEU Score: {mean_bleu:.4f}")
