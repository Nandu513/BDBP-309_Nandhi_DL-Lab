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
        return [self.stoi.get(tok, self.stoi["<UNK>"]) for tok in self.tokenizer_eng(text)]


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

# Split Data
dataset_full = FlickrDataset(images_path, df, transform=transform)
train_size = int(0.8 * len(dataset_full))
test_size = len(dataset_full) - train_size
train_dataset, test_dataset = random_split(dataset_full, [train_size, test_size])

pad_idx = dataset_full.vocab.stoi["<PAD>"]

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=MyCollate(pad_idx))
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=MyCollate(pad_idx))

print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")


# Encoder (ResNet18)
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


# Pretrained Embedding Loader (GloVe)
def load_glove_embeddings(vocab, embedding_dim=300, glove_path="../glove.6B.300d.txt"):
    embeddings = np.random.normal(scale=0.6, size=(len(vocab), embedding_dim))
    print("Loading GloVe embeddings...")
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            if word in vocab.stoi:
                idx = vocab.stoi[word]
                embeddings[idx] = vector
    return torch.tensor(embeddings, dtype=torch.float32)


# Decoder (LSTM)
class DecoderLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, embeddings=None, freeze_emb=False):
        super(DecoderLSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        if embeddings is not None:
            self.embed.weight = nn.Parameter(embeddings)
            self.embed.weight.requires_grad = not freeze_emb

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.embed(captions[:, :-1])
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        hiddens, _ = self.lstm(inputs)
        outputs = self.linear(hiddens)
        return outputs

    def generate_caption(self, feature, vocab, max_len=20):
        caption = []
        with torch.no_grad():
            states = None
            input_token = feature.unsqueeze(1)
            for _ in range(max_len):
                hiddens, states = self.lstm(input_token, states)
                output = self.linear(hiddens.squeeze(1))
                predicted_id = output.argmax(1).item()
                word = vocab.itos.get(predicted_id, "<UNK>")
                if word == "<EOS>":
                    break
                caption.append(word)
                input_token = self.embed(torch.tensor([predicted_id]).to(feature.device)).unsqueeze(1)
        return caption


# Combine CNN + LSTM
class CNNtoLSTM(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, embeddings=None):
        super(CNNtoLSTM, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderLSTM(embed_size, hidden_size, vocab_size, embeddings=embeddings)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs


# BLEU Score Calculation
def compute_bleu(reference, candidate, n_gram=4):
    def ngrams(tokens, n):
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    weights = [0.25] * n_gram
    precisions = []
    for n in range(1, n_gram + 1):
        ref_ngrams = Counter(ngrams(reference, n))
        cand_ngrams = Counter(ngrams(candidate, n))
        overlap = sum(min(count, ref_ngrams[gram]) for gram, count in cand_ngrams.items())
        total = max(sum(cand_ngrams.values()), 1)
        precisions.append(overlap / total)
    bleu = np.exp(sum(w * np.log(max(p, 1e-9)) for w, p in zip(weights, precisions)))
    return bleu


# Initialize Model
embed_size = 300
hidden_size = 512
vocab_size = len(dataset_full.vocab)

if os.path.exists("../glove.6B.300d.txt"):
    glove_embeddings = load_glove_embeddings(dataset_full.vocab, embed_size)
else:
    glove_embeddings = None
    print("GloVe file not found, using random embeddings.")

model = CNNtoLSTM(embed_size, hidden_size, vocab_size, embeddings=glove_embeddings).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
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
    print(f"Epoch [{epoch + 1}/{num_epochs}] Train Loss: {total_loss / len(train_loader):.4f}")

print("\nTraining complete.\n")

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

        # Generate captions
        features = model.encoder(imgs)
        for i in range(imgs.size(0)):
            pred_caption = model.decoder.generate_caption(features[i], dataset_full.vocab)
            ref_caption = dataset_full.vocab.tokenizer_eng(dataset_full.captions[i])
            bleu = compute_bleu(ref_caption, pred_caption)
            bleu_scores.append(bleu)

avg_test_loss = test_loss / len(test_loader)
mean_bleu = np.mean(bleu_scores)

print(f"\nTest Loss: {avg_test_loss:.4f}")
print(f"Average BLEU Score: {mean_bleu:.4f}")
