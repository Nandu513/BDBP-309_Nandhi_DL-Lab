import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from collections import Counter
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")

data_location = "../archive"
images_path = os.path.join(data_location, "Images")
captions_file = os.path.join(data_location, "captions.txt")

embed_size = 256
hidden_size = 512
num_epochs = 2        
batch_size = 32
learning_rate = 1e-3
max_len = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(captions_file)


class Vocabulary:
    def __init__(self, freq_threshold):
        self.freq_threshold = freq_threshold
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer(text):
        return nltk.word_tokenize(text.lower())

    def build_vocab(self, sentence_list):
        freqs = Counter()
        for sentence in sentence_list:
            tokens = self.tokenizer(sentence)
            freqs.update(tokens)

        for word, freq in freqs.items():
            if freq >= self.freq_threshold:
                idx = len(self.itos)
                self.itos[idx] = word
                self.stoi[word] = idx

    def numericalize(self, text):
        tokenized = self.tokenizer(text)
        return [
            self.stoi.get(token, self.stoi["<UNK>"])
            for token in tokenized
        ]

# Build vocab
nltk.download("punkt")
vocab = Vocabulary(freq_threshold=5)
vocab.build_vocab(df["caption"].tolist())


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

class FlickrDataset(Dataset):
    def __init__(self, dataframe, img_dir, vocab, transform=None, max_len=20):
        self.df = dataframe
        self.img_dir = img_dir
        self.vocab = vocab
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = self.df.iloc[idx, 0]
        caption = self.df.iloc[idx, 1]

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Numerical caption
        tokens = [self.vocab.stoi["<SOS>"]]
        tokens += self.vocab.numericalize(caption)
        tokens.append(self.vocab.stoi["<EOS>"])

        # Pad / truncate
        if len(tokens) < self.max_len:
            tokens += [self.vocab.stoi["<PAD>"]] * (self.max_len - len(tokens))
        else:
            tokens = tokens[:self.max_len]

        return image, torch.tensor(tokens)

dataset = FlickrDataset(df, images_path, vocab, transform, max_len=max_len)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


# 5. Feature extractor (ResNet18)
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
modules = list(resnet.children())[:-1]
resnet = nn.Sequential(*modules)
resnet.to(device)
resnet.eval()
for p in resnet.parameters():
    p.requires_grad = False


class ImageCaptionRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(ImageCaptionRNN, self).__init__()
        self.fc_img = nn.Linear(512, embed_size)  # ResNet18 output dim
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc_out = nn.Linear(hidden_size, vocab_size)

    def forward(self, img_features, captions):
        # Encode image
        img_features = self.fc_img(img_features).unsqueeze(1)  # (B,1,E)
        embeddings = self.embed(captions[:, :-1])              # (B,L-1,E)

        # Concatenate image feature at start of caption embeddings
        inputs = torch.cat((img_features, embeddings), dim=1)  # (B,L,E)

        outputs, _ = self.rnn(inputs)
        outputs = self.fc_out(outputs)
        return outputs

model = ImageCaptionRNN(embed_size, hidden_size, len(vocab)).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    model.train()
    for i, (imgs, caps) in enumerate(dataloader):
        imgs, caps = imgs.to(device), caps.to(device)

        # Extract features
        with torch.no_grad():
            feats = resnet(imgs).view(imgs.size(0), -1)  # (B,512)

        # Forward
        outputs = model(feats, caps)
        loss = criterion(outputs.reshape(-1, outputs.size(2)), caps.reshape(-1))

        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataloader)}], Loss: {loss.item():.4f}")


def generate_caption(model, image_path, vocab, max_len=20):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = resnet(image).view(1, -1)
        feat = model.fc_img(feat).unsqueeze(1)  # (1,1,E)

        states = None
        caption = [vocab.stoi["<SOS>"]]

        for _ in range(max_len):
            emb = model.embed(torch.tensor([caption[-1]]).to(device)).unsqueeze(1)
            inputs = emb if len(caption) > 1 else feat
            out, states = model.rnn(inputs, states)
            out = model.fc_out(out.squeeze(1))
            pred = out.argmax(1).item()
            caption.append(pred)
            if pred == vocab.stoi["<EOS>"]:
                break

    words = [vocab.itos[idx] for idx in caption]
    return " ".join(words[1:-1])  # remove <SOS>, <EOS>

test_img = os.path.join(images_path, df.iloc[55, 0])
print("Generated caption:", generate_caption(model, test_img, vocab))

