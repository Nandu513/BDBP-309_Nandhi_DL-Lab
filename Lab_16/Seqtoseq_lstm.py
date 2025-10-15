import pandas as pd
import os
import string
from string import digits
import re
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

lines=pd.read_csv("Hindi_English_Truncated_Corpus.csv",encoding='utf-8')
print(lines['source'].value_counts())

lines=lines[lines['source']=='ted']

print(lines.head(20))


print(pd.isnull(lines).sum())

lines=lines[~pd.isnull(lines['english_sentence'])]

lines.drop_duplicates(inplace=True)


lines=lines.sample(n=25000,random_state=42)
print(lines.shape)

# Lowercase all characters
lines['english_sentence']=lines['english_sentence'].apply(lambda x: x.lower())
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: x.lower())


# Remove quotes
lines['english_sentence']=lines['english_sentence'].apply(lambda x: re.sub("'", '', x))
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: re.sub("'", '', x))


exclude = set(string.punctuation) # Set of all special characters
# Remove all the special characters
lines['english_sentence']=lines['english_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: ''.join(ch for ch in x if ch not in exclude))


# Remove all numbers from text
remove_digits = str.maketrans('', '', digits)
lines['english_sentence']=lines['english_sentence'].apply(lambda x: x.translate(remove_digits))
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: x.translate(remove_digits))

lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x: re.sub("[२३०८१५७९४६]", "", x))

# Remove extra spaces
lines['english_sentence']=lines['english_sentence'].apply(lambda x: x.strip())
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: x.strip())
lines['english_sentence']=lines['english_sentence'].apply(lambda x: re.sub(" +", " ", x))
lines['hindi_sentence']=lines['hindi_sentence'].apply(lambda x: re.sub(" +", " ", x))


# Add start and end tokens to target sequences
lines['hindi_sentence'] = lines['hindi_sentence'].apply(lambda x : 'START_ '+ x + ' _END')

print(lines.head())

### Get English and Hindi Vocabulary
all_eng_words=set()
for eng in lines['english_sentence']:
    for word in eng.split():
        if word not in all_eng_words:
            all_eng_words.add(word)


all_hindi_words=set()
for hin in lines['hindi_sentence']:
    for word in hin.split():
        if word not in all_hindi_words:
            all_hindi_words.add(word)



print(len(all_eng_words))
print(len(all_hindi_words))


lines['length_eng_sentence']=lines['english_sentence'].apply(lambda x:len(x.split(" ")))
lines['length_hin_sentence']=lines['hindi_sentence'].apply(lambda x:len(x.split(" ")))

print(lines.head())

print(lines[lines['length_eng_sentence']>30].shape)

lines=lines[lines['length_eng_sentence']<=20]
lines=lines[lines['length_hin_sentence']<=20]

print(lines.shape)

print("maximum length of Hindi Sentence ",max(lines['length_hin_sentence']))
print("maximum length of English Sentence ",max(lines['length_eng_sentence']))


max_length_src=max(lines['length_hin_sentence'])
max_length_tar=max(lines['length_eng_sentence'])


input_words = sorted(list(all_eng_words))
target_words = sorted(list(all_hindi_words))
# print(input_words)
num_encoder_tokens = len(all_eng_words)
num_decoder_tokens = len(all_hindi_words)
print(num_encoder_tokens, num_decoder_tokens)

num_decoder_tokens += 1 #for zero padding

input_token_index = dict([(word, i+1) for i, word in enumerate(input_words)])
target_token_index = dict([(word, i+1) for i, word in enumerate(target_words)])

reverse_input_char_index = dict((i, word) for word, i in input_token_index.items())
reverse_target_char_index = dict((i, word) for word, i in target_token_index.items())


lines = shuffle(lines)
print(lines.head(10))


X, y = lines['english_sentence'], lines['hindi_sentence']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,random_state=42)
print(X_train.shape, X_test.shape)


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperparameters
embedding_dim = 256
hidden_size = 512
batch_size = 64
num_epochs = 10
max_encoder_seq_length = max_length_src
max_decoder_seq_length = max_length_tar

# Dataset
class TranslationDataset(Dataset):
    def __init__(self, X, y, input_token_index, target_token_index, max_encoder_len, max_decoder_len):
        self.X = X.values
        self.y = y.values
        self.input_token_index = input_token_index
        self.target_token_index = target_token_index
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Encoder sequence
        enc_seq = np.zeros(self.max_encoder_len, dtype=int)
        for i, word in enumerate(self.X[idx].split()):
            if i < self.max_encoder_len:
                enc_seq[i] = self.input_token_index.get(word, 0)

        # Decoder input sequence
        dec_seq = np.zeros(self.max_decoder_len, dtype=int)
        dec_target_seq = np.zeros(self.max_decoder_len, dtype=int)
        for i, word in enumerate(self.y[idx].split()):
            if i < self.max_decoder_len:
                dec_seq[i] = self.target_token_index.get(word, 0)
                if i > 0:
                    dec_target_seq[i-1] = self.target_token_index.get(word, 0)

        return torch.tensor(enc_seq, dtype=torch.long), torch.tensor(dec_seq, dtype=torch.long), torch.tensor(dec_target_seq, dtype=torch.long)


train_dataset = TranslationDataset(X_train, y_train, input_token_index, target_token_index, max_encoder_seq_length, max_decoder_seq_length)
val_dataset = TranslationDataset(X_test, y_test, input_token_index, target_token_index, max_encoder_seq_length, max_decoder_seq_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Model
class Encoder(nn.Module):
    def __init__(self, input_vocab_size, embed_size, hidden_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size+1, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        x = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(x)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, target_vocab_size, embed_size, hidden_size):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(target_vocab_size+1, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, target_vocab_size+1)

    def forward(self, x, hidden, cell):
        x = self.embedding(x)
        outputs, (hidden, cell) = self.lstm(x, (hidden, cell))
        outputs = self.fc(outputs)
        return outputs, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg):
        batch_size, trg_len = trg.shape
        trg_vocab_size = self.decoder.fc.out_features

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(device)

        hidden, cell = self.encoder(src)

        decoder_input = trg[:, :1]  # first token (START_)

        for t in range(1, trg_len):
            out, hidden, cell = self.decoder(decoder_input, hidden, cell)
            outputs[:, t, :] = out[:, 0, :]
            decoder_input = trg[:, t].unsqueeze(1)  # teacher forcing

        return outputs

encoder = Encoder(num_encoder_tokens, embedding_dim, hidden_size).to(device)
decoder = Decoder(num_decoder_tokens, embedding_dim, hidden_size).to(device)
model = Seq2Seq(encoder, decoder).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for enc_seq, dec_seq, dec_target in train_loader:
        enc_seq, dec_seq, dec_target = enc_seq.to(device), dec_seq.to(device), dec_target.to(device)
        optimizer.zero_grad()
        output = model(enc_seq, dec_seq)       # (batch, trg_len, vocab_size)
        output = output[:,1:,:]                # remove first token
        batch_size, seq_len, vocab_size = output.shape
        output = output.contiguous().view(batch_size*seq_len, vocab_size)
        dec_target = dec_target[:, :seq_len].contiguous().view(-1)
        loss = criterion(output, dec_target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)

    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for enc_seq, dec_seq, dec_target in val_loader:
            enc_seq, dec_seq, dec_target = enc_seq.to(device), dec_seq.to(device), dec_target.to(device)
            output = model(enc_seq, dec_seq)
            output = output[:,1:,:]
            batch_size, seq_len, vocab_size = output.shape
            output = output.contiguous().view(batch_size*seq_len, vocab_size)
            dec_target = dec_target[:, :seq_len].contiguous().view(-1)
            loss = criterion(output, dec_target)
            val_loss += loss.item()
        val_loss /= len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# Test prediction for one sample
def translate_sentence(sentence):
    model.eval()
    enc_seq = np.zeros(max_encoder_seq_length, dtype=int)
    for i, word in enumerate(sentence.split()):
        if i < max_encoder_seq_length:
            enc_seq[i] = input_token_index.get(word, 0)
    enc_seq = torch.tensor(enc_seq, dtype=torch.long).unsqueeze(0).to(device)

    hidden, cell = encoder(enc_seq)

    decoder_input = torch.tensor([[target_token_index['START_']]], dtype=torch.long).to(device)
    decoded_sentence = []

    for _ in range(max_decoder_seq_length):
        output, hidden, cell = decoder(decoder_input, hidden, cell)
        pred_token = output.argmax(2)[:, -1].item()
        word = reverse_target_char_index.get(pred_token, '')
        if word == '_END':
            break
        decoded_sentence.append(word)
        decoder_input = torch.tensor([[pred_token]], dtype=torch.long).to(device)

    return ' '.join(decoded_sentence)

# Pick a random test sample
sample_idx = 0
sample_eng = X_test.iloc[sample_idx]
sample_hin = y_test.iloc[sample_idx]

pred_hin = translate_sentence(sample_eng)

print("English Sentence:", sample_eng)
print("Actual Hindi Sentence:", sample_hin)
print("Predicted Hindi Sentence:", pred_hin)
