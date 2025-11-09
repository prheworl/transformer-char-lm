import os, urllib.request, torch
DATA_DIR = "data"
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
PATH = os.path.join(DATA_DIR, "tiny_shakespeare.txt")

def download_data():
    os.makedirs(DATA_DIR, exist_ok=True)
    if not os.path.exists(PATH):
        urllib.request.urlretrieve(URL, PATH)
    with open(PATH, "r", encoding="utf-8") as f:
        return f.read()

def build_vocab(text):
    vocab = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(vocab)}
    itos = {i: ch for ch, i in stoi.items()}
    def encode(s): return torch.tensor([stoi[c] for c in s], dtype=torch.long)
    def decode(t): return "".join([itos[int(i)] for i in t])
    return stoi, itos, encode, decode

def make_splits(encoded, split=(0.9, 0.05, 0.05)):
    n = len(encoded)
    n_train = int(n * split[0]); n_val = int(n * split[1])
    train = encoded[:n_train]; val = encoded[n_train:n_train+n_val]; test = encoded[n_train+n_val:]
    return train, val, test

class CharDataset:
    def __init__(self, data, block_size):
        self.data = data; self.block = block_size
    def __len__(self): return len(self.data) - self.block - 1
    def get_batch(self, batch_size, device):
        ix = torch.randint(0, len(self.data) - self.block - 1, (batch_size,))
        x = torch.stack([self.data[i:i+self.block] for i in ix]).to(device)
        y = torch.stack([self.data[i+1:i+self.block+1] for i in ix]).to(device)
        return x, y