import torch
import torch.nn as nn
import torch.nn.functional as F

#hyperparameter
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 1e-4 # as we use self attention so lr would be low
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embedding = 384
n_heads = 6 #no of head for multihead attention
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

#download the dataset
with open('input.txt','r') as f:
    text = f.read()

# build the vocabulary from the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)
max_iters = 3000
#create a mapping from chars to integer
stoi = {s:i for i,s in enumerate(chars)}
itos={i:s for s,i in stoi.items()}
#encoder -> take a string and o/p list of integers
encoder = lambda s: [stoi[c] for c in s]
decoder = lambda l:''.join([itos[i] for i in l])

#train and test splitting
data = torch.tensor(encoder(text),dtype = torch.long)
n = int(0.9*len(data))
train_data = data[:n]
test_data = data[n:]

def get_batch(split):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size]for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x = x.to(device)
    y = y.to(device)
    return x,y

class Head(nn.Module):
    '''one head attention'''
    def __init__(self,head_size):
        super().__init__()
        self.key = nn.Linear(n_embedding,head_size,bias=False)
        self.query = nn.Linear(n_embedding,head_size,bias=False)
        self.value = nn.Linear(n_embedding,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)
        self.head_size = head_size

    def forward(self,x):
        B,T,C = x.shape 
        k = self.key(x) #(B,T,C) -> (B,T,Head_size)
        q = self.query(x)#(B,T,C) -> (B,T,Head_size)
        v = self.value(x)#(B,T,C) -> (B,T,Head_size)

        # head_size = C for now
        wei = (q @ k.transpose(-2,-1)*self.head_size**-0.5) # -> (B,T,T)
        wei = wei.masked_fill(self.tril[:T,:T] ==0, float('-inf'))
        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)
        out = wei @ v # -> (B,T,T) @ (B,T,hs) = (B,T,hs)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self,n_heads,h_size):
        super().__init__()
        self.n_heads = n_heads
        self.h_size = h_size
        self.heads = nn.ModuleList([Head(h_size) for _ in range(n_heads)])
        self.proj = nn.Linear(h_size*n_heads,n_embedding)
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim = -1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    def __init__(self,n_embedding):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Linear(n_embedding,4*n_embedding),
            nn.ReLU(),
            nn.Linear(4*n_embedding,n_embedding),
            nn.Dropout(dropout)
        )
    def forward(self,x):
        return self.seq(x)
    
class Block(nn.Module):
    '''Transfomer block: communication followed by computation'''
    def __init__(self,n_heads,h_size):
        super().__init__()
        h_size_multi = h_size // n_heads
        self.sa = MultiHeadAttention(n_heads=n_heads,h_size=h_size_multi)
        self.lm1 = nn.LayerNorm(n_embedding)
        self.lm2 = nn.LayerNorm(n_embedding)
        self.ffwd = FeedForward(n_embedding)
    def forward(self,x):
        x = (self.sa(self.lm1(x)))+x
        x = self.ffwd(self.lm2(x)) + x
        return x
    
class GPTLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size,n_embedding)
        self.positional_embedding_table = nn.Embedding(block_size,n_embedding)
        self.blocks = nn.Sequential(*[Block(n_heads,n_embedding) for _ in range(n_layer)])
        self.lm = nn.LayerNorm(n_embedding)
        self.lm_head = nn.Linear(n_embedding,vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self,module):
        if isinstance(module,nn.Linear):
            torch.nn.init.normal_(module.weight,mean=0.0,std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module,nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self,idx,targets= None):
        B,T =idx.shape
        token_embeddings = self.token_embedding_table(idx) # (B,T,C)
        pos_embeddings = self.positional_embedding_table(torch.arange(T,device=device))#(T,C)
        x = token_embeddings+pos_embeddings #(B,T,C) broadcast
        x = self.blocks(x) #(B,T,C) 
        x = self.lm(x)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape # here C == vocab_size
            logits = logits.view(B*T,vocab_size) 
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)

        return logits, loss
    
    def generate(self,idx,max_new_tokens):
        #idx -> (B,T)
        for _ in range(max_new_tokens):
            idx_new = idx[:,-block_size:]
            logits,loss = self(idx_new) #logits -> (B,T,C)
            logits = logits[:,-1,:]# ->(B,C)
            probs = F.softmax(logits,dim = -1) #(B,C)
            idx_next = torch.multinomial(probs,num_samples=1,replacement=True)
            idx = torch.cat((idx,idx_next),dim = 1)

        return idx
    

model = GPTLanguageModel().to(device)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','test']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x,y = get_batch(split)
            logits, loss = model(x,y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
        


# print the no of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

#create the pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(),lr= learning_rate)

for iter in range(max_iters):

    #every once in a while the loss in the train and val loss
    if iter%eval_interval == 0:
        losses = estimate_loss()
        print(f'Iter {iter}: Train Loss: {losses["train"]:.4f}, Val Loss: {losses["test"]:.4f}')

    #train the model
    optimizer.zero_grad()
    x,y = get_batch('train')
    logits, loss = model(x,y)
    loss.backward()
    optimizer.step()

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decoder(model.generate(context, max_new_tokens=500)[0].tolist()))
open('more.txt', 'w').write(decoder(model.generate(context, max_new_tokens=10000)[0].tolist()))






