import torch
import torch.nn as nn
import torch.nn.functional as F

# intialization of variables
batch_size = 32
block_size = 8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_interval = 300
learning_rate = 1e-2
eval_iters = 200

# define the manual_seed
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

@torch.no_grad()
#estimate the loss
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','test']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            x,y = get_batch(split)
            logits,loss = model(x,y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


#simple BigarmLanguage model
class BigramLanguageModel(nn.Module):
    def __init__(self,vocab_size):
        super().__init__()
        #each ele in vocab will have dim of vocab size -> the prob of generate the next chars among all the chars
        self.token_embedding_table = nn.Embedding(vocab_size,vocab_size)

    def forward(self,idx,targets = None):
        #idx dim : (Batch,Time)
        logits = self.token_embedding_table(idx) #-> batch,time,channel
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)

        return logits,loss

    def generate(self,idx,max_new_tokens):
        #idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            #prediction
            logits,loss = self(idx)
            #get the logits of the last time step, this is the prediction for the next token
            logits = logits[:, -1, :]  #(B, C)
            #apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) #(B, C)
            #sample from the distribution to get next token
            idx_next = torch.multinomial(probs, num_samples=1) #(B, 1)
            #concatenate the new token to the input sequence
            idx = torch.cat((idx, idx_next), dim=1) #(B, T+1)
        return idx
    
model = BigramLanguageModel(vocab_size)
m = model.to(device)

#create pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(),lr= learning_rate)

#train the model
for iter in range(max_iters):
    if iter % eval_interval ==0:
        losses = estimate_loss()
        print(f"step {iter}: training loss: {losses['train']:.4f}, val loss:{losses['test']:.4f}")

    x,y = get_batch('train')
    logtis,loss = model(x,y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate some text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decoder(m.generate(context, max_new_tokens=500)[0].tolist()))