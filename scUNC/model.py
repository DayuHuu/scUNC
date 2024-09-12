import torch
from torch import nn
from torch.nn import functional as F

class Network(nn.Module):
    def __init__(self, input_A: int,input_B: int, embedding_size: int, act_fn=torch.nn.LeakyReLU):
        super(Network, self).__init__()

        self.encoder1 = torch.nn.Sequential(
            torch.nn.Linear(input_A, 512),
            act_fn(inplace=True),
            torch.nn.Linear(512, 256),
            act_fn(inplace=True),
            torch.nn.Linear(256, 128),
            act_fn(inplace=True),
            torch.nn.Linear(128, embedding_size//2))

        self.encoder2 = torch.nn.Sequential(
            torch.nn.Linear(input_B, 512),
            act_fn(inplace=True),
            torch.nn.Linear(512, 256),
            act_fn(inplace=True),
            torch.nn.Linear(256, 128),
            act_fn(inplace=True),
            torch.nn.Linear(128, embedding_size//2))

        self.trans_enc = nn.TransformerEncoderLayer(d_model= embedding_size, nhead=1, dim_feedforward=256)
        self.extract_layers = nn.TransformerEncoder(self.trans_enc, num_layers=1)
        self.layer4 = nn.Linear(embedding_size, 300)
        self.layer5_1 = nn.Linear(300, 500)
        self.layer6_1 = nn.Linear(500, input_A)
        self.layer6_2 = nn.Linear(300, input_B)
        self.drop = 0.5
    def encode(self, Xs: torch.Tensor) -> torch.Tensor:

        x1, x2 = Xs
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x2)
        y = self.extract_layers(torch.cat((x1, x2), 1))
        
        return y

    def decode(self, embedded) :
        x = F.dropout(F.relu(self.layer4(embedded)), self.drop)
        out1 = F.relu(self.layer5_1(x))
        out1 = self.layer6_1(out1)
        out2 = self.layer6_2(x)
        return out1, out2

    def forward(self, Xs) :
        embedded = self.encode(Xs)
        out1,out2 = self.decode(embedded)
        return out1,out2

    def start_training(self, trainloader, n_epochs, device, optimizer, loss_fn):
        for _ in range(n_epochs):
            for batch_idx, (xs, _) in enumerate(trainloader):
                for v in range(2):
                    xs[v] = torch.squeeze(xs[v]).to(device)
                emb = self.encode(xs)
                out1, out2 = self.decode(emb)
                loss = loss_fn(out1, xs[0])+loss_fn(out2, xs[1])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()



