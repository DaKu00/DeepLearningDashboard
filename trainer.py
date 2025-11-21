import torch
import torch.nn as nn
import torch.optim as optim
import asyncio
import math

# 1. 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Self-Attention 모듈 (기존 코드 유지)
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "임베딩 크기(embed_size)는 헤드 수(heads)로 나누어 떨어져야 합니다."

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads * self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out

# 3. 트랜스포머 블록 (기존 코드 유지)
class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

# 4. 간단한 분류 모델 (기존 코드 유지)
class SimpleTransformerClassifier(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
        num_classes
    ):
        super(SimpleTransformerClassifier, self).__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size,
                    heads,
                    dropout=dropout,
                    forward_expansion=forward_expansion,
                )
                for _ in range(num_layers)
            ]
        )
        
        self.fc_out = nn.Linear(embed_size, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(
            (self.word_embedding(x) + self.position_embedding(positions))
        )

        for layer in self.layers:
            out = layer(out, out, out, mask)
            
        out = out.mean(dim=1) 
        out = self.fc_out(out)
        return out

# 5. Trainer 클래스
class Trainer:
    def __init__(self, config, callback=None):
        self.config = config
        self.callback = callback
        self.stop_requested = False
        
        # 하이퍼파라미터 설정
        self.src_vocab_size = 100
        self.embed_size = 256
        self.num_layers = 2
        self.heads = 8
        self.forward_expansion = 4
        self.dropout = 0.1
        self.max_length = 100
        self.num_classes = 2
        
        # 사용자 설정
        self.learning_rate = config.get("learning_rate", 1e-3)
        self.batch_size = config.get("batch_size", 32)
        self.num_epochs = config.get("num_epochs", 100)
        
        # 모델 초기화
        self.model = SimpleTransformerClassifier(
            self.src_vocab_size,
            self.embed_size,
            self.num_layers,
            self.heads,
            device,
            self.forward_expansion,
            self.dropout,
            self.max_length,
            self.num_classes
        ).to(device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def get_batch(self, batch_size):
        data = torch.randint(0, self.src_vocab_size, (batch_size, self.max_length)).to(device)
        targets = torch.randint(0, self.num_classes, (batch_size,)).to(device)
        return data, targets

    async def train(self):
        print(f"Starting training on {device}...")
        self.model.train()
        
        for epoch in range(self.num_epochs):
            if self.stop_requested:
                print("Training stopped by user.")
                break
                
            # Training Step
            data, targets = self.get_batch(self.batch_size)
            scores = self.model(data)
            loss = self.criterion(scores, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Validation Step (Simple simulation)
            self.model.eval()
            with torch.no_grad():
                val_data, val_targets = self.get_batch(self.batch_size)
                val_scores = self.model(val_data)
                val_loss = self.criterion(val_scores, val_targets)
            self.model.train()
            
            # Report progress
            if self.callback:
                await self.callback({
                    "epoch": epoch + 1,
                    "total_epochs": self.num_epochs,
                    "train_loss": loss.item(),
                    "val_loss": val_loss.item()
                })
            
            # Simulate some time per epoch for visualization
            await asyncio.sleep(0.1)

        print("Training finished!")

    def stop(self):
        self.stop_requested = True
