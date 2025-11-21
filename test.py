import torch
import torch.nn as nn
import torch.optim as optim
import math

# 1. 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Self-Attention 모듈
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
        # values, keys, query 형태: (N, 시퀀스 길이, 임베딩 크기)
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        # 임베딩을 여러 개의 헤드로 분할
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        # 각 학습 예제에 대해 query*keys 행렬 곱셈을 수행 (einsum 사용)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])
        # queries 형태: (N, query_len, heads, heads_dim),
        # keys 형태: (N, key_len, heads, heads_dim)
        # energy: (N, heads, query_len, key_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        attention = torch.softmax(energy / (self.embed_size ** (1 / 2)), dim=3)
        # attention 형태: (N, heads, query_len, key_len)

        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )
        # out 형태: (N, query_len, heads, head_dim) 그 후 마지막 두 차원을 평탄화(flatten)

        out = self.fc_out(out)
        # out 형태: (N, query_len, embed_size)
        return out

# 3. 트랜스포머 블록
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

        # 스킵 연결(Skip Connection) 추가, 정규화(Normalization) 및 드롭아웃(Dropout) 적용
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out

# 4. 간단한 분류 모델
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

        # 인코더에서는 query, key, value가 모두 동일함 (디코더에서는 달라짐)
        for layer in self.layers:
            out = layer(out, out, out, mask)
            
        # 분류를 위한 Global Average Pooling
        out = out.mean(dim=1) 
        out = self.fc_out(out)
        return out

# 5. 하이퍼파라미터
SRC_VOCAB_SIZE = 100
EMBED_SIZE = 256
NUM_LAYERS = 2
HEADS = 8
FORWARD_EXPANSION = 4
DROPOUT = 0.1
MAX_LENGTH = 100
NUM_CLASSES = 2
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
NUM_EPOCHS = 100

# 6. 모델 초기화
model = SimpleTransformerClassifier(
    SRC_VOCAB_SIZE,
    EMBED_SIZE,
    NUM_LAYERS,
    HEADS,
    device,
    FORWARD_EXPANSION,
    DROPOUT,
    MAX_LENGTH,
    NUM_CLASSES
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# 7. 더미 데이터 생성기
def get_batch(batch_size, seq_len, vocab_size, num_classes):
    data = torch.randint(0, vocab_size, (batch_size, seq_len)).to(device)
    targets = torch.randint(0, num_classes, (batch_size,)).to(device)
    return data, targets

# 8. 학습 루프
print("Starting training...")
model.train()

for epoch in range(NUM_EPOCHS):
    data, targets = get_batch(BATCH_SIZE, MAX_LENGTH, SRC_VOCAB_SIZE, NUM_CLASSES)
    
    # 순전파 (Forward pass)
    scores = model(data)
    loss = criterion(scores, targets)
    
    # 역전파 (Backward pass)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {loss.item():.4f}")

print("Training finished!")

# 9. 간단한 추론 테스트
model.eval()
with torch.no_grad():
    test_data, _ = get_batch(1, MAX_LENGTH, SRC_VOCAB_SIZE, NUM_CLASSES)
    output = model(test_data)
    prediction = torch.argmax(output, dim=1)
    print(f"\nTest Input shape: {test_data.shape}")
    print(f"Prediction: {prediction.item()}")
