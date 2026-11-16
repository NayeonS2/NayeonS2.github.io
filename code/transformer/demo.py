import math

import torch
import torch.nn.functional as F

torch.manual_seed(0)  # 난수를 고정해서 실행할 때마다 같은 결과가 나오게 한다


def attention(q, k, v, mask=None):
    """Scaled dot-product attention: softmax(QK^T / sqrt(d_k)) V  (논문 식 (1))

    q, k, v: (..., T, d_k) 모양. 앞쪽 차원(배치, 헤드)은 몇 개든 상관없다.
    mask: (T, T) bool 텐서. True인 칸만 볼 수 있고 False인 칸은 가린다.
    """
    d_k = q.size(-1)
    # 모든 query와 모든 key의 내적을 한 번에 계산한다 -> (..., T, T)
    # scores[i, j]는 "i번째 위치가 j번째 위치와 얼마나 관련 있는가"를 뜻한다.
    # sqrt(d_k)로 나누는 이유는 아래 3번 검사에서 직접 확인한다.
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        # 가릴 칸을 -inf로 채우면 softmax를 거친 뒤 가중치가 정확히 0이 된다.
        scores = scores.masked_fill(~mask, float("-inf"))
    # 마지막 축(key 방향)으로 softmax를 하면 각 query의 가중치 합이 1이 된다.
    weights = scores.softmax(dim=-1)
    # 가중치로 value를 가중 평균한다 -> (..., T, d_k)
    return weights @ v, weights


class MultiHeadAttention(torch.nn.Module):
    """MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O  (논문 3.2.2절)"""

    def __init__(self, d_model, h):
        super().__init__()
        assert d_model % h == 0  # d_model을 헤드 수로 나눠 떨어져야 헤드마다 같은 차원을 줄 수 있다
        self.h, self.d_k = h, d_model // h  # 논문 기본값: h=8, d_k=512/8=64
        # W^Q, W^K, W^V: 헤드 h개의 투영 행렬을 (d_model -> d_model) 하나로 묶어 둔 것이다.
        # 출력 d_model 차원을 h조각으로 자르면 헤드별 W_i^Q, W_i^K, W_i^V가 된다.
        self.w_q = torch.nn.Linear(d_model, d_model)
        self.w_k = torch.nn.Linear(d_model, d_model)
        self.w_v = torch.nn.Linear(d_model, d_model)
        self.w_o = torch.nn.Linear(d_model, d_model)  # 헤드를 이어 붙인 뒤 곱하는 W^O

    def split(self, x):
        # (B, T, d_model) -> (B, T, h, d_k) -> (B, h, T, d_k)
        # 헤드 축을 앞으로 빼면 attention()이 모든 헤드를 한 번에 병렬로 계산한다.
        B, T, _ = x.shape
        return x.view(B, T, self.h, self.d_k).transpose(1, 2)

    def forward(self, x_q, x_kv, mask=None):
        # self-attention이면 x_q와 x_kv에 같은 텐서를 넣는다.
        # cross-attention이면 x_q는 디코더 쪽, x_kv는 인코더 출력 z를 넣는다.
        q, k, v = self.split(self.w_q(x_q)), self.split(self.w_k(x_kv)), self.split(self.w_v(x_kv))
        out, weights = attention(q, k, v, mask)  # out: (B, h, T, d_k)
        B, _, T, _ = out.shape
        # (B, h, T, d_k) -> (B, T, h, d_k) -> (B, T, h*d_k): 헤드 h개의 결과를 이어 붙인다(Concat)
        out = out.transpose(1, 2).reshape(B, T, self.h * self.d_k)
        return self.w_o(out), weights  # W^O로 섞어서 다시 (B, T, d_model)로 돌려준다


def positional_encoding(max_len, d_model):
    """사인·코사인 위치 인코딩 (논문 3.5절)

    PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pos = torch.arange(max_len).unsqueeze(1)  # (max_len, 1): 위치 0, 1, 2, ...
    # 1 / 10000^(2i/d_model)을 차원 쌍마다 계산한다 -> (d_model/2,)
    # exp(log(.))로 바꿔 쓴 것은 큰 거듭제곱을 직접 계산하지 않으려는 흔한 구현 방식이다.
    div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)  # 짝수 번째 차원에는 sin
    pe[:, 1::2] = torch.cos(pos * div)  # 홀수 번째 차원에는 cos
    return pe


# ---- 1. 직접 구현한 attention이 PyTorch 내장 함수와 같은 값을 내는지 확인 ----
# 배치 2개, 헤드 4개, 문장 길이 6, 헤드당 차원 16인 무작위 q, k, v를 만든다.
B, h, T, d_k = 2, 4, 6, 16
q, k, v = torch.randn(B, h, T, d_k), torch.randn(B, h, T, d_k), torch.randn(B, h, T, d_k)
# 아래 삼각 행렬(대각선 포함)만 True인 causal 마스크.
# i번째 query는 0..i번째 key만 볼 수 있다. 디코더 self-attention이 쓰는 마스크다.
causal = torch.tril(torch.ones(T, T, dtype=torch.bool))

ours, w = attention(q, k, v, causal)
ref = F.scaled_dot_product_attention(q, k, v, attn_mask=causal)  # PyTorch 내장 구현(정답 기준)
print("max |ours - torch| :", f"{(ours - ref).abs().max().item():.2e}")  # 부동소수점 오차 수준이면 같은 값이다
assert torch.allclose(ours, ref, atol=1e-5)
# 대각선 위쪽(미래 위치)의 가중치가 전부 0인지 확인한다.
assert torch.all(w.triu(diagonal=1) == 0), "미래 위치에 attention이 새면 안 된다"
# 4번째 query(인덱스 3)는 앞의 네 위치만 보므로 뒤의 두 값이 0이어야 한다.
print("4번째 query의 가중치:", [round(x, 3) for x in w[0, 0, 3].tolist()])

# ---- 2. 멀티헤드 attention의 입출력 모양 확인 ----
d_model, heads = 64, 8  # 헤드당 차원은 64 / 8 = 8
mha = MultiHeadAttention(d_model, heads)
x = torch.randn(B, T, d_model)
y, w = mha(x, x)  # self-attention: query와 key·value가 같은 입력에서 나온다
print("입력", tuple(x.shape), "-> 출력", tuple(y.shape), "/ 가중치", tuple(w.shape))
# 출력은 입력과 같은 모양이어야 잔차 연결(x + Sublayer(x))로 더할 수 있다.
assert y.shape == x.shape and w.shape == (B, heads, T, T)
# 헤드마다, query마다 가중치를 key 방향으로 더하면 1이어야 한다(softmax의 성질).
assert torch.allclose(w.sum(-1), torch.ones(B, heads, T)), "가중치 합은 1이어야 한다"

# ---- 3. 왜 sqrt(d_k)로 나누는가: 내적의 분산은 d_k에 비례한다 ----
# 성분이 평균 0, 분산 1인 무작위 벡터 쌍 1만 개의 내적을 구해 분산을 잰다.
# 이론상 내적의 분산은 d_k이고 sqrt(d_k)로 나누면 1이 된다(논문 각주 4).
for d in (16, 64, 512):
    a, b = torch.randn(10000, d), torch.randn(10000, d)
    dots = (a * b).sum(-1)  # 행마다 a·b 내적 -> (10000,)
    print(f"d_k={d:>3}  내적 분산={dots.var().item():7.1f}  sqrt(d_k)로 나눈 뒤={(dots / math.sqrt(d)).var().item():.2f}")

# ---- 4. 위치 인코딩: 같은 거리만큼 떨어진 위치끼리는 내적이 같다 ----
# sin·cos 덧셈 정리 때문에 PE[p]·PE[p+k] = sum_i cos(k * w_i) 이고 p와는 상관이 없다.
# 즉 두 위치 사이의 "거리"만 내적에 남는다. 거리 3인 두 쌍(10,13), (30,33)으로 확인한다.
pe = positional_encoding(50, d_model)
d1 = (pe[10] @ pe[13]).item()
d2 = (pe[30] @ pe[33]).item()
print(f"PE[10]·PE[13] = {d1:.4f},  PE[30]·PE[33] = {d2:.4f}")
assert abs(d1 - d2) < 1e-4
print("모든 검사를 통과했다.")  # 위의 assert가 하나라도 실패하면 이 줄까지 오지 않는다
