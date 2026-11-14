"""Every number in the Extra revision part, recomputed (VERIFY in book.mk). One assert per number in the text."""
from math import exp, isclose

r = lambda x, n: round(x, n)

# ch 08: softmax over four candidate next tokens for "The capital of France is"
z = {'Paris': 4, 'Lyon': 2, 'the': 1, 'banana': -2}
e = {k: exp(v) for k, v in z.items()}
assert [r(e[k], 2) for k in z] == [54.60, 7.39, 2.72, 0.14]
s = sum(e.values()); assert r(s, 2) == 64.84
p = {k: e[k] / s for k in z}
assert [r(p[k], 3) for k in z] == [0.842, 0.114, 0.042, 0.002]
assert isclose(sum(p.values()), 1)
# ch 08: step-by-step answer in the fine-tuning example, 3x + 5 = 20
assert 20 - 5 == 15 and 15 / 3 == 5

# ch 09: word-level IDs (first appearance), character IDs, BPE merges, windows, GPT-3 mix
c = "The cat runs . The cat jumps . The cat sleeps . The dog runs . The dog jumps . The dog sleeps .".split()
v = {}
for w in c: v.setdefault(w, len(v) + 1)
assert len(c) == 24 and len(v) == 7
assert [v[w] for w in c] == [1,2,3,4, 1,2,5,4, 1,2,6,4, 1,7,3,4, 1,7,5,4, 1,7,6,4]
a = {ch: i for i, ch in enumerate("abcdefghijklmnopqrstuvwxyz")}; a[' '] = 26
assert [a[ch] for ch in "how are you"] == [7,14,22,26,0,17,4,26,24,14,20]
from collections import Counter
words, merges = [list(w) for w in ("low", "lowest", "newer", "wider")], []
def apply(w, m):
    o, i = [], 0
    while i < len(w):
        if i + 1 < len(w) and (w[i], w[i+1]) == m: o.append(w[i] + w[i+1]); i += 2
        else: o.append(w[i]); i += 1
    return o
for _ in range(3):
    cnt = Counter(p for w in words for p in zip(w, w[1:]))
    m = max(cnt, key=cnt.get); merges.append(m); words = [apply(w, m) for w in words]
assert merges == [('l','o'), ('lo','w'), ('e','r')]
w = list("lower")
for m in merges: w = apply(w, m)
assert w == ['low', 'er']
N, n, B = 60, 10, 2
k = (N - 1) // n; assert k == 5 and k * n == 50 and -(-k // B) == 3
assert max(j for j in range(10) if n*j + 11 <= N) == 4
assert r(0.60 * 300 / 410, 2) == 0.44

# ch 10: ordinal distances, analogy cosines, lookup, positional add, shapes
assert abs(22 - 21) == 1 and abs(21 - 6) == 15
from math import sqrt
king, man, woman = (0.9,0.8,0.9,0.7,0.6), (0.9,0.2,0.3,0.7,0.6), (0.1,0.2,0.3,0.5,0.6)
queen, cat = (0.1,0.8,0.8,0.5,0.6), (0.5,0.0,0.0,0.1,0.0)
rv = tuple(round(a - b + c, 2) for a, b, c in zip(king, man, woman)); assert rv == (0.1,0.8,0.9,0.5,0.6)
dot = lambda a, b: sum(x * y for x, y in zip(a, b)); nrm = lambda a: sqrt(dot(a, a))
assert r(dot(rv, rv), 2) == 2.07 and r(dot(queen, queen), 2) == 1.90 and r(dot(cat, cat), 2) == 0.26
assert [r(nrm(x), 3) for x in (rv, queen, cat)] == [1.439, 1.378, 0.510]
assert r(dot(rv, queen), 2) == 1.98 and r(dot(rv, cat), 2) == 0.10
assert r(dot(rv, queen) / nrm(rv) / nrm(queen), 3) == 0.998 and r(dot(rv, cat) / nrm(rv) / nrm(cat), 3) == 0.136
E = [(0.1,0.3,-0.2),(0.5,-0.4,0.2),(0.2,0.1,0.6),(0.7,-0.1,0.4),(-0.3,0.8,0.1),(0.0,0.2,-0.5),(0.4,0.4,0.3),(0.6,0.0,0.5)]
o = [0,0,0,1,0,0,0,0]
fox = tuple(sum(o[i] * E[i][j] for i in range(8)) for j in range(3)); assert fox == (0.7,-0.1,0.4)
assert tuple(r(a + b, 2) for a, b in zip(fox, (0.1, 0.0, -0.2))) == (0.8, -0.1, 0.2)
assert 6 * 768 == 4608 and r(50257 / 768, 1) == 65.4

# ch 11: three-token attention (full and causal), softmax saturation, layer norm, sizes
from math import inf
Qm = [(1,0),(0,1),(1,1)]; Vm = [(1,0),(0,1),(1,1)]
S = [[dot(q, k) / sqrt(2) for k in Qm] for q in Qm]
assert [r(x, 3) for x in S[0]] == [0.707, 0.0, 0.707] and r(S[2][2], 3) == 1.414
def soft(row):
    m = max(row); e = [exp(x - m) if x > -inf else 0 for x in row]; t = sum(e); return [x / t for x in e]
assert r(exp(S[0][0]), 3) == 2.028 and r(2 * exp(S[0][0]) + 1, 3) == 5.056 and r(exp(S[2][2]), 3) == 4.113
assert r(2 * exp(S[0][0]) + exp(S[2][2]), 3) == 8.169 and r(1 + exp(S[0][0]), 3) == 3.028
blend = lambda w: tuple(r(sum(w[j] * Vm[j][c] for j in range(3)), 3) for c in range(2))
w1 = soft(S[0]); assert [r(x, 3) for x in w1] == [0.401, 0.198, 0.401] and blend(w1) == (0.802, 0.599)
A = [soft([x if j <= i else -inf for j, x in enumerate(S[i])]) for i in range(3)]
assert [[r(x, 3) for x in row] for row in A] == [[1, 0, 0], [0.330, 0.670, 0], [0.248, 0.248, 0.503]]
assert [blend(row) for row in A] == [(1, 0), (0.330, 0.670), (0.752, 0.752)]
assert [r(x, 3) for x in soft([1, 2, 3])] == [0.090, 0.245, 0.665] and [r(x, 3) for x in soft([10, 20, 30])] == [0, 0, 1]
xs = [2, 4, 6, 8]; mu = sum(xs) / 4; var = sum((x - mu) ** 2 for x in xs) / 4
assert mu == 5 and var == 5 and r(sqrt(var), 3) == 2.236
assert [r((x - mu) / sqrt(var), 3) for x in xs] == [-1.342, -0.447, 0.447, 1.342]
assert 512 // 4 == 128 and 4 * 128 == 512 and 768 // 12 == 64 and 4 * 768 == 3072
assert 1024 ** 2 == 1048576 and 2048 ** 2 == 4194304 == 4 * 1048576

# ch 12: temperature on two tokens, top-k / top-p / combined on ten logits
two = lambda T: 1 / (1 + exp(-1 / T))
assert r(1 + exp(-1), 3) == 1.368
assert [r(two(T), 3) for T in (0.1, 0.5, 1, 2, 10)] == [1.000, 0.881, 0.731, 0.622, 0.525]
z10 = [3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.0, -0.5, -1.0, -1.5]
def softT(z, T=1): e = [exp(x / T) for x in z]; t = sum(e); return [x / t for x in e]
def renorm(q): t = sum(q); return [x / t for x in q]
def topp(q, p):
    run, out = 0, []
    for x in q:
        run += x; out.append(r(run, 3))
        if run >= p: return out
P1 = softT(z10)
assert [r(x, 3) for x in P1] == [0.396, 0.240, 0.146, 0.088, 0.054, 0.033, 0.020, 0.012, 0.007, 0.004]
assert r(sum(P1[:5]), 3) == 0.924 and [r(x, 3) for x in renorm(P1[:5])] == [0.429, 0.260, 0.158, 0.096, 0.058]
assert topp(P1, 0.9) == [0.396, 0.636, 0.782, 0.871, 0.924]
assert [r(sum(P1[:i]), 3) for i in range(6, 11)] == [0.957, 0.976, 0.988, 0.996, 1.000]
P8 = softT(z10, 0.8)
assert [r(x, 3) for x in P8] == [0.466, 0.249, 0.133, 0.071, 0.038, 0.020, 0.011, 0.006, 0.003, 0.002]
K8 = renorm(P8[:5]); assert [r(x, 3) for x in K8] == [0.486, 0.260, 0.139, 0.075, 0.040]
assert topp(K8, 0.9) == [0.486, 0.746, 0.886, 0.960]
assert [r(x, 3) for x in renorm(K8[:4])] == [0.506, 0.271, 0.145, 0.078]

# ch 13: parameter count, steps, loss, GELU, gradient descent, memory
from math import log, erf
V, d, nctx, L = 50257, 768, 1024, 12
assert V * d == 38597376 and nctx * d == 786432 and 4 * d * d == 2359296 and 8 * d * d == 4718592
assert 12 * d * d == 7077888 and L * 12 * d * d == 84934656
N = V * d + nctx * d + L * 12 * d * d; assert N == 124318464 and N + V * d == 162915840
assert 1024 * 32 == 32768 and r(50e6 / 32768, 1) == 1525.9 and 1526 * 32 * 1024 == 50003968
assert (60 - 1) // 10 == 5 and -(-5 // 2) == 3 and 16 * 2 == 32
assert r(-log(0.70), 3) == 0.357
ls = [-log(q) for q in (0.01, 0.05, 0.02, 0.03, 0.01)]
assert [r(x, 3) for x in ls] == [4.605, 2.996, 3.912, 3.507, 4.605] and r(sum(r(x, 3) for x in ls), 3) == 19.625 and r(sum(ls) / 5, 2) == 3.92
assert r(log(V), 2) == 10.82 and r(log(100), 2) == 4.61 and r(exp(-3), 3) == 0.050 and round(exp(3)) == 20
Phi = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
assert r(Phi(1), 3) == 0.841 and r(Phi(-1), 3) == 0.159
w0 = 5; g = 2 * (w0 - 3); assert g == 4 and (w0 - 3) ** 2 == 4
assert r(w0 - 0.1 * g, 2) == 4.6 and r((w0 - 0.1 * g - 3) ** 2, 2) == 2.56
assert r(w0 - 1.1 * g, 2) == 0.6 and r((w0 - 1.1 * g - 3) ** 2, 2) == 5.76
assert 10**9 * 4 == 4e9 and 4 + 4 + 8 == 16 and 16 + 10 == 26 and 16 + 15 == 31
assert N * 4 == 497273856 and r(N * 4 / 1e9, 2) == 0.50 and 16 * 10**10 == 160e9

# ch 14: scores with and without the cache, cached step 3, cache bytes
assert sum(t * t for t in range(1, 5)) == 30 and sum(range(1, 5)) == 10
sq, li = sum(t * t for t in range(1, 1025)), sum(range(1, 1025))
assert sq == 358438400 and li == 524800 and round(sq / li) == 683
q3 = (1, 1); sc = [dot(q3, k) / sqrt(2) for k in Qm]
assert [r(x, 3) for x in sc] == [0.707, 0.707, 1.414] and r(sqrt(2), 3) == 1.414
assert [r(x, 3) for x in soft(sc)] == [0.248, 0.248, 0.503] and blend(soft(sc)) == (0.752, 0.752)
kv = lambda L, nkv, dh, n, B=1, b=2: 2 * L * nkv * dh * n * B * b
assert kv(12, 12, 64, 1024) == 37748736 and kv(12, 12, 64, 1024) / 2**20 == 36
big = kv(60, 128, 128, 100000); assert big == 393216000000 and round(big / 1e9) == 393 and round(big / 2**30) == 366
assert big / 80e9 > 4

# ch 15: KV-cache memory with n_kv key-value heads
assert [2 * nkv * 4 * 128 for nkv in (4, 2, 1)] == [4096, 2048, 1024] and 512 // 4 == 128
assert r(big / 1e9, 1) == 393.2 and r(kv(60, 8, 128, 100000) / 1e9, 1) == 24.6 == r(393.2 / 16, 1)
assert r(kv(60, 1, 128, 100000) / 1e9, 2) == 3.07 and r(393.2 / 128, 2) == 3.07 and 128 // 8 == 16
assert kv(12, 12, 64, 1) == 36864 == 36 * 1024 and kv(12, 12, 64, 1024, 8) / 2**20 == 288

print('verify: ok')
