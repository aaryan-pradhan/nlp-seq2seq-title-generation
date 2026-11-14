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

print('verify: ok')
