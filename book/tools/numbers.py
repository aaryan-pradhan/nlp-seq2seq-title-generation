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

print('verify: ok')
