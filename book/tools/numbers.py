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

print('verify: ok')
