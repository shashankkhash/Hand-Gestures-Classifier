# NumPy port of Nueral_network.py: same architecture, init, per-sample SGD, eta=0.1
import json, os, sys, time
import numpy as np
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
def load(lst):
    X, y, names = [], [], []
    for f in open(os.path.join(R, lst)).read().split():
        p = os.path.join(R, f)
        if not os.path.exists(p): continue
        X.append(np.array(Image.open(p), dtype=float).ravel() / 255); y.append(1 if 'down' in f else 0); names.append(f)
    return np.array(X), np.array(y, float), names
Xtr, ytr, ntr = load('downgesture_train.list'); Xte, yte, nte = load('downgesture_test.list')
sig = lambda s: 1 / (1 + np.exp(-s))
rng = np.random.default_rng(7)
d, H, eta = Xtr.shape[1], 100, 0.1
W1 = (2 * rng.random((H, d + 1)) - 1) / 100; W2 = (2 * rng.random(H + 1) - 1) / 100
def fwd(X):
    Xb = np.hstack([np.ones((len(X), 1)), X]); h = np.hstack([np.ones((len(X), 1)), sig(Xb @ W1.T)])
    return sig(h @ W2)
hist = []
t = time.time()
for ep in range(1, 1001):
    for i in range(len(Xtr)):
        x = np.concatenate([[1], Xtr[i]]); h = np.concatenate([[1], sig(W1 @ x)]); o = sig(h @ W2)
        dL = 2 * (o - ytr[i]) * (o - o * o)
        dh = (h - h * h) * W2 * dL
        W1 -= eta * np.outer(dh[1:], x); W2 -= eta * h * dL
    if ep in (1,2,3,5) or ep % 10 == 0:
        ptr, pte = fwd(Xtr), fwd(Xte)
        hist.append(dict(epoch=ep, train_acc=float(((ptr >= .5) == ytr).mean()), test_acc=float(((pte >= .5) == yte).mean()),
                         train_mse=float(((ptr - ytr) ** 2).mean()), test_mse=float(((pte - yte) ** 2).mean())))
pte = fwd(Xte)
pred = (pte >= .5).astype(int)
cm = dict(tp=int(((pred == 1) & (yte == 1)).sum()), fp=int(((pred == 1) & (yte == 0)).sum()),
          fn=int(((pred == 0) & (yte == 1)).sum()), tn=int(((pred == 0) & (yte == 0)).sum()))
wrong = [dict(file=n, p=float(p), label=int(l)) for n, p, l in zip(nte, pte, yte) if (p >= .5) != l]
json.dump(dict(n_train=len(Xtr), n_test=len(Xte), pos_train=int(ytr.sum()), pos_test=int(yte.sum()), hist=hist, cm=cm, wrong=wrong,
               seconds=time.time() - t), open(os.path.join(HERE, 'results.json'), 'w'), indent=1)
print(hist[-1], cm, len(wrong), time.time() - t)
