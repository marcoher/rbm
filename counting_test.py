from __future__ import division
from model2 import RBM2
import numpy as np

n = 15

samples = 400

X = np.zeros((samples, n, 2), dtype=np.int32)

for i in xrange(samples):
    m = np.random.randint(1, n//2)
    j1 = np.random.choice(n, size=(m), replace=False)
    j2 = np.random.choice(n, size=(m), replace=False)
    X[i, j1, 0] = 1
    X[i, j2, 1] = 1

print X[:5,:,0]
print X[:5,:,1]

print np.sum(X[:5,:,0], axis=(1))
print np.sum(X[:5,:,1], axis=(1))

Xtrain = X[:300,:,:]
Xeval = X[300:,:,:]

brbm1 = RBM2(n, 70)

brbm1.fit(Xtrain, X_eval=Xeval, max_epochs=50000, cd_steps=1, lr=0.05, verbose=100)

p1 = np.array([0,1,0,1,0,1,0,0,1,1,0,0,0,0,0])
print "Target number of on units: %s" % np.sum(p1)

p2 = brbm1.generate(p1, samples=50, iters=15)
q = np.sum(p2, axis=1)
print "Number of on units: %s +/- %s" % (np.mean(q), np.std(q))
