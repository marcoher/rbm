from __future__ import division
from model import rbm
import numpy as np

n = 12
m = 3
samples = n * (n - 1) // 3
data = np.zeros((samples, n), dtype=np.int32)
for i in xrange(samples):
    j = np.random.choice(n,size=(m))
    data[i,j] = 1

print data[:5]

print np.sum(data[:5], axis=1)

rbm1 = rbm(n, n + 5, visible_type="binary", seed=1111, name="model_1")

rbm1.fit(data, max_epochs=50000, cd_steps=10, learning_rate=0.1, verbose=500)

print "Dreams with 1 iteration:"
dreams1 =  rbm1.generate(num_samples=10, iters=1)
print dreams1
print "Counts:"
print np.sum(dreams1, axis=1)

print "Dreams with 10 iterations:"
dreams2 = rbm1.generate(num_samples=10, iters=10)
print dreams2
print "Counts:"
print np.sum(dreams2, axis=1)

print "Dreams with 15 iterations:"
dreams3 = rbm1.generate(num_samples=10, iters=15)
print dreams3
print "Counts:"
print np.sum(dreams3, axis=1)

print "Vaccuum free energy:"
f0 = rbm1.free_energy(np.zeros((1,n), dtype=np.int32))
print f0

fenergy1 = rbm1.free_energy(data)
print "Free energy of data:"
print fenergy1
print "Free energy of data relative to F(0):"
print fenergy1 - f0

fenergy2 = rbm1.free_energy(dreams1)
print "Free energy of dreams1 relative to F(0):"
print fenergy2 - f0

fenergy3 = rbm1.free_energy(dreams2)
print "Free energy of dreams2 relative to F(0):"
print fenergy3 - f0

fenergy4 = rbm1.free_energy(dreams3)
print "Free energy of dreams3 relative to F(0):"
print fenergy4 - f0
