import numpy as np
#import timedomain_shaker


for i in range(17):
    frequency = 200 + i * 50
    print("\n\nFrequency: %e\n" % frequency)
    exec(open("examples/timedomain_shaker.py").read())