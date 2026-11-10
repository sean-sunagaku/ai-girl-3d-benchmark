import sys, os
sys.path.insert(0, os.getcwd())
import bpy, numpy as np, collections
from fable.base import Body
b = Body('input'); V = b.V0; enc = b.enclosed
M = np.array([0, -0.13, 1.478])
near = np.where(np.linalg.norm(V - M, axis=1) < 0.06)[0]
start = [i for i in near if not enc[i] and any(enc[j] for j in b.adj[i]) and V[i,1] < -0.125 and 1.465 < V[i,2] < 1.49 and abs(V[i,0]) < 0.03]
ring = {i: 0 for i in start}; q = collections.deque(start)
while q:
    i = q.popleft()
    for j in b.adj[i]:
        if j not in ring and not enc[j]:
            ring[j] = ring[i] + 1; q.append(j)
for k in range(0, 9):
    ids = [i for i, r in ring.items() if r == k and V[i,0] >= -1e-6]
    up = sorted([i for i in ids if V[i,2] > 1.4777 + (0.0 if k else 0)], key=lambda i: V[i,0])
    lo = sorted([i for i in ids if V[i,2] <= 1.4777], key=lambda i: V[i,0])
    print('ring', k, 'n', len(ids))
    print('  up:', ' '.join('(%.1f,%.1f,%.1f)' % (V[i,0]*1000, (V[i,2]-1.4777)*1000, V[i,1]*1000) for i in up))
    print('  lo:', ' '.join('(%.1f,%.1f,%.1f)' % (V[i,0]*1000, (V[i,2]-1.4777)*1000, V[i,1]*1000) for i in lo))
