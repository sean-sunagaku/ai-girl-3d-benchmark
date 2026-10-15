import bpy, numpy as np
cv = bpy.data.objects["Hair_Strands"].data
pos = np.empty(len(cv.points)*3, np.float32); cv.position_data.foreach_get("vector", pos); pos=pos.reshape(-1,3)
off = np.empty(len(cv.curves), np.int32); cv.curves.foreach_get("first_point_index", off)
sizes = np.diff(np.append(off, len(pos)))
wing=[]
for i,(o,n) in enumerate(zip(off,sizes)):
    P=pos[o:o+n]
    if ((P[:,0] > 0.10) & (P[:,2] > 1.50)).any():
        wing.append(i)
print("strands", len(off), "wing", len(wing))
for i in wing[:25]:
    P=pos[off[i]:off[i]+sizes[i]]
    j=np.argmax(P[:,0]*(P[:,2]>1.50))
    print("root", P[0].round(3), "maxX pt", P[j].round(3), "k", j, "/", len(P), "tip", P[-1].round(3))
import sys
sys.argv=[sys.argv[0]]
src=open("build.py").read().split("def main():")[0]
exec(src)
T=pose_definition(); Ai=np.array(T["head"].inverted())
for i in wing[:8]:
    P=pos[off[i]]; q=Ai[:3,:3]@P+Ai[:3,3]; print("local root", q.round(4), "density", scalp_density(q[None])[0])
