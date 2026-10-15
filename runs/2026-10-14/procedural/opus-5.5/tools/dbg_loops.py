import bpy, sys, numpy as np
sys.argv=[sys.argv[0]]
exec(open("build.py").read().replace('if __name__ == "__main__":\n    main()',''))
import builtins
orig = boundary_loops
def bl(F):
    loops = orig(F)
    V = None
    print("NLOOPS", len(loops), [len(l) for l in loops])
    return loops
boundary_loops = bl
os.environ["STOP_AFTER"]="sweater"
STOP_AFTER="sweater"
main()
