"""Dump SASS summary of a z7 kernel instantiation (CPU): exp_r7_sass.py <src> <name-substring>."""
import glob
import os
import subprocess
import sys

import importlib
import exp_r7_zip_run as R

Z = importlib.import_module(sys.argv[1])
ext = R.build(Z)
so = ext.__file__
pat = sys.argv[2]
out = subprocess.run(["cuobjdump", "-sass", so], capture_output=True, text=True).stdout
blocks = out.split("Function : ")
for b in blocks[1:]:
    name = b.split("\n", 1)[0]
    if pat in name:
        lines = [l for l in b.split("\n") if "/*" in l and ";" in l]
        print("=====", name, len(lines), "instrs")
        for l in lines:
            ins = l.split("*/", 1)[1].strip() if "*/" in l else l
            if any(k in ins for k in ("LDG", "HMMA", "BRA", "BAR", "STS", "LDS", "EXIT", "PRMT", "LOP3")):
                print(ins[:110])
