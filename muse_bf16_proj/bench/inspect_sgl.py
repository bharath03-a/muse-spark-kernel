import os
import glob

def main():
    print("Searching for hopper_bf16_gemv...")
    for p in glob.glob("/sgl-workspace/**/hopper_bf16_gemv*", recursive=True):
        print("Found:", p)
        if p.endswith((".cuh", ".cu", ".h", ".cpp")):
            print(f"=== {p} ===")
            print(open(p).read())

if __name__ == "__main__":
    main()
