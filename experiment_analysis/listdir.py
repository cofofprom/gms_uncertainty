import os
import pickle
import argparse

def print_config(run_folder):
    print(f"=== {run_folder} ===")

    with open(os.path.join(run_folder, 'config'), 'r') as f:
        for line in f:
            print('    ', line.strip())

    print()

def main():
    parser = argparse.ArgumentParser("listdir", description="This script walks though given experiment folder and prints configs of each run")
    parser.add_argument("dir", action='store', type=str)

    args = parser.parse_args()

    for run_dir in os.listdir(args.dir):
        rundir_path = os.path.join(args.dir, run_dir)
        if not os.path.isdir(rundir_path): continue

        print_config(rundir_path)

if __name__ == '__main__':
    main()