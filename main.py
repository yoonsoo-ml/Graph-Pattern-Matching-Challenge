import argparse
from backtracker import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--query_path', type=str)
    parser.add_argument('--cs_path', type=str)
    args = parser.parse_args()
    backtracker = BacktrackerV10(args.data_path, args.query_path, args.cs_path)
    backtracker.run()
