
import argparse
import json
import os
import sys
import requests
from tqdm import tqdm

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_dir", type=str, default="./models")
    parser.add_argument("--model", default = 'densenet40', type=str, choices=["densnet40"])
    args = parser.parse_args()
    print("input args:\n", json.dumps(vars(args), indent=4, separators=(",", ":")))
    return args

    urls = ['https://figshare.com/ndownloader/files/37506373']

    for url in urls:
        r = requests.get(url, stream=True)
        with open(f"{args.download_dir}/{filename}", "wb") as f:
            f.write(r.content)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
