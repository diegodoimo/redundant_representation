import argparse
import os
import requests

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--download_dir", type=str, default="./models")
    parser.add_argument("--dataset", default = 'cifar10', type=str, choices=["cifar10"])
    parser.add_argument("--model", default = 'densenet40', type=str, choices=["densnet40"])
    parser.add_argument("--w", default = '256', type=str, choices=["256"])
    args = parser.parse_args([])
    return args


def main(args):
    if not os.path.exists(args.download_dir):
        os.makedirs(args.download_dir)

    urls = ['https://figshare.com/ndownloader/files/37506373']

    for url in urls:
        r = requests.get(url, stream=True)
        with open(f"{args.download_dir}/{args.dataset}_{args.model}_{args.w}.pth.tar", "wb") as f:
            f.write(r.content)

if __name__ == "__main__":
    args = parse_arguments()
    main(args)
