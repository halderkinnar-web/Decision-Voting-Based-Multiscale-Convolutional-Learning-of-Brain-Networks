import argparse
from brainnet_dv.config import Config
from brainnet_dv.experiment import run_experiment

def main():
    parser = argparse.ArgumentParser(description="Run multiscale decision-voting brain-network experiments.")
    parser.add_argument("--config", default="configs/config.json")
    parser.add_argument("--data", default=None, help="Path to the HDF5-based .mat dataset.")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    cfg = Config.from_json(args.config)
    if args.data:
        cfg.data_path = args.data
    if args.output:
        cfg.output_dir = args.output
    run_experiment(cfg)

if __name__ == "__main__":
    main()
