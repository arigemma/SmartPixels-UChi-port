from sklearn.preprocessing import StandardScaler
import joblib
import yaml
import argparse
import numpy as np
from dataset import SmartPixelsDataset
from utils import SymLogTransform, CreateFolder


def main(args):

    print(f"Deriving scaling for {args.data_config}...")

    config_dict = yaml.safe_load(open(args.data_config, "r"))

    image_path = config_dict['root'] + config_dict['train'][0][0]
    label_path = config_dict['root'] + config_dict['train'][0][1]

    ds = SmartPixelsDataset(
        [image_path],
        [label_path],
        shuffle=False,
        # important: pre-transform with symlog
        transform=SymLogTransform(),
    )

    X, Y = [], []
    for i, (x, y) in enumerate(ds):
        X.append(x)
        Y.append(y)
    X = np.stack(X)
    Y = np.stack(Y)

    print(f"Data loaded: X shape: {X.shape}, Y shape: {Y.shape}")

    X = X.reshape(-1, 1)

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    scaler_X.fit(X)
    scaler_y.fit(Y)

    print(f"Input scaler shape: {scaler_X.mean_.shape}, mean:\n{scaler_X.mean_}, std:\n{np.sqrt(scaler_X.var_)}")
    print(f"Output scaler shape: {scaler_y.mean_.shape}, mean:\n{scaler_y.mean_}, std:\n{np.sqrt(scaler_y.var_)}")

    # 2. Save both scalers
    joblib.dump(scaler_X, f'{args.prefix}/scaler_X.pkl')
    joblib.dump(scaler_y, f'{args.prefix}/scaler_y.pkl')


if __name__ == "__main__":
    """
    Compute normalization parameters for SmartPixels dataset.
    Use a single file for both input and target scaling parameters.
    """

    parser = argparse.ArgumentParser(description="Compute normalization parameters for SmartPixels dataset.")
    parser.add_argument(
        "--data-config", "-c", type=str, default="config/v3.yml"
    )
    parser.add_argument("--prefix", type=str, action=CreateFolder, default="scalers", help="Output file prefix.")
    args = parser.parse_args()
    main(args)
