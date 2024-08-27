import pandas as pd


if __name__ == "__main__":

    comp_vals = [
        {
            "Method": "nnU-Net ResEnc L (iso)",
            "Patch Size": "192x192x192",
            "Spacing": "1.0x1.0x1.0",
            "Batch Size": 2,
            "Dataset": "BTCV",
            "DSC": 84.01,
        },
        {
            "Method": "nnU-Net ResEnc L",
            "Patch Size": "80x256x256",
            "Spacing": "3x0.76x0.76",
            "Batch Size": 2,
            "Dataset": "BTCV",
            "DSC": 83.35,
        },
        {
            "Method": "MedNeXt k3",
            "Patch Size": "128x128x128",
            "Spacing": "1x1x1",
            "Batch Size": 2,
            "Dataset": "BTCV",
            "DSC": 84.70,
        },
        {
            "Method": "nnU-Net ResEnc L (iso)",
            "Patch Size": "96x256x256",
            "Spacing": "1.0x1.0x1.0",
            "Batch Size": 3,
            "Dataset": "ACDC",
            "DSC": 92.64,
        },
        {
            "Method": "nnU-Net ResEnc L",
            "Patch Size": "20x256x224",
            "Spacing": "5x1.56x1.56",
            "Batch Size": 10,
            "Dataset": "ACDC",
            "DSC": 91.69,
        },
        {
            "Method": "MedNeXt k3",
            "Patch Size": "128x128x128",
            "Spacing": "1x1x1",
            "Batch Size": 2,
            "Dataset": "ACDC",
            "DSC": 92.65,
        },
    ]
    df = pd.DataFrame(comp_vals)
    df = df.set_index(['Dataset', 'Method'])
    print(df.to_latex(index=True))
    print(0)
