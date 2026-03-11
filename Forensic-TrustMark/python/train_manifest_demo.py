import argparse

from trustmark.data import ManifestDataModule


def parse_args():
    parser = argparse.ArgumentParser(description="Manifest dataloader demo for TrustMark training.")
    parser.add_argument(
        "--train-manifest",
        default="data_manifests/celeba_hq_128_800_train.csv",
        help="Path to train CSV manifest with an img_path column.",
    )
    parser.add_argument(
        "--val-manifest",
        default="data_manifests/celeba_hq_128_100_val.csv",
        help="Path to val CSV manifest with an img_path column.",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--resolution", type=int, default=128)
    parser.add_argument("--secret-len", type=int, default=100)
    parser.add_argument("--cover-key", default="image")
    parser.add_argument("--secret-key", default="secret")
    return parser.parse_args()


def main():
    args = parse_args()
    dm = ManifestDataModule(
        train_manifest_csv=args.train_manifest,
        val_manifest_csv=args.val_manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        resolution=args.resolution,
        secret_len=args.secret_len,
        cover_key=args.cover_key,
        secret_key=args.secret_key,
        pin_memory=False,
    )
    dm.setup("fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    print("Train image shape:", tuple(train_batch[args.cover_key].shape))
    print("Train secret shape:", tuple(train_batch[args.secret_key].shape))
    print("Train sample path:", train_batch["img_path"][0])
    print("Val image shape:", tuple(val_batch[args.cover_key].shape))
    print("Val secret shape:", tuple(val_batch[args.secret_key].shape))

    # In your training script, pass `dm` to lightning Trainer:
    # trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
