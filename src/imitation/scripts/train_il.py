from argparse import ArgumentParser
from pathlib import Path

import torch

from src.components.utils.paths import CONFIG_DIR, IL_DATASET_DIR, MODEL_WEIGHTS
from src.components.utils.config_loading import (
    load_normalized_config,
    derive_experiment_tag,
)
from src.imitation.imitation_agent import ImitationAgent
from src.imitation.il_train_runner import ILTrainRunner
from src.imitation.utils.il_dataset import ImitationLearningDataset


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Path to a config file (e.g. a generated Slurm config). Defaults to configs/config.json."
    )
    parser.add_argument(
        "--map_version",
        type=str,
        default=None,
        help="[Deprecated] Backwards compatible alias for --experiment_tag."
    )
    parser.add_argument(
        "--experiment_tag",
        type=str,
        default=None,
        help="Name of the experiment folder under data/model_weights/. Derived from the config path if omitted."
    )
    parser.add_argument(
        "--weights_save_folder",
        type=str,
        default=None,
        help="Directory for saving model weights. If None, a default folder is used."
    )
    parser.add_argument(
        "--checkpoint_path_load",
        type=str, default=None,
        help="Path to load model weights from. If None, training starts from scratch."
    )
    parser.add_argument(
        "--dataset_map_version",
        type=str,
        default=None,
        help="[Deprecated] Backwards compatible alias for --dataset_tag."
    )
    parser.add_argument(
        "--dataset_tag",
        type=str,
        default=None,
        help="Folder under data/il_dataset to read. Defaults to the derived experiment tag."
    )
    parser.add_argument("--epochs", type=int, default=70, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--neural_slam_lr_mapper",
        type=float,
        default=None,
        help="Learning rate for Neural SLAM Mapper network. Uses --lr if not specified."
    )
    parser.add_argument(
        "--neural_slam_lr_pose",
        type=float,
        default=None,
        help="Learning rate for Neural SLAM Pose Estimator network. Uses --lr if not specified."
    )
    parser.add_argument(
        "--neural_slam_weight",
        type=float,
        default=1.0,
        help="Weight for Neural SLAM loss relative to IL loss."
    )
    args = parser.parse_args()


    config_path = Path(args.config_path) if args.config_path else CONFIG_DIR / "config.json"
    config = load_normalized_config(config_path)

    derived_tag = derive_experiment_tag(config_path, config)
    experiment_tag = args.experiment_tag or args.map_version or derived_tag
    experiment_tag = experiment_tag or "default"

    dataset_tag = args.dataset_tag or args.dataset_map_version or experiment_tag

    print(f"[INFO]: Loaded config from: {config_path}")
    print(f"[INFO]: Experiment tag: {experiment_tag}")
    print(f"[INFO]: Dataset tag   : {dataset_tag}")

    print("\nLoaded navigation configs:")
    for k, v in config["navigation"].items():
        print(f"  {k}: {v}")
    print("\n----------------------")
    print("\nLoaded exploration configs:")
    for k, v in config["exploration"].items():
        print(f"  {k}: {v}")

    data_dir = IL_DATASET_DIR / dataset_tag
    print(f"\n[INFO]: Using dataset tag: {dataset_tag}")

    # Check if we're using Neural SLAM
    is_neural_slam = config["exploration"].get("map_version") == "neural_slam"
    if is_neural_slam:
        print("[INFO]: Neural SLAM mode enabled")

        # Add Neural SLAM configs to the main configs if not present
        if "neural_slam" not in config["exploration"]:
            config["exploration"]["neural_slam"] = {
                "vision_range_cm": 320,
                "cell_size_cm": 25,
                "vision_range_cells": 64,
                "sensor_noise": {
                    "position_std": 0.02,
                    "rotation_std": 2.0
                },
                "training": {
                    "mapper_lr": args.neural_slam_lr_mapper or args.lr,
                    "pose_estimator_lr": args.neural_slam_lr_pose or args.lr,
                    "batch_size": args.batch_size,
                    "sequence_length": 2,
                    "loss_weight": args.neural_slam_weight
                }
            }

        # Update learning rates from command line
        neural_slam_config = config["exploration"]["neural_slam"]["training"]
        if args.neural_slam_lr_mapper:
            neural_slam_config["mapper_lr"] = args.neural_slam_lr_mapper
        if args.neural_slam_lr_pose:
            neural_slam_config["pose_estimator_lr"] = args.neural_slam_lr_pose

        neural_slam_config["loss_weight"] = args.neural_slam_weight

    default_weights_folder = MODEL_WEIGHTS / experiment_tag
    weights_save_folder = Path(args.weights_save_folder) if args.weights_save_folder else default_weights_folder
    weights_save_folder.mkdir(parents=True, exist_ok=True)

    print(f"[INFO]: Saving weights into: {weights_save_folder}")
    print(f"[INFO]: Using data from: {data_dir}")

    # Load dataset
    dataset = ImitationLearningDataset(data_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize agent with Neural SLAM support
    agent = ImitationAgent(config=config, num_actions=dataset.num_actions, device=device)

    # Add Neural SLAM configs to agent if needed
    if is_neural_slam:
        agent.neural_slam_config = config["exploration"]["neural_slam"]["training"]

    # Load checkpoint if specified
    if args.checkpoint_path_load:
        checkpoint_path = Path(args.checkpoint_path_load)
        if checkpoint_path.exists():
            if checkpoint_path.is_dir():
                # Directory containing separate network weights
                encoder_path = checkpoint_path / "feature_encoder.pth"  # Look for encoder
                if not encoder_path.exists():
                    # Try pattern-based encoder file
                    encoder_files = list(checkpoint_path.glob("feature_encoder_*.pth"))
                    encoder_path = encoder_files[0] if encoder_files else None

                neural_slam_path = checkpoint_path / "neural_slam_networks" if is_neural_slam else None

                if encoder_path and encoder_path.exists():
                    agent.load_weights(str(encoder_path),
                                       neural_slam_path=str(neural_slam_path) if neural_slam_path else None,
                                       device=device)
                    print(f"Weights loaded from: {checkpoint_path}")
                else:
                    print(f"WARNING: Encoder weights not found at {checkpoint_path}")
            else:
                # Single file checkpoint
                agent.load_state_dict(torch.load(str(checkpoint_path), map_location=device))
                print(f"Weights loaded from: {checkpoint_path}")
        else:
            print(f"WARNING: Checkpoint not found at {checkpoint_path}. Training will start from scratch.")

    print(f"\n[INFO] Starting training for {args.epochs} epochs")
    print(f"[INFO] Weights will be saved in: {weights_save_folder}")

    if is_neural_slam:
        neural_slam_config = config["exploration"]["neural_slam"]["training"]
        print(f"[INFO] Neural SLAM training enabled:")
        print(f"  - Mapper LR: {neural_slam_config['mapper_lr']}")
        print(f"  - Pose Estimator LR: {neural_slam_config['pose_estimator_lr']}")
        print(f"  - Loss Weight: {neural_slam_config['loss_weight']}")
    print()

    # Initialize trainer
    runner = ILTrainRunner(agent, dataset, device=device, lr=args.lr, batch_size=args.batch_size)

    # Run training
    runner.run(num_epochs=args.epochs, save_folder=str(weights_save_folder))

    # Additional Neural SLAM specific logging/saving
    if is_neural_slam and hasattr(agent, 'neural_slam_networks') and agent.neural_slam_networks:
        print(f"\n[INFO] Neural SLAM training completed")
        print(f"[INFO] Neural SLAM networks saved in: {weights_save_folder}/neural_slam_networks/")


if __name__ == "__main__":
    main()