#!/bin/bash

set -e


DATA_PATH="./data/financial_transactions"
MODEL_NAME="financial_transactions_synth"
EPOCHS=1000
BATCH_SIZE=8
GPUS=1

# source venv/bin/activate

pip install -r requirements.txt

mkdir -p gan_models gan_results

echo "Starting GAN training..."
echo "Dataset: $DATA_PATH"
echo "Epochs: $EPOCHS"
echo "Batch Size: $BATCH_SIZE"

CUDA_VISIBLE_DEVICES=$GPUS python - <<END
from src.phase2_differential_privacy.gan_manager import StyleGANTrainer

if __name__ == "__main__":
    trainer = StyleGANTrainer(
        data_path="$DATA_PATH",
        results_dir="gan_results",
        models_dir="gan_models"
    )

    config = trainer.train(
        num_epochs=$EPOCHS,
        batch_size=$BATCH_SIZE,
        lr=0.002
    )

    print("Training completed with config:", config)
END

echo "Training finished. Models saved to gan_models/"
