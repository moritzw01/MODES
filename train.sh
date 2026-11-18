#!/bin/bash
#SBATCH --job-name=modes_training       # Name des Jobs in der Slurm-Queue
#SBATCH --output=/home/apxye/multimodal/ID3/logs/%x_%j.out # Datei für Standard-Ausgabe (%x=Jobname, %j=Job-ID)
#SBATCH --error=/home/apxye/multimodal/ID3/logs/%x_%j.err          # Datei für Fehlermeldungen
#SBATCH --partition=debug     
#SBATCH --gres=gpu:1                    # Anzahl der GPUs (falls benötigt, sonst Zeile löschen)
#SBATCH --cpus-per-task=8               # Anzahl der CPU-Kerne
#SBATCH --mem=32G                       # Arbeitsspeicher
#SBATCH --time=24:00:00                 # Maximale Laufzeit (HH:MM:SS)

# Bricht das Skript sofort ab, wenn ein Befehl fehlschlägt
set -e

# Log-Verzeichnis erstellen, falls es noch nicht existiert
mkdir -p logs

echo "Job gestartet auf Node: $(hostname)"
echo "Aktuelles Verzeichnis: $(pwd)"
uv sync

echo "Starte main.py..."
uv run python -u main.py \
    --log_path "/home/apxye/multimodal/ID3/logs/out.txt" \
    --plot_path "/home/apxye/multimodal/ID3/plots" \
    --ckpt_path "/home/apxye/multimodal/ID3/checkpoints" \
    --data_dir "/home/apxye/multimodal/ID3/m3di" \
    --zm_size 400 \
    --zs_size 600 \
    --beta 1e-6 \
    --gamma 1 \
    --n_samples 1000 \
    --n_epochs 1 \
    --iter_count 5 \
    --modality_names M1 M2 \
    --model "mutual_info"
