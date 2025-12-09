#!/bin/bash

# Clear all experiments in wandb directory
# Usage: ./scripts/clear_wandb.sh

WANDB_DIR="./wandb"

# Check if wandb directory exists
if [ ! -d "$WANDB_DIR" ]; then
    echo "Warning: $WANDB_DIR directory does not exist, no cleanup needed"
    exit 0
fi

# Check if directory is empty
if [ -z "$(ls -A $WANDB_DIR)" ]; then
    echo "Info: $WANDB_DIR directory is empty, no cleanup needed"
    exit 0
fi

# Display contents to be deleted
echo "=========================================="
echo "Ready to clear all experiments in wandb directory"
echo "=========================================="
echo "Directory: $WANDB_DIR"
echo "Contents:"
ls -lh "$WANDB_DIR" | head -10
if [ $(ls -1 "$WANDB_DIR" | wc -l) -gt 10 ]; then
    echo "... (more files)"
fi
echo "=========================================="

# Ask for confirmation
read -p "Are you sure you want to delete all contents? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Operation cancelled"
    exit 0
fi

# Delete all contents
echo "Deleting..."
rm -rf "$WANDB_DIR"/*
rm -rf "$WANDB_DIR"/.* 2>/dev/null  # Delete hidden files, ignore errors

# Check if deletion was successful
if [ -z "$(ls -A $WANDB_DIR 2>/dev/null)" ]; then
    echo "✓ Successfully cleared wandb directory"
else
    echo "Warning: Some files may not have been deleted"
    ls -la "$WANDB_DIR"
fi

