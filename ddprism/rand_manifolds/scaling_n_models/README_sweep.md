# How to run a sweep with this code.

### 1. Create a Sweep

```bash
wandb sweep --project rand-manifolds-cont sweep_config.yaml
```

This will create a new wandb sweep and display the sweep ID, `sweep_id_value`. This need to be passed to sbatch scripts.

### 2. Run Individual Trials

```bash
sbatch --export=WANDB_SWEEP_ID=sweep_id_value script.sh ...
```