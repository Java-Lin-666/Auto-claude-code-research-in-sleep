"""Modal launcher for GradVax experiments."""
import modal
from pathlib import Path

app = modal.App("gradvax")

BASE_DIR = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "numpy", "tqdm", "Pillow")
    .add_local_dir(
        str(BASE_DIR),
        remote_path="/workspace",
        ignore=["**/__pycache__/**", "**/*.pyc", "data/cache/**", "results/**", "logs/**"],
    )
)

results_vol = modal.Volume.from_name("gradvax-results-v2", create_if_missing=True)
data_vol = modal.Volume.from_name("gradvax-data", create_if_missing=True)

STEPS = 131072
DATA = "/data/cache"
LOGS = "/results"


@app.function(
    image=image,
    gpu="T4",
    timeout=18000,  # 5 hours — enough for 131K steps on T4
    volumes={"/results": results_vol, "/data": data_vol},
    max_containers=8,
)
def run_experiment(cmd_args: list):
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "/workspace/train.py"] + cmd_args,
        cwd="/workspace",
        check=False,
    )
    results_vol.commit()
    return result.returncode


def make_args(dataset, gamma, n_labeled, run_id, seed=42, **flags):
    args = [
        "--dataset", dataset,
        "--gamma", str(gamma),
        "--n_labeled", str(n_labeled),
        "--total_steps", str(STEPS),
        "--eval_every", "500",
        "--log_dir", LOGS,
        "--run_id", run_id,
        "--seed", str(seed),
        "--data_root", DATA,
    ]
    for k, v in flags.items():
        if v is True:
            args.append(f"--{k}")
        elif v is not False and v is not None:
            args += [f"--{k}", str(v)]
    return args


# All experiment configs
EXPERIMENTS = {
    # Sanity check (1000 steps)
    "sanity": [
        "--dataset", "cifar100", "--gamma", "100", "--n_labeled", "500",
        "--total_steps", "1000", "--eval_every", "500",
        "--log_dir", LOGS, "--run_id", "sanity_check_s42",
        "--seed", "42", "--data_root", DATA,
    ],

    # Phase A (baseline diagnostic — same run as C1b_base, reuse results)
    # A1 shares run_id with C1b_base, so only run once under C1b_base

    # Phase B
    "B1_oracle_base": make_args("cifar100", 100, 500, "cifar100_g100_oracle_baseline_s42", oracle=True),
    "B1_oracle_gv":   make_args("cifar100", 100, 500, "cifar100_g100_oracle_gradvax_s42", gradvax=True, oracle=True),
    # B2_gradvax shares run_id with C1b_gv, so only run once under C1b_gv

    # Phase C1
    "C1a_base":   make_args("cifar10",  100, 250, "cifar10_g100_baseline_s42"),
    "C1a_gv":     make_args("cifar10",  100, 250, "cifar10_g100_gradvax_s42", gradvax=True),
    "C1b_base":   make_args("cifar100", 100, 500, "cifar100_g100_baseline_s42"),
    "C1b_gv":     make_args("cifar100", 100, 500, "cifar100_g100_gradvax_s42", gradvax=True),
    "C1c_base":   make_args("cifar100", 150, 500, "cifar100_g150_baseline_s42"),
    "C1c_gv":     make_args("cifar100", 150, 500, "cifar100_g150_gradvax_s42", gradvax=True),

    # Phase C2
    "C2_downweight": make_args("cifar100", 100, 500, "cifar100_g100_head_downweight_s42", head_downweight=0.5),
    "C2_clip":       make_args("cifar100", 100, 500, "cifar100_g100_grad_clip_s42", grad_clip=1.0),

    # Phase D1
    "D1_full":      make_args("cifar100", 100, 500, "cifar100_g100_gradvax_full_s42", gradvax=True),
    "D1_nodom":     make_args("cifar100", 100, 500, "cifar100_g100_gradvax_nodom_s42", gradvax=True, no_domination=True),
    "D1_noconflict":make_args("cifar100", 100, 500, "cifar100_g100_gradvax_noconflict_s42", gradvax=True, no_conflict=True),
    "D1_noema":     make_args("cifar100", 100, 500, "cifar100_g100_gradvax_noema_s42", gradvax=True, no_ema=True),

    # Phase D2 (tau sweep)
    "D2_tau2":   make_args("cifar100", 100, 500, "cifar100_g100_gradvax_tau2_s42", gradvax=True, tau=2),
    "D2_tau5":   make_args("cifar100", 100, 500, "cifar100_g100_gradvax_tau5_s42", gradvax=True, tau=5),
    "D2_tau10":  make_args("cifar100", 100, 500, "cifar100_g100_gradvax_tau10_s42", gradvax=True, tau=10),
    "D2_tauinf": make_args("cifar100", 100, 500, "cifar100_g100_gradvax_tauinf_s42", gradvax=True, tau=1e9),

    # Phase E (short overhead run — override total_steps)
    "E1_base": [
        "--dataset", "cifar100", "--gamma", "100", "--n_labeled", "500",
        "--total_steps", "2000", "--eval_every", "500",
        "--log_dir", LOGS, "--run_id", "cifar100_g100_overhead_baseline_s42",
        "--seed", "42", "--data_root", DATA,
    ],
    "E1_gv": [
        "--dataset", "cifar100", "--gamma", "100", "--n_labeled", "500",
        "--total_steps", "2000", "--eval_every", "500",
        "--log_dir", LOGS, "--run_id", "cifar100_g100_overhead_gradvax_s42",
        "--seed", "42", "--data_root", DATA, "--gradvax",
    ],
}


@app.local_entrypoint()
def main(phase: str = "sanity"):
    """Run experiments. phase: A|B|C|D|E|all or comma-separated keys."""
    phase_map = {
        "A": ["C1b_base"],  # A1 reuses cifar100_g100_baseline_s42
        "B": ["B1_oracle_base", "B1_oracle_gv", "C1b_gv"],  # B2 reuses cifar100_g100_gradvax_s42
        "C": ["C1a_base", "C1a_gv", "C1b_base", "C1b_gv", "C1c_base", "C1c_gv",
              "C2_downweight", "C2_clip"],
        "D": ["D1_full", "D1_nodom", "D1_noconflict", "D1_noema",
              "D2_tau2", "D2_tau5", "D2_tau10", "D2_tauinf"],
        "E": ["E1_base", "E1_gv"],
    }

    if phase == "all":
        keys = list(EXPERIMENTS.keys())
    elif "," in phase:
        keys = [k.strip() for k in phase.split(",")]
    elif phase in phase_map:
        keys = phase_map[phase]
    elif phase in EXPERIMENTS:
        keys = [phase]
    else:
        print(f"Unknown phase: {phase}. Use A/B/C/D/E/all or a specific key.")
        return

    # Override total_steps for E phase (already set in make_args above)
    configs = [(k, EXPERIMENTS[k]) for k in keys if k in EXPERIMENTS]
    print(f"Launching {len(configs)} experiments: {[k for k, _ in configs]}")

    for key, rc in zip(
        [k for k, _ in configs],
        run_experiment.map([args for _, args in configs], order_outputs=True)
    ):
        status = "OK" if rc == 0 else f"FAILED (rc={rc})"
        print(f"  {key}: {status}")

    print("Done. Results in Modal volume 'gradvax-results'.")
