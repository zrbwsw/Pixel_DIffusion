"""Guard the historical training recipe and safe launch/resume boundaries."""

import fcntl
from pathlib import Path

import pytest
import torch

import main_rmsnorm2_jit_only as entry


PROJECT = Path(__file__).resolve().parents[1]
CONFIG = PROJECT / "configs/rmsnorm2_residual_0_1_B80_gbs1024.yaml"


def recipe(*overrides):
    return entry.parse_args(["--config", str(CONFIG), *overrides])


def checkpoint_payload(saved, epoch=20):
    weights = {"jit_pre_blend_norm.weight": torch.ones(2)}
    return {"args": saved, "epoch": epoch, "model": weights,
            "model_ema1": weights.copy(), "model_ema2": weights.copy(),
            "optimizer": {"state": {0: {"step": torch.tensor(10.)}},
                          "param_groups": [{"params": [0]}]}}


def test_recipe_matches_the_completed_checkpoint_arguments():
    checkpoint = (
        PROJECT.parent / "Ours_BaseT_rmsnorm2_Residual_1016/checkpoints"
        / "BaseT-Residual(t,1-t)-NoDinoPre-PgT-blend_rmsnorm2-NoDDTHead-B-80"
        / "checkpoint-80.pth"
    )
    if not checkpoint.is_file():
        pytest.skip("The local baseline checkpoint is not distributed with source code")
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    saved = state["args"]
    saved = saved if isinstance(saved, dict) else vars(saved)
    current = vars(recipe())
    assert state["epoch"] == 80
    ignored = {
        "config", "exp_name", "output_dir", "resume", "metrics_log_name",
        "data_path", "dino_weights", "lr", "world_size", "rank", "local_rank",
        "gpu", "distributed", "dist_backend", "dist_url", "preflight",
        "dino_generation_path_enabled", "jit_input_rmsnorm",
    }
    assert set(saved) - ignored == set(current) - ignored
    assert {k: v for k, v in saved.items() if k not in ignored} == {
        k: v for k, v in current.items() if k not in ignored
    }
    assert saved["world_size"] == 4
    assert current["batch_size"] * 4 * current["accum_iter"] == 1024
    assert saved["lr"] == pytest.approx(current["blr"] * 1024 / 256)
    for group in state["optimizer"]["param_groups"]:
        assert group["betas"] == (0.9, 0.95)
        assert group["eps"] == 1e-8 and group["weight_decay"] == 0


def test_online_evaluation_remains_the_baseline_protocol():
    args = recipe()
    entry.verify_recipe(args)
    assert args.online_eval and not args.evaluate_gen
    assert (args.sampling_method, args.num_sampling_steps, args.cfg) == ("heun", 50, 1.0)
    assert (args.interval_min, args.interval_max) == (0.0, 1.0)
    assert (args.num_images, args.gen_bsz, args.eval_freq) == (50000, 64, 20)
    assert (args.ema_decay1, args.ema_decay2) == (0.9999, 0.9996)
    assert not args.keep_eval_images and not args.resume_eval_images
    assert not args.skip_online_eval and args.save_eval_npz is None
    assert args.resume == "" and not args.dino_generation_path_enabled


def test_no_input_norm_matches_completed_jit_only_recipe(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    config = PROJECT / "configs/jit_only_no_input_rmsnorm_B80_gbs1024_8gpu.yaml"
    args = entry.parse_args(["--config", str(config)])
    entry.verify_recipe(args)
    assert not args.jit_input_rmsnorm and not args.dino_generation_path_enabled
    assert args.resume == "" and args.start_epoch == 0
    assert (args.batch_size, args.global_batch_size, args.accum_iter) == (128, 1024, 1)
    checkpoint = PROJECT / "checkpoints/BaseT-Residual(0,1)-NoDinoPre-PgT-blend_rmsnorm2-NoDDTHead-B-80/checkpoint-80.pth"
    if not checkpoint.exists():
        pytest.skip("Completed local JiT-only checkpoint is not distributed")
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=False)
    saved = state["args"] if isinstance(state["args"], dict) else vars(state["args"])
    ignored = {"config", "exp_name", "output_dir", "resume", "metrics_log_name", "preflight", "lr",
               "world_size", "rank", "local_rank", "gpu", "distributed", "dist_backend", "dist_url",
               "jit_input_rmsnorm", "start_epoch"}  # Completed control resumed at epoch 61; ablation starts fresh.
    assert {k:v for k,v in saved.items() if k not in ignored} == {
        k:v for k,v in vars(args).items() if k not in ignored}
    assert saved["lr"] == pytest.approx(args.blr * args.global_batch_size / 256)


@pytest.mark.parametrize("override", [
    ("--blr", "0.000025"), ("--batch_size", "128"),
    ("--num_sampling_steps", "100"), ("--gen_bsz", "128"),
    ("--cfg", "2.7"), ("--ema_decay1", "0.9996"),
    ("--hflip_prob", "0"), ("--dino_jit_input_mode", "native"),
    ("--no-dino_jit_pre_blend_rmsnorm",),
])
def test_single_factor_guard_rejects_other_recipe_changes(override):
    with pytest.raises(ValueError, match="Single-factor recipe differs"):
        entry.verify_recipe(recipe(*override))


def test_single_factor_guard_rejects_reenabling_generation():
    with pytest.raises(ValueError, match="requires --no-dino_generation_path_enabled"):
        entry.verify_recipe(recipe("--dino_generation_path_enabled"))


@pytest.fixture
def runtime_assets(tmp_path, monkeypatch):
    import dino_v3_lora
    import engine_jit

    data = tmp_path / "imagenet"
    (data / "train").mkdir(parents=True)
    weights = tmp_path / "dino.pth"
    weights.write_bytes(b"local weights are resolved, never loaded by preflight")
    monkeypatch.setenv("IMAGENET_ROOT", str(data))
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setattr(dino_v3_lora, "_resolve_weights", lambda *args: weights)
    monkeypatch.setattr(dino_v3_lora, "_resolve_repo", lambda: PROJECT / "SSL_code/dinov3")
    monkeypatch.setattr(engine_jit, "online_evaluation_assets", lambda size: ("statistics.npz", "inception.pth"))
    monkeypatch.setattr(entry, "verify_reference", lambda: {"source_commit": "verified-reference"})
    # Resume checks require the saved experiment ID to match its directory.
    # Keep all fake checkpoints inside a temporary project, never real runs.
    monkeypatch.setattr(entry, "PROJECT_ROOT", tmp_path)
    checkpoint_root = tmp_path / "checkpoints"
    checkpoint_root.mkdir()
    return checkpoint_root / recipe().exp_name


def test_preflight_is_read_only_and_starts_from_scratch(runtime_assets):
    output = runtime_assets
    report = entry.check_runtime(recipe("--output_dir", str(output)))
    assert not output.exists()
    assert report["output_dir"] == str(output.resolve())
    assert report["training_mode"] == "from_scratch" and report["resume_checkpoint"] is None
    assert (report["world_size"], report["batch_size_per_gpu"], report["accum_iter"]) == (4, 256, 1)
    assert report["global_batch_size"] == 1024 and report["effective_lr"] == pytest.approx(2e-4)


def test_preflight_refuses_wrong_world_size(runtime_assets, monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "8")
    with pytest.raises(ValueError, match="requires 4 x 256"):
        entry.check_runtime(recipe("--output_dir", str(runtime_assets)))


def test_resume_refuses_another_experiment_or_missing_checkpoint(runtime_assets):
    output = runtime_assets
    with pytest.raises(ValueError, match="must share this project's checkpoint root"):
        entry.check_runtime(recipe("--output_dir", str(output), "--resume", str(output.parent.parent / "baseline")))
    with pytest.raises(FileNotFoundError):
        entry.check_runtime(recipe("--output_dir", str(output), "--resume", str(output)))


@pytest.mark.parametrize("norm_enabled", [True, False])
def test_historical_resume_defaults_to_rmsnorm_and_cannot_cross_ablations(runtime_assets, norm_enabled):
    output = runtime_assets
    output.mkdir()
    args = recipe("--output_dir", str(output), "--resume", str(output))
    saved = vars(args).copy()
    saved.pop("jit_input_rmsnorm")  # Historical control did not have this flag.
    saved.update(lr=0.0002, world_size=4)
    torch.save(checkpoint_payload(saved, 20), output / "checkpoint-last.pth")
    args.jit_input_rmsnorm = norm_enabled
    if norm_enabled:
        assert entry.check_runtime(args)["jit_input_rmsnorm"] is True
    else:
        with pytest.raises(ValueError, match="Resume configuration mismatch: jit_input_rmsnorm"):
            entry.check_runtime(args)


@pytest.mark.parametrize("mutation", [None, "generation", "sampling", "lr", "complete"])
def test_resume_only_accepts_its_own_matching_incomplete_run(runtime_assets, mutation):
    output = runtime_assets
    output.mkdir()
    args = recipe("--output_dir", str(output), "--resume", str(output))
    saved = vars(args).copy()
    saved.update(lr=0.0002, world_size=4, rank=0, gpu=0, distributed=True, dist_backend="nccl")
    epoch = 20
    if mutation == "generation":
        saved["dino_generation_path_enabled"] = True
    elif mutation == "sampling":
        saved["num_sampling_steps"] = 100
    elif mutation == "lr":
        saved["lr"] = 0.0001
    elif mutation == "complete":
        epoch = 80
    torch.save(checkpoint_payload(saved, epoch), output / "checkpoint-last.pth")
    if mutation is not None:
        with pytest.raises(ValueError, match="Resume configuration mismatch|Invalid or already completed"):
            entry.check_runtime(args)
    else:
        report = entry.check_runtime(args)
        assert report["training_mode"] == "resume"
        assert report["resume_checkpoint"] == str(output / "checkpoint-last.pth")


@pytest.mark.parametrize("saved_world,saved_batch", [(4, 256), (8, 128)])
def test_eight_gpu_resume_preserves_global_batch_and_recipe(runtime_assets, monkeypatch, saved_world, saved_batch):
    output = runtime_assets
    output.mkdir()
    saved = vars(recipe("--output_dir", str(output), "--resume", str(output))).copy()
    saved.update(lr=0.0002, world_size=saved_world, batch_size=saved_batch)
    torch.save(checkpoint_payload(saved, 61), output / "checkpoint-last.pth")
    monkeypatch.setenv("WORLD_SIZE", "8")
    args = recipe("--batch_size", "128", "--output_dir", str(output), "--resume", str(output))
    report = entry.check_runtime(args)
    assert (report["world_size"], report["batch_size_per_gpu"], report["global_batch_size"]) == (8, 128, 1024)
    assert report["effective_lr"] == pytest.approx(2e-4)
    assert report["resume_state"]["completed_epochs"] == 61
    assert report["resume_state"]["saved_world_size"] == saved_world
    assert report["resume_state"]["restores"] == ["model", "optimizer", "model_ema1", "model_ema2"]
    assert report["sampling"]["gen_bsz_per_gpu"] == 64


@pytest.mark.parametrize("saved_world,saved_batch", [(4, 128), (8, 256), (16, 64)])
def test_resume_rejects_incompatible_saved_layout(runtime_assets, saved_world, saved_batch):
    output = runtime_assets
    output.mkdir()
    args = recipe("--output_dir", str(output), "--resume", str(output))
    saved = vars(args).copy()
    saved.update(lr=0.0002, world_size=saved_world, batch_size=saved_batch)
    torch.save(checkpoint_payload(saved), output / "checkpoint-last.pth")
    with pytest.raises(ValueError, match="saved GPU/batch layout"):
        entry.check_runtime(args)


@pytest.mark.parametrize("missing", ["model", "optimizer", "model_ema1", "model_ema2"])
def test_resume_requires_model_adamw_and_both_emas(runtime_assets, missing):
    output = runtime_assets
    output.mkdir()
    args = recipe("--output_dir", str(output), "--resume", str(output))
    saved = vars(args).copy()
    saved.update(lr=0.0002, world_size=4)
    state = checkpoint_payload(saved)
    del state[missing]
    torch.save(state, output / "checkpoint-last.pth")
    with pytest.raises(ValueError, match="Incomplete resume checkpoint"):
        entry.check_runtime(args)


@pytest.mark.parametrize("rank", [0, 1])
def test_only_rank_zero_owns_output_lock_during_training(tmp_path, monkeypatch, rank):
    output = tmp_path / "run"
    output.mkdir()
    # These are exactly the files the foreground launcher writes before torchrun.
    for name in ("train.log", "launcher.log", "run_info.txt", "resource_preflight.json",
                 "config_base.yaml", "source_sha256.txt", "started_at.txt", "train_args.txt"):
        (output / name).write_text("launcher metadata")
    args = recipe("--output_dir", str(output))
    monkeypatch.chdir(PROJECT)
    monkeypatch.setenv("RANK", str(rank))
    monkeypatch.setattr(entry, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(entry, "parse_args", lambda argv: args)
    monkeypatch.setattr(entry, "check_runtime", lambda args: {"output_dir": str(output)})
    monkeypatch.setattr(entry.reference.main_jit, "Denoiser", entry.reference.main_jit.Denoiser)
    monkeypatch.setattr(entry.reference.main_jit, "HFImageDataset", entry.reference.main_jit.HFImageDataset)
    called = []

    def fake_training(actual):
        called.append(actual)
        lock_path = tmp_path / ".runtime_locks/run.lock"
        if rank == 0:
            with lock_path.open("a") as other:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            assert not lock_path.exists()

    monkeypatch.setattr(entry.reference.main_jit, "main", fake_training)
    entry.main([])
    assert called == [args]


def test_rank_zero_refuses_existing_training_output(tmp_path, monkeypatch):
    output = tmp_path / "run"
    output.mkdir()
    (output / "checkpoint-last.pth").write_bytes(b"existing checkpoint must not be overwritten")
    args = recipe("--output_dir", str(output))
    monkeypatch.chdir(PROJECT)
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(entry, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(entry, "parse_args", lambda argv: args)
    monkeypatch.setattr(entry, "check_runtime", lambda args: {"output_dir": str(output)})
    monkeypatch.setattr(entry.reference.main_jit, "Denoiser", entry.reference.main_jit.Denoiser)
    monkeypatch.setattr(entry.reference.main_jit, "HFImageDataset", entry.reference.main_jit.HFImageDataset)
    monkeypatch.setattr(entry.reference.main_jit, "main", lambda args: pytest.fail("Must reject before training"))
    with pytest.raises(FileExistsError, match="resume explicitly"):
        entry.main([])


@pytest.mark.parametrize("mutation", [None, "source_identity", "target_identity", "existing_output"])
def test_b160_continuation_uses_matched_checkpoint_in_same_project(runtime_assets, monkeypatch, mutation):
    source = runtime_assets
    source.mkdir()
    saved = vars(recipe("--output_dir", str(source))).copy()
    saved.update(lr=0.0002, world_size=8, batch_size=128)
    if mutation == "source_identity":
        saved["exp_name"] = "different-experiment"
    torch.save(checkpoint_payload(saved, epoch=80), source / "checkpoint-last.pth")
    monkeypatch.setenv("WORLD_SIZE", "8")
    config = PROJECT / "configs/rmsnorm2_residual_0_1_B160_gbs1024_8gpu.yaml"
    args = entry.parse_args(["--config", str(config)])
    # Relative paths stay valid after a project-directory rename.
    assert args.resume == f"checkpoints/{source.name}"
    output = source.parent / args.exp_name
    if mutation == "target_identity":
        output = source.parent / "different-target"
        args.output_dir = str(output)
    if mutation == "existing_output":
        output.mkdir()
        (output / "checkpoint-last.pth").write_bytes(b"must not overwrite")
    if mutation is not None:
        expected = {
            "source_identity": "Resume experiment identity",
            "target_identity": "output must match",
            "existing_output": "already has checkpoints",
        }
        with pytest.raises(ValueError, match=expected[mutation]):
            entry.check_runtime(args)
    else:
        report = entry.check_runtime(args)
        assert report["output_dir"] == str(output)
        assert report["resume_checkpoint"] == str(source / "checkpoint-last.pth")
        assert report["resume_state"]["completed_epochs"] == 80
        assert report["epochs"] == 160 and report["effective_lr"] == pytest.approx(2e-4)
        assert not output.exists()
