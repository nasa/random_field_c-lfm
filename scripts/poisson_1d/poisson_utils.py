from pathlib import Path

from clfm.utils import get_epoch_and_step_for_checkpoint


def get_result_and_model_path(case_params, result_dir):
    dir, name = get_logging_path_and_name(str(result_dir), case_params)
    result_path = dir / name / f'version_{case_params["version"]}'
    epoch, step = get_epoch_and_step_for_checkpoint(result_path)
    model_path = result_path / f"checkpoints/epoch={epoch}-step={step}.ckpt"
    return result_path, model_path


def get_logging_path_and_name(root_dir, params_dict):
    nn_keys = [
        "latent_dim",
        "h_trunk",
        "h_branch",
        "h_encoder",
        "nhl_trunk",
        "nhl_branch",
        "nhl_encoder",
        "grad_clip",
        "bs",
        "lr",
    ]
    run_keys = [
        "N_data",
        "num_sensors",
        "num_colloc",
        "res_weight",
        "kld_weight",
        "epochs",
        "seed",
    ]
    nn_dir = "-".join(f"{k}={params_dict[k]}" for k in nn_keys)
    run_name = "-".join(f"{k}={params_dict[k]}" for k in run_keys)
    path = Path(f"{root_dir}/{nn_dir}")
    return path, run_name
