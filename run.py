
import os
import sys
import numpy as np
import torch

from models.nafnet_architecture import NAFNetSR


def load_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    checkpoint_path = os.path.join(
        os.path.dirname(__file__),
        "models",
        "nafnet_hf_final_checkpoint.pth"
    )

    model = NAFNetSR(
        img_channel=1,
        width=32,
        enc_blocks=(2, 2, 4),
        middle_blocks=4,
        dec_blocks=(2, 2, 2)
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model = model.to(device)
    model.eval()

    return model, device


def main():

    if len(sys.argv) != 3:

        print(
            "Usage: python run.py <input-dir> <output-dir>"
        )

        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.isdir(input_dir):

        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    model, device = load_model()

    files = sorted(
        f
        for f in os.listdir(input_dir)
        if f.endswith(".npy")
    )

    if len(files) == 0:

        raise RuntimeError(
            "No .npy files found in input directory."
        )

    print(
        f"Device: {device}"
    )

    print(
        f"Found {len(files)} input files."
    )

    with torch.no_grad():

        for filename in files:

            input_path = os.path.join(
                input_dir,
                filename
            )

            output_path = os.path.join(
                output_dir,
                filename
            )

            arr = np.load(
                input_path
            ).astype(
                np.float32
            )

            if arr.ndim == 3 and arr.shape[-1] == 1:
                arr = arr[..., 0]

            if arr.ndim != 2:

                raise ValueError(
                    f"{filename}: expected "
                    f"(H,W) or (H,W,1), got {arr.shape}"
                )

            if not np.isfinite(arr).all():

                raise ValueError(
                    f"{filename}: input contains NaN/Inf"
                )

            x = torch.from_numpy(
                arr
            ).unsqueeze(0).unsqueeze(0)

            x = x.to(
                device
            )

            pred = model(x)

            pred = pred.clamp(
                0.0,
                1.0
            )

            output = (
                pred[0, 0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

            if not np.isfinite(output).all():

                raise ValueError(
                    f"{filename}: output contains NaN/Inf"
                )

            if output.ndim != 2:

                raise ValueError(
                    f"{filename}: output shape "
                    f"{output.shape} is not (H,W)"
                )

            if output.min() < 0.0 or output.max() > 1.0:

                raise ValueError(
                    f"{filename}: output outside [0,1]"
                )

            np.save(
                output_path,
                output
            )

    print(
        f"Successfully generated {len(files)} outputs."
    )


if __name__ == "__main__":
    main()
