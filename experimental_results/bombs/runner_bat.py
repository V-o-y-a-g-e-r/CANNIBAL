import subprocess

data_path = "others/pavia.npy"
ref_map_path = "others/pavia_gt.npy"
base_dest_path = "bombs/pavia_out_"

for bands_per_antibody in range(5, 103, 5):
    dest_path = f"{base_dest_path}{bands_per_antibody}"

    command = [
        "python",
        "-m",
        "bombs.runner",
        "--bands_per_antibody",
        str(bands_per_antibody),
        "--data_path",
        data_path,
        "--ref_map_path",
        ref_map_path,
        "--dest_path",
        dest_path,
    ]

    subprocess.run(command)
