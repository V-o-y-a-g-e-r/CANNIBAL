import subprocess

data_path = r"C:\Users\luktu\Downloads\gawll (1)\others\pavia.npy"
ref_map_path = r"C:\Users\luktu\Downloads\gawll (1)\others\pavia_gt.npy"
base_dest_path = "mi/pavia_out"

for band_num in range(5, 103, 5):
    dest_path = f"{base_dest_path}{band_num}"

    command = [
        "python",
        "-m",
        "mi.runner",
        "--bands_num",
        str(band_num),
        "--data_path",
        data_path,
        "--ref_map_path",
        ref_map_path,
        "--dest_path",
        dest_path,
    ]

    subprocess.run(command)
