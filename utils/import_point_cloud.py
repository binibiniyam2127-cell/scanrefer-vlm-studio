import os
import argparse
import numpy as np


def parse_ply(ply_path: str) -> np.ndarray:
    """
    Parses a 3D .ply file (ASCII or Binary Little Endian) and extracts [X, Y, Z, R, G, B].
    """
    with open(ply_path, "rb") as f:
        # Parse header
        line = f.readline().decode("ascii", errors="ignore").strip()
        if line != "ply":
            raise ValueError(f"File {ply_path} is not a valid PLY format.")

        format_type = ""
        vertex_count = 0
        properties = []

        while True:
            line = f.readline().decode("ascii", errors="ignore").strip()
            if line.startswith("format"):
                format_type = line.split()[1]
            elif line.startswith("element vertex"):
                vertex_count = int(line.split()[2])
            elif line.startswith("property"):
                parts = line.split()
                properties.append((parts[1], parts[2]))
            elif line == "end_header":
                break

        print(f"Loading {vertex_count} vertices (Format: {format_type})...")

        if format_type == "ascii":
            data = []
            for _ in range(vertex_count):
                vals = [float(v) for v in f.readline().decode("ascii").split()]
                data.append(vals)
            data = np.array(data, dtype=np.float32)
        elif format_type == "binary_little_endian":
            # Map property types to numpy dtypes
            type_map = {
                "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
                "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
                "int": "<i4", "int32": "<i4", "uint": "<u4", "uint32": "<u4"
            }
            dtype_fields = [(name, type_map.get(t, "<f4")) for t, name in properties]
            struct_dtype = np.dtype(dtype_fields)
            raw = np.frombuffer(f.read(vertex_count * struct_dtype.itemsize), dtype=struct_dtype)
            
            # Extract XYZ
            xyz = np.stack([raw["x"], raw["y"], raw["z"]], axis=-1).astype(np.float32)
            
            # Extract RGB if present
            if "red" in raw.dtype.names and "green" in raw.dtype.names and "blue" in raw.dtype.names:
                rgb = np.stack([raw["red"], raw["green"], raw["blue"]], axis=-1).astype(np.float32)
            else:
                rgb = np.ones_like(xyz) * 128.0

            return np.concatenate([xyz, rgb], axis=-1)
        else:
            raise NotImplementedError(f"PLY format {format_type} is not supported.")

    # ASCII extraction
    prop_names = [p[1] for p in properties]
    x_i = prop_names.index("x")
    y_i = prop_names.index("y")
    z_i = prop_names.index("z")
    xyz = data[:, [x_i, y_i, z_i]]

    if "red" in prop_names and "green" in prop_names and "blue" in prop_names:
        r_i = prop_names.index("red")
        g_i = prop_names.index("green")
        b_i = prop_names.index("blue")
        rgb = data[:, [r_i, g_i, b_i]]
    else:
        rgb = np.ones_like(xyz) * 128.0

    return np.concatenate([xyz, rgb], axis=-1)


def import_point_cloud(input_path: str, output_scene_name: str, scans_dir: str = "data/scannet/scans"):
    """
    Imports a .ply or .npy point cloud and saves it in the ScanRefer format [N, 6].
    """
    os.makedirs(scans_dir, exist_ok=True)
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".ply":
        pts = parse_ply(input_path)
    elif ext == ".npy":
        pts = np.load(input_path)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError("NPY point cloud must be of shape (N, 3) or (N, 6)")
        if pts.shape[1] == 3:
            pts = np.concatenate([pts, np.ones_like(pts) * 128.0], axis=-1)
    else:
        raise ValueError(f"Unsupported point cloud format: {ext}. Supported formats are .ply and .npy.")

    out_filename = f"{output_scene_name}.npy" if not output_scene_name.endswith(".npy") else output_scene_name
    out_path = os.path.join(scans_dir, out_filename)
    np.save(out_path, pts.astype(np.float32))
    print(f"Successfully saved {len(pts)} points to {out_path}!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import real-world 3D point cloud into ScanRefer project")
    parser.add_argument("--input", type=str, required=True, help="Path to input .ply or .npy point cloud")
    parser.add_argument("--scene_name", type=str, required=True, help="Name for the scene (e.g. scene0002_00)")
    parser.add_argument("--scans_dir", type=str, default="data/scannet/scans")
    args = parser.parse_args()
    import_point_cloud(args.input, args.scene_name, args.scans_dir)

