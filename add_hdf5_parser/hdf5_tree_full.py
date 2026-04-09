import h5py
import sys
import os

# 终端运行：python hdf5_tree_full.py jackal_2019-08-02-16-23-30_0_r00.hdf5


def print_hdf5_structure(name, obj, indent=0):
    """递归打印 HDF5 文件结构"""
    prefix = "    " * indent + "├── " if indent > 0 else ""
    if isinstance(obj, h5py.Group):
        print(f"{prefix}{name}/ (Group)")
        for key, item in obj.items():
            print_hdf5_structure(key, item, indent + 1)
    elif isinstance(obj, h5py.Dataset):
        shape = obj.shape
        dtype = obj.dtype
        print(f"{prefix}{name} (Dataset) shape={shape}, dtype={dtype}")
    else:
        print(f"{prefix}{name} (Unknown type)")


def show_hdf5_tree(file_path):
    """打开 HDF5 文件并显示其完整结构"""
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return
    try:
        with h5py.File(file_path, "r") as f:
            print(f"{os.path.basename(file_path)} (root)")
            for key, item in f.items():
                print_hdf5_structure(key, item, 1)
    except Exception as e:
        print(f"❌ 打开文件失败: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python hdf5_tree_full.py your_file.hdf5")
        sys.exit(1)
    file_path = sys.argv[1]
    show_hdf5_tree(file_path)
