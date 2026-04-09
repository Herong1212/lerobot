#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
hdf5_tree.py — 以树形结构查看 .hdf5 文件

用法:
    python hdf5_tree.py file.h5 [--attrs] [--max-attr-len 120] [--depth N]
                                [--show-links] [--no-filters] [--follow-soft]
                                [--encoding utf-8]

依赖:
    pip install h5py

使用示例：
    # 基本结构树
    python hdf5_tree.py jackal_2019-08-02-16-23-30_0_r00.hdf5

    # 同时展示属性（attributes），并显示链接目标
    python hdf5_tree.py jackal_2019-08-02-16-23-30_0_r00.hdf5 --attrs --show-links

    # 限制递归深度与属性显示长度
    python hdf5_tree.py jackal_2019-08-02-16-23-30_0_r00.hdf5 --attrs --depth 3 --max-attr-len 80

"""

import argparse
import h5py
import sys
from typing import Optional

BRANCH = "├── "
LAST = "└── "
PIPE = "│   "
BLANK = "    "


def fmt_filters(dset: h5py.Dataset) -> str:
    """格式化数据集的过滤器/存储信息"""
    try:
        comp = dset.compression
        comp_opts = dset.compression_opts
        shuffle = dset.shuffle
        fletcher32 = dset.fletcher32
        chunks = dset.chunks
        scaleoffset = dset.scaleoffset
    except Exception:
        # 某些驱动/虚拟布局可能不支持全部属性
        return ""
    parts = []
    if chunks:
        parts.append(f"chunks={chunks}")
    if comp:
        parts.append(f"compression={comp}({comp_opts})")
    if shuffle:
        parts.append("shuffle=True")
    if fletcher32:
        parts.append("fletcher32=True")
    if scaleoffset is not None:
        parts.append(f"scaleoffset={scaleoffset}")
    return (" [" + ", ".join(parts) + "]") if parts else ""


def fmt_shape_dtype(obj) -> str:
    """格式: (shape) dtype"""
    try:
        shape = getattr(obj, "shape", None)
        dtype = getattr(obj, "dtype", None)
        if shape is not None and dtype is not None:
            return f" (shape={shape}, dtype={dtype})"
    except Exception:
        pass
    return ""


def safe_str(x, encoding="utf-8", maxlen=120) -> str:
    """尽量把属性值转字符串，截断过长输出"""
    try:
        if isinstance(x, (bytes, bytearray)):
            s = x.decode(encoding, errors="replace")
        else:
            s = str(x)
    except Exception:
        s = repr(x)
    if len(s) > maxlen:
        s = s[:maxlen] + "…"
    return s


def print_attrs(obj, prefix, show_attrs, max_attr_len, encoding):
    if not show_attrs:
        return
    if hasattr(obj, "attrs"):
        for k in obj.attrs.keys():
            try:
                v = obj.attrs[k]
            except Exception as e:
                v = f"<error reading attr: {e}>"
            print(f"{prefix}    @{k} = {safe_str(v, encoding, max_attr_len)}")


def is_link(obj) -> Optional[str]:
    """若为软/外部链接，返回类型字符串，否则 None"""
    if isinstance(obj, h5py.SoftLink):
        return "softlink"
    if isinstance(obj, h5py.ExternalLink):
        return "externallink"
    if isinstance(obj, h5py.HardLink):
        return "hardlink"
    return None


def walk(name, obj, depth_left, prefix, is_last, args, visited_ids):
    branch = LAST if is_last else BRANCH
    next_prefix = prefix + (BLANK if is_last else PIPE)

    # 链接情况（以链接存储时，obj 是 Link 类型）
    link_type = is_link(obj)
    if link_type:
        target = str(obj.path) if hasattr(obj, "path") else ""
        extra = f" <{link_type}:{target}>" if args.show_links else " <link>"
        print(prefix + branch + name + extra)
        return

    # 常规对象
    line = prefix + branch + name
    if isinstance(obj, h5py.Dataset):
        line += fmt_shape_dtype(obj)
        if not args.no_filters:
            line += fmt_filters(obj)
        print(line)
        print_attrs(obj, next_prefix, args.attrs, args.max_attr_len, args.encoding)
    elif isinstance(obj, h5py.Group):
        print(line + " (group)")
        print_attrs(obj, next_prefix, args.attrs, args.max_attr_len, args.encoding)

        if depth_left == 0:
            return

        # 避免循环（极少数文件可能通过 hardlink 循环引用）
        oid = obj.id.__hash__() if hasattr(obj, "id") else None
        if oid is not None:
            if oid in visited_ids:
                print(next_prefix + LAST + "<cycle detected, skipped>")
                return
            visited_ids.add(oid)

        # 列出成员：数据集在前，组在后，链接最后，按名称排序
        keys = list(obj.keys())
        keys.sort()
        items = []
        for k in keys:
            linfo = obj.get(k, getlink=True)
            if isinstance(linfo, (h5py.SoftLink, h5py.ExternalLink, h5py.HardLink)):
                items.append(("link", k, linfo))
            else:
                try:
                    child = obj[k]
                    if isinstance(child, h5py.Dataset):
                        items.insert(0, ("dset", k, child))
                    elif isinstance(child, h5py.Group):
                        items.append(("group", k, child))
                    else:
                        items.append(("other", k, child))
                except Exception as e:
                    items.append(("error", k, f"<error opening: {e}>"))

        n = len(items)
        for i, (_, k, child) in enumerate(items):
            child_is_last = i == n - 1
            if args.follow_soft and isinstance(child, h5py.SoftLink):
                # 软链接转为目标对象（若可解析）
                try:
                    target_obj = obj.get(k, getlink=False)
                    walk(
                        k,
                        target_obj,
                        depth_left - 1,
                        next_prefix,
                        child_is_last,
                        args,
                        visited_ids,
                    )
                    continue
                except Exception:
                    pass
            walk(
                k, child, depth_left - 1, next_prefix, child_is_last, args, visited_ids
            )
    else:
        print(line + f" ({type(obj).__name__})")


def main():
    p = argparse.ArgumentParser(description="以树形结构查看 HDF5 文件")
    p.add_argument("path", help=".h5 或 .hdf5 文件路径")
    p.add_argument("--attrs", action="store_true", help="显示各节点属性 (attributes)")
    p.add_argument("--max-attr-len", type=int, default=120, help="属性值的最大显示长度")
    p.add_argument(
        "--depth", type=int, default=1_000_000, help="最大递归深度（默认不限）"
    )
    p.add_argument(
        "--show-links", action="store_true", help="显示链接目标（软/外部链接）"
    )
    p.add_argument("--follow-soft", action="store_true", help="尝试跟随软链接遍历")
    p.add_argument(
        "--no-filters", action="store_true", help="不显示数据集压缩/分块等过滤器信息"
    )
    p.add_argument(
        "--encoding", default="utf-8", help="解码字节型属性的编码（默认 utf-8）"
    )
    args = p.parse_args()

    try:
        with h5py.File(args.path, "r") as f:
            root_name = f.filename.split("/")[-1]
            print(f"{root_name} (root)")
            print_attrs(f["/"], "", args.attrs, args.max_attr_len, args.encoding)
            # 遍历根下一级开始
            keys = list(f.keys())
            keys.sort()
            n = len(keys)
            visited_ids = set()
            for i, k in enumerate(keys):
                child = f.get(k, getlink=True)
                walk(k, child, args.depth, "", i == n - 1, args, visited_ids)
    except FileNotFoundError:
        print(f"文件不存在: {args.path}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"无法打开 HDF5 文件: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
