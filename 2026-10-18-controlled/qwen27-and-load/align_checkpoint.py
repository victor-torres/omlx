"""Cópia alinhada de um checkpoint safetensors: cada tensor começa em offset absoluto múltiplo de 8.

Uso: align_checkpoint.py ORIGEM DESTINO
Não altera a origem. Copia os arquivos que não são .safetensors como estão (o index.json
mapeia tensor -> arquivo, e os nomes dos arquivos não mudam). Confere cada tensor por SHA-256.
"""
import hashlib, json, os, shutil, struct, sys
from pathlib import Path

CHUNK = 64 << 20


def read_header(f):
    n = struct.unpack("<Q", f.read(8))[0]
    return n, json.loads(f.read(n))


def copy_range(src, dst, lo, hi):
    src.seek(lo)
    left = hi - lo
    while left:
        buf = src.read(min(CHUNK, left))
        dst.write(buf)
        left -= len(buf)


def align_shard(src_path, dst_path):
    with open(src_path, "rb") as src:
        n, header = read_header(src)
        start = 8 + n
        meta = header.pop("__metadata__", None)
        order = sorted(header, key=lambda k: header[k]["data_offsets"][0])
        # Offsets novos: cada tensor num múltiplo de 8 a partir do início dos dados.
        cursor, new = 0, {}
        for k in order:
            lo, hi = header[k]["data_offsets"]
            cursor += (-cursor) % 8
            new[k] = dict(header[k], data_offsets=[cursor, cursor + hi - lo])
            cursor += hi - lo
        out = ({"__metadata__": meta} if meta is not None else {}) | new
        encoded = json.dumps(out, separators=(",", ":")).encode()
        encoded += b" " * ((-(8 + len(encoded))) % 8)  # dados começam em offset múltiplo de 8
        tmp = dst_path.with_suffix(".partial")
        with open(tmp, "wb") as dst:
            dst.write(struct.pack("<Q", len(encoded)) + encoded)
            pos = 0
            for k in order:
                lo, hi = header[k]["data_offsets"]
                nlo = new[k]["data_offsets"][0]
                dst.write(b"\0" * (nlo - pos))
                copy_range(src, dst, start + lo, start + hi)
                pos = nlo + hi - lo
        os.replace(tmp, dst_path)
    return len(order)


def tensor_hashes(path):
    with open(path, "rb") as f:
        n, header = read_header(f)
        header.pop("__metadata__", None)
        out = {}
        for k, v in header.items():
            lo, hi = v["data_offsets"]
            f.seek(8 + n + lo)
            h, left = hashlib.sha256(), hi - lo
            while left:
                buf = f.read(min(CHUNK, left))
                h.update(buf)
                left -= len(buf)
            out[k] = (v["dtype"], v["shape"], h.hexdigest())
        return out


def main(src_dir, dst_dir):
    src_dir, dst_dir = Path(src_dir).resolve(), Path(dst_dir)
    if dst_dir.exists():
        sys.exit(f"destino já existe: {dst_dir}")
    dst_dir.mkdir(parents=True)
    for p in sorted(src_dir.iterdir()):
        if p.is_dir():
            continue
        if p.suffix == ".safetensors":
            count = align_shard(p, dst_dir / p.name)
            same = tensor_hashes(p) == tensor_hashes(dst_dir / p.name)
            print(f"{p.name}: {count} tensores, conferência {'ok' if same else 'FALHOU'}", flush=True)
            if not same:
                sys.exit(1)
        else:
            shutil.copy2(p, dst_dir / p.name)
    print("FIM", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:3])
