"""Tira do cache do macOS as páginas dos .safetensors de um modelo (msync MS_INVALIDATE), sem sudo.
Imprime a fração residente antes e depois (mincore)."""
import ctypes, mmap, os, sys, glob
libc = ctypes.CDLL(None, use_errno=True)
libc.mmap.restype = ctypes.c_void_p
libc.mmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_longlong]
libc.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
PAGE = os.sysconf("SC_PAGE_SIZE"); MS_INVALIDATE = 0x2
def one(path, evict):
    fd = os.open(path, os.O_RDONLY); size = os.fstat(fd).st_size
    addr = libc.mmap(None, size, mmap.PROT_READ, mmap.MAP_SHARED, fd, 0)
    n = (size + PAGE - 1) // PAGE; vec = ctypes.create_string_buffer(n)
    libc.mincore(addr, size, vec); before = sum(b & 1 for b in vec.raw)
    if evict: libc.msync(addr, size, MS_INVALIDATE)
    libc.mincore(addr, size, vec); after = sum(b & 1 for b in vec.raw)
    libc.munmap(addr, size); os.close(fd)
    return before * PAGE, after * PAGE, size
tot = [0, 0, 0]
for f in sorted(glob.glob(os.path.join(os.path.realpath(sys.argv[1]), "*.safetensors"))):
    b, a, s = one(f, "--check" not in sys.argv)
    tot[0] += b; tot[1] += a; tot[2] += s
print(f"residente antes {tot[0]/1e9:.2f} GB, depois {tot[1]/1e9:.2f} GB, de {tot[2]/1e9:.2f} GB")
