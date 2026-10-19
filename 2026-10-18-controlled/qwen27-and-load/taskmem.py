"""Memória do próprio processo via task_info(TASK_VM_INFO): residente, comprimida, footprint."""
import ctypes
class _VM(ctypes.Structure):
    _fields_ = [("virtual_size", ctypes.c_uint64), ("region_count", ctypes.c_int32), ("page_size", ctypes.c_int32),
                ("resident_size", ctypes.c_uint64), ("resident_size_peak", ctypes.c_uint64),
                ("device", ctypes.c_uint64), ("device_peak", ctypes.c_uint64),
                ("internal", ctypes.c_uint64), ("internal_peak", ctypes.c_uint64),
                ("external", ctypes.c_uint64), ("external_peak", ctypes.c_uint64),
                ("reusable", ctypes.c_uint64), ("reusable_peak", ctypes.c_uint64),
                ("pv_pmap", ctypes.c_uint64), ("pv_resident", ctypes.c_uint64), ("pv_virtual", ctypes.c_uint64),
                ("compressed", ctypes.c_uint64), ("compressed_peak", ctypes.c_uint64), ("compressed_lifetime", ctypes.c_uint64),
                ("phys_footprint", ctypes.c_uint64), ("_rest", ctypes.c_uint64 * 64)]
_lib = ctypes.CDLL(None)
_task = ctypes.c_uint32.in_dll(_lib, "mach_task_self_")
def task_mem():
    info = _VM(); count = ctypes.c_uint32(ctypes.sizeof(_VM) // 4)
    if _lib.task_info(_task, 22, ctypes.byref(info), ctypes.byref(count)):
        raise OSError("task_info falhou")
    return {k: getattr(info, k) / 1e9 for k in ("resident_size", "internal", "external", "compressed", "compressed_lifetime", "phys_footprint")}
if __name__ == "__main__":
    print(task_mem())
