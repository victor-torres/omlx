import ctypes,os,time,numpy as np, pathlib
lib=ctypes.CDLL(None,use_errno=True)
page=os.sysconf('SC_PAGE_SIZE')
class Top(ctypes.Structure):
 _fields_=[('obj_id',ctypes.c_uint32),('ref_count',ctypes.c_uint32),('private',ctypes.c_uint32),('shared',ctypes.c_uint32),('mode',ctypes.c_uint8)]
task=ctypes.c_uint32.in_dll(lib,'mach_task_self_').value
maps=[np.memmap(x,mode='r',dtype='uint8') for x in pathlib.Path(os.environ['MODEL_DIR']).glob('*.safetensors')]
for i in range(3):
 t=time.perf_counter();total=0
 for m in maps:
  addr=ctypes.c_uint64(m.ctypes.data);size=ctypes.c_uint64();info=Top();count=ctypes.c_uint32(ctypes.sizeof(Top)//4);obj=ctypes.c_uint32()
  ret=lib.mach_vm_region(task,ctypes.byref(addr),ctypes.byref(size),12,ctypes.byref(info),ctypes.byref(count),ctypes.byref(obj))
  assert not ret,ret
  assert addr.value==m.ctypes.data and size.value>=m.nbytes,(addr.value,size.value,m.ctypes.data,m.nbytes)
  if obj.value:lib.mach_port_deallocate(task,obj)
  total+=page*(info.private+info.shared)
 print('TOP',time.perf_counter()-t,total,flush=True)
t=time.perf_counter();total=0
for m in maps:
 n=(len(m)+page-1)//page;vec=np.empty(n,dtype='uint8')
 ret=lib.mincore(ctypes.c_void_p(m.ctypes.data),ctypes.c_size_t(n*page),ctypes.c_void_p(vec.ctypes.data));assert not ret
 total+=np.count_nonzero(vec&1)*page
print('MINCORE',time.perf_counter()-t,total,flush=True)
