"""Servidor de bench isolado: shim do cluster no scratch, fusion controlada por arquivo."""
import json, logging, os
from pathlib import Path
here = Path(__file__).resolve().parent
import omlx.cluster.worker_shim as shim
_shim = shim.ensure_cluster_python_shim
shim.ensure_cluster_python_shim = lambda: _shim(home=here / "home")
from omlx.model_settings import ModelSettingsManager
_get = ModelSettingsManager.get_settings
def get_settings(self, model_id):
    s = _get(self, model_id)
    ctl = here / "fusion.json"
    if ctl.exists():
        s.moe_gate_up_fusion_enabled = json.loads(ctl.read_text())["fusion"]
    return s
ModelSettingsManager.get_settings = get_settings
from omlx.patches import moe_gate_up_fusion as fusion
_fuse = fusion.apply_moe_gate_up_fusion
def fuse(model):
    n = _fuse(model)
    logging.getLogger("bench").info("BENCH_FUSION layers=%s", n)
    return n
fusion.apply_moe_gate_up_fusion = fuse
from omlx.cli import main
main()
