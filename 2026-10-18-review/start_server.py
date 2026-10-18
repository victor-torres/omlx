"""Keep the test server's peer-interpreter shim inside the scratch directory."""
from pathlib import Path
import omlx.cluster.worker_shim as shim
original = shim.ensure_cluster_python_shim
shim.ensure_cluster_python_shim = lambda: original(home=Path(__file__).resolve().parent/'bench-home')
# Benchmark-only internal fusion control, restricted to a separate model ID.
# It is never part of the PR or any user's persistent settings.
import os
if os.environ.get('OMLX_BENCH_FUSION_CONTROL') == '1':
    from omlx.model_settings import ModelSettingsManager
    get_settings = ModelSettingsManager.get_settings
    def benchmark_settings(self, model_id):
        result = get_settings(self, model_id)
        if model_id == 'Ornith-stock-unfused':
            result.moe_gate_up_fusion_enabled = False
        return result
    ModelSettingsManager.get_settings = benchmark_settings
from omlx.cli import main
main()
