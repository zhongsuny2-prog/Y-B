#!/usr/bin/env python3
"""Apply the torch_npu Inductor compatibility patches.

Required by: torch 2.15.0.dev20261005 + torch_npu 2.15.0+git281a08e.
Run with the target interpreter, e.g.

    python apply_torch_npu_patches.py

It locates site-packages itself; pass a path as argv[1] to override.
Each edit keeps a ``.orig-bak`` copy next to the file it touches.
Idempotent: re-running is a no-op.
"""

import pathlib
import shutil
import sys

if len(sys.argv) > 1:
    SP = pathlib.Path(sys.argv[1])
else:
    import torch_npu  # noqa: F401  (import for its side effect of locating the package)

    SP = pathlib.Path(__import__("torch_npu").__file__).parent.parent

print(f"site-packages: {SP}")


def patch(rel, edits, marker):
    path = SP / rel
    text = path.read_text(encoding="utf-8")
    if marker in text:
        print(f"  already patched: {rel}")
        return
    backup = path.with_name(path.name + ".orig-bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    for old, new in edits:
        n = text.count(old)
        assert n == 1, f"{rel}: anchor found {n} times: {old[:70]!r}"
        text = text.replace(old, new, 1)
    path.write_text(text, encoding="utf-8")
    print(f"  patched: {rel}")


# ---------------------------------------------------------------------------
# 1. lowering.py -- torch 2.15 renamed the kwarg torch passes to make_reduction
#    (strict_sum -> strict_reduction). torch_npu's copy still has the old
#    signature, so Inductor lowering raises TypeError.
# ---------------------------------------------------------------------------
patch(
    "torch_npu/_inductor/lowering.py",
    [
        (
            "def make_reduction(reduction_type: ReductionType, override_return_dtype=None):\n",
            "def make_reduction(\n"
            "    reduction_type: ReductionType,\n"
            "    override_return_dtype=None,\n"
            "    *,\n"
            "    strict_reduction: bool = False,\n"
            "):\n",
        )
    ],
    marker="strict_reduction: bool = False",
)

# ---------------------------------------------------------------------------
# 2. npu_combined_scheduling.py
#    a) can_fuse_reduction_epilogue reads _nv_universal_gemm_scheduling, which
#       NPUCombinedScheduling never initializes -> override it to False.
#    b) torch 2.15 added more inherited methods with the same problem
#       (can_fuse_reduction_pair, get_fusion_pair_priority). Rather than
#       patching each caller, install an inert stand-in object so any present
#       or future predicate answers "no NVGEMM".
# ---------------------------------------------------------------------------
STUB = '''class _InertNvUniversalGemmScheduling:
    """Stand-in for NVUniversalGemmScheduling on NPU.

    NPU has no NV universal-GEMM templates, so every inherited
    ``CUDACombinedScheduling`` branch that consults this object should simply
    not fire. Answering the predicates with False retires them; the remaining
    calls only happen inside those branches.
    """

    def is_nv_universal_gemm_template(self, node) -> bool:
        return False

    def is_nv_universal_gemm_fused_template(self, node) -> bool:
        return False

    def has_nvgemm_bool_output(self, node) -> bool:
        return False

    def has_conflicting_epilogue_reductions(self, node1, node2) -> bool:
        return False

    def get_fusion_pair_priority(self, node1, node2) -> int:
        # CUDACombinedScheduling returns this verbatim when it is < 2, and
        # otherwise falls through to Triton's priority + 2. >= 2 therefore
        # means "no NVGEMM opinion", which is what NPU wants.
        return 2

    def __getattr__(self, name):
        # A predicate added by a newer torch: answer "no NVGEMM" instead of
        # raising. Non-predicate additions need an explicit entry above.
        return lambda *args, **kwargs: False


'''

ANCHOR = "class NPUCombinedScheduling(CUDACombinedScheduling, TritonScheduling):"
EPILOGUE = (
    "    def can_fuse_reduction_epilogue(self, node1, node2) -> bool:\n"
    "        # CUDACombinedScheduling reads self._nv_universal_gemm_scheduling,\n"
    "        # which NPUCombinedScheduling never initializes. NPU has no such fusion.\n"
    "        return False\n"
    "\n"
)
HORIZ = "    def can_fuse_horizontal(self, node1: BaseSchedulerNode, node2: BaseSchedulerNode):\n"
INIT_OLD = "        self._catlass_scheduling = CATLASSScheduling(scheduler)\n"
INIT_NEW = (
    "        self._catlass_scheduling = CATLASSScheduling(scheduler)\n"
    "        # The CUDA __init__ is skipped above, so this attribute would be\n"
    "        # missing; give it an inert object rather than patching each\n"
    "        # inherited caller.\n"
    "        self._nv_universal_gemm_scheduling = _InertNvUniversalGemmScheduling()\n"
)

path = SP / "torch_npu/_inductor/codegen/npu_combined_scheduling.py"
text = path.read_text(encoding="utf-8")
if "_InertNvUniversalGemmScheduling" in text and EPILOGUE in text:
    print("  already patched: npu_combined_scheduling.py")
else:
    backup = path.with_name(path.name + ".orig-bak")
    if not backup.exists():
        shutil.copy2(path, backup)
    if EPILOGUE not in text:
        assert text.count(HORIZ) == 1
        text = text.replace(HORIZ, EPILOGUE + HORIZ, 1)
    if "_InertNvUniversalGemmScheduling" not in text:
        assert text.count(ANCHOR) == 1
        text = text.replace(ANCHOR, STUB + ANCHOR, 1)
        assert text.count(INIT_OLD) == 1
        text = text.replace(INIT_OLD, INIT_NEW, 1)
    path.write_text(text, encoding="utf-8")
    print("  patched: npu_combined_scheduling.py")

# ---------------------------------------------------------------------------
# 3. scheduling.py -- TritonScheduling enables the sub-parent epilogue plan,
#    but on NPU its tiling path reaches statically_known_equals() with a shape
#    list and raises SympifyError. NPU has no such plan.
# ---------------------------------------------------------------------------
patch(
    "torch_npu/_inductor/codegen/scheduling.py",
    [
        (
            "class NPUTritonScheduling(TritonScheduling):\n    kernel_type = NPUIndexTritonKernel\n",
            "class NPUTritonScheduling(TritonScheduling):\n"
            "    kernel_type = NPUIndexTritonKernel\n"
            "    # TritonScheduling enables the sub-parent epilogue plan, but its tiling\n"
            "    # path reaches statically_known_equals() with a shape list and raises\n"
            "    # SympifyError on NPU. NPU has no such plan.\n"
            "    supports_sub_parent_epilogue = False\n",
        )
    ],
    marker="supports_sub_parent_epilogue = False",
)

print("all patches applied")
