"""
The few things that differ between the GPUs this runs on.

CUDA (NVIDIA, and AMD through ROCm, which PyTorch also calls "cuda") and MPS
(Apple Silicon, through Metal) compute the same model through the same code.
Where they differ is around the edges: which device to pick by default, how to
wait for queued work before reading a clock, and what memory can be asked
about.

Memory is the one real difference. CUDA counts the peak it has allocated. MPS
has no such counter. What it reports instead is what Metal's allocator holds
from the system, which keeps freed buffers in heaps for reuse: a cheap number
that says how much of the machine a run is taking, which is what peak() gives
for MPS, but a coarse one - it moves in whole heaps, and a step that fits in
heaps already held adds nothing to it. To measure a peak the way CUDA counts
one, use PeakTracker, which samples live tensor memory after every operation.

On Apple Silicon the GPU's memory is the machine's memory: there is no card to
fill separately from RAM. The budget PyTorch reports for it,
recommended_max_memory(), is the part of RAM Metal is willing to give one
process - about three quarters of it.
"""

import torch


def default():
    """The device a run picks when not told: CUDA, then MPS, then the CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def kind(device):
    """'cuda', 'mps' or 'cpu' for a torch.device or a string naming one."""
    if device is None:
        return default()
    return device.type if hasattr(device, "type") else str(device).split(":")[0]


def is_gpu(device):
    return kind(device) in ("cuda", "mps")


def sync(device):
    """Wait until everything queued on `device` has run, before reading a clock."""
    k = kind(device)
    if k == "cuda":
        torch.cuda.synchronize(device)
    elif k == "mps":
        torch.mps.synchronize()


def empty_cache(device):
    k = kind(device)
    if k == "cuda":
        torch.cuda.empty_cache()
    elif k == "mps":
        torch.mps.empty_cache()


def reset_peak(device):
    """Start a fresh peak. On MPS this empties the cache: what Metal holds is
    the only peak MPS has, and emptying is how it comes back down."""
    k = kind(device)
    if k == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elif k == "mps":
        torch.mps.empty_cache()


def allocated(device):
    """Bytes in live tensors on `device` now; None on the CPU."""
    k = kind(device)
    if k == "cuda":
        return torch.cuda.memory_allocated(device)
    if k == "mps":
        return torch.mps.current_allocated_memory()
    return None


def peak(device):
    """Peak bytes since the last reset_peak; None on the CPU. On MPS, what
    Metal holds - cheap but coarse; see the module docstring."""
    k = kind(device)
    if k == "cuda":
        return torch.cuda.max_memory_allocated(device)
    if k == "mps":
        return torch.mps.driver_allocated_memory()
    return None


class PeakTracker:
    """
    The peak of live tensor memory on `device` over a `with` block - what
    CUDA's max_memory_allocated counts, for MPS, which keeps no such counter.
    It is sampled after every operation, the backward's included, so it
    costs a Python call per op: measure memory in a step that is not timed.
    On CUDA it reads the real counter and costs nothing; on the CPU, None.

        with D.PeakTracker(dev) as pt:
            step()
        pt.peak    # bytes above what was live when the block began
        pt.high    # the peak itself: everything live at the worst moment
    """

    def __init__(self, device):
        self.device = device
        self.kind = kind(device)
        self.peak = self.high = None
        self._mode = None

    def __enter__(self):
        if self.kind == "cuda":
            torch.cuda.synchronize(self.device)
            self._base = torch.cuda.memory_allocated(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        elif self.kind == "mps":
            from torch.utils._python_dispatch import TorchDispatchMode

            tracker = self

            class _Sample(TorchDispatchMode):
                def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                    out = func(*args, **(kwargs or {}))
                    m = torch.mps.current_allocated_memory()
                    if m > tracker._high:
                        tracker._high = m
                    return out

            torch.mps.synchronize()
            self._base = self._high = torch.mps.current_allocated_memory()
            self._mode = _Sample()
            self._mode.__enter__()
        return self

    def __exit__(self, *exc):
        if self.kind == "cuda":
            torch.cuda.synchronize(self.device)
            self.high = torch.cuda.max_memory_allocated(self.device)
            self.peak = self.high - self._base
        elif self.kind == "mps":
            self._mode.__exit__(*exc)
            self.high = self._high
            self.peak = self.high - self._base
        return False


def total(device):
    """Bytes the device can give this process; None on the CPU."""
    k = kind(device)
    if k == "cuda":
        idx = torch.device(device).index
        idx = torch.cuda.current_device() if idx is None else idx
        return torch.cuda.get_device_properties(idx).total_memory
    if k == "mps":
        return torch.mps.recommended_max_memory()
    return None


def name(device):
    """A line naming the device, for the start of a run."""
    k = kind(device)
    if k == "cuda":
        return torch.cuda.get_device_name(device)
    if k == "mps":
        import platform
        import subprocess
        try:
            chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True,
                                  timeout=2).stdout.strip()
        except Exception:                                  # noqa: BLE001
            chip = ""
        return f"{chip or 'Apple Silicon'} GPU (Metal), macOS {platform.mac_ver()[0]}"
    return "CPU"


OutOfMemoryError = getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError)


def is_oom(e):
    """Whether exception `e` is a GPU running out of memory. CUDA raises
    OutOfMemoryError; MPS raises a plain RuntimeError ("MPS backend out of
    memory ..."), so `except OutOfMemoryError` alone would miss it:

        try: ...
        except Exception as e:
            if not D.is_oom(e):
                raise
    """
    return isinstance(e, OutOfMemoryError) or (
        isinstance(e, RuntimeError) and "out of memory" in str(e))
