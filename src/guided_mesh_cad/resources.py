"""Stdlib process admission and limits; all memory values use MiB (2**20)."""
from dataclasses import asdict, dataclass
import ctypes
import math
import os
import sys
import time


class ResourceError(RuntimeError):
    code = 'resource_unavailable'


@dataclass(frozen=True)
class ResourceLimits:
    memory_limit_mib: float = 4096
    reserve_physical_mib: float = 1536
    reserve_commit_mib: float = 1536
    timeout_seconds: float | None = None
    poll_seconds: float = .1

    @classmethod
    def from_mapping(cls, value=None):
        value = {} if value is None else value
        if not isinstance(value, dict):
            raise ValueError('resources must be an object')
        unknown = set(value) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError('unknown resource keys: ' + ', '.join(sorted(unknown)))
        result = cls(**value)
        for key, val in asdict(result).items():
            if key == 'timeout_seconds' and val is None:
                continue
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(key + ' must be a finite number')
            if val < 0 or (key in ('memory_limit_mib', 'timeout_seconds', 'poll_seconds') and val == 0):
                raise ValueError(key + ' is outside its permitted range')
        if not .02 <= result.poll_seconds <= 5:
            raise ValueError('poll_seconds must be between .02 and 5')
        if not 1 <= result.memory_limit_mib * 1048576 <= sys.maxsize:
            raise ValueError('memory_limit_mib is outside the supported byte range')
        return result

    def to_dict(self):
        return asdict(self)


def memory_status():
    """Available physical and commit bytes, without native CAD imports."""
    if os.name == 'nt':
        class Memory(ctypes.Structure):
            _fields_ = [('length', ctypes.c_ulong), ('load', ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ('total_physical', 'available_physical', 'total_page', 'available_commit',
                 'total_virtual', 'available_virtual', 'extended_virtual')]
        data = Memory()
        data.length = ctypes.sizeof(data)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(data)):
            raise ResourceError('cannot query system memory')
        return {'available_physical': data.available_physical,
                'available_commit': data.available_commit}
    try:
        with open('/proc/meminfo', encoding='ascii') as handle:
            info = {line.split(':')[0]: int(line.split()[1]) * 1024 for line in handle}
        return {'available_physical': info['MemAvailable'],
                'available_commit': max(0, info['CommitLimit'] - info['Committed_AS'])}
    except (OSError, KeyError, ValueError) as exc:
        raise ResourceError('supported system memory accounting is unavailable') from exc


def check_reserves(limits, snapshot=None):
    snapshot = memory_status() if snapshot is None else snapshot
    for field, reserve in [('available_physical', limits.reserve_physical_mib),
                           ('available_commit', limits.reserve_commit_mib)]:
        if snapshot[field] < reserve * 1048576:
            raise ResourceError(field + ' is below the configured reserve')
    return snapshot


class ProcessGuard:
    """Keep a Windows job handle live until termination; no inherited handles."""
    def __init__(self, process, limits):
        self.process, self.limits, self.job = process, limits, None
        if os.name == 'nt':
            self._attach_windows_job()

    def _attach_windows_job(self):
        from ctypes import wintypes
        class Basic(ctypes.Structure):
            _fields_ = [('process_time', ctypes.c_longlong), ('job_time', ctypes.c_longlong),
                        ('flags', wintypes.DWORD), ('min_working', ctypes.c_size_t),
                        ('max_working', ctypes.c_size_t), ('active', wintypes.DWORD),
                        ('affinity', ctypes.c_size_t), ('priority', wintypes.DWORD),
                        ('scheduling', wintypes.DWORD)]
        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_ulonglong) for name in
                        ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]
        class Extended(ctypes.Structure):
            _fields_ = [('basic', Basic), ('io', IO), ('process_memory', ctypes.c_size_t),
                        ('job_memory', ctypes.c_size_t), ('peak_process', ctypes.c_size_t),
                        ('peak_job', ctypes.c_size_t)]
        api = ctypes.WinDLL('kernel32', use_last_error=True)
        api.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        api.CreateJobObjectW.restype = wintypes.HANDLE
        api.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        api.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        api.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.api = api
        self.job = api.CreateJobObjectW(None, None)
        if not self.job:
            raise ResourceError('cannot create worker memory job')
        info = Extended()
        info.basic.flags = 0x100 | 0x200 | 0x2000  # process/job memory; kill on close
        info.process_memory = info.job_memory = int(self.limits.memory_limit_mib * 1048576)
        if not api.SetInformationJobObject(self.job, 9, ctypes.byref(info), ctypes.sizeof(info)) or not api.AssignProcessToJobObject(self.job, int(self.process._handle)):
            self.close()
            raise ResourceError('cannot enforce worker memory job')

    def terminate(self):
        if self.job:
            self.api.TerminateJobObject(self.job, 1)
        elif self.process.poll() is None:
            # The child starts a new session on POSIX; its descendants share it.
            import signal
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self.process.wait()

    def close(self):
        if self.job:
            self.api.CloseHandle(self.job)
            self.job = None


def apply_child_limit(limits):
    if os.name != 'nt':
        import resource
        cap = int(limits.memory_limit_mib * 1048576)
        resource.setrlimit(resource.RLIMIT_AS, (cap, cap))


def wait_process(process, limits, guard):
    start = time.monotonic()
    while process.poll() is None:
        if limits.timeout_seconds is not None and time.monotonic() - start >= limits.timeout_seconds:
            guard.terminate()
            return 'timeout'
        try:
            check_reserves(limits)
        except ResourceError:
            guard.terminate()
            return 'resource_reserve_exhausted'
        time.sleep(limits.poll_seconds)
    return None
