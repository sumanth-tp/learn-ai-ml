"""Child process that runs model-written chart code. Never import this in the parent.

Usage: python -I runner.py <data.json> <code.py> <out.png> <workdir> <mem_mb> <cpu_s>

Order matters: set resource limits, import and warm up the trusted libraries, THEN
install the audit hook, THEN run the untrusted code. Anything the libraries need to do
(read fonts, build caches) happens before the doors close.
"""

import contextlib
import os
import sys


def _limit(mem_mb: int, cpu_s: int) -> None:
    import resource

    for name, value in (
        ("RLIMIT_AS", mem_mb * 1024 * 1024),
        ("RLIMIT_CPU", cpu_s),
        ("RLIMIT_FSIZE", 20 * 1024 * 1024),
        ("RLIMIT_NPROC", 0),
    ):
        res = getattr(resource, name, None)
        if res is None:
            continue
        # macOS refuses some limits (RLIMIT_AS, RLIMIT_NPROC); the audit hook and the
        # container's own limits still apply.
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(res, (value, value))


BLOCKED_PREFIXES = (
    "socket.",
    "subprocess.",
    "os.system",
    "os.exec",
    "os.spawn",
    "os.posix_spawn",
    "os.fork",
    "os.forkpty",
    "os.kill",
    "os.remove",
    "os.unlink",
    "os.rmdir",
    "os.rename",
    "os.replace",
    "os.chmod",
    "os.chown",
    "os.symlink",
    "os.link",
    "os.truncate",
    "os.putenv",
    "os.unsetenv",
    "shutil.",
    "ctypes.",
    "urllib.",
    "http.",
    "ftplib.",
    "smtplib.",
    "webbrowser.",
    "sqlite3.",
    "pty.",
    "fcntl.",
)
WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


def main() -> int:
    data_path, code_path, out_path, workdir, mem_mb, cpu_s = sys.argv[1:7]
    _limit(int(mem_mb), int(cpu_s))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    df = pd.read_json(data_path, orient="split")
    with open(code_path, encoding="utf-8") as fh:
        code = compile(fh.read(), "<chart>", "exec")
    warm = plt.figure()
    warm.canvas.draw()
    plt.close(warm)

    root = os.path.realpath(workdir) + os.sep

    def hook(event: str, args: tuple) -> None:
        if event.startswith(BLOCKED_PREFIXES):
            raise PermissionError(f"sandbox blocked {event}")
        if event == "open" and args:
            path, mode, flags = [*args, None, None][:3]
            if isinstance(path, int) or path is None:
                return
            writing = (isinstance(mode, str) and any(c in mode for c in "wax+")) or (
                isinstance(flags, int) and flags & WRITE_FLAGS
            )
            if writing and not os.path.realpath(os.fsdecode(path)).startswith(root):
                raise PermissionError(f"sandbox blocked write outside workdir: {path}")

    sys.addaudithook(hook)
    namespace = {"df": df, "pd": pd, "np": np, "plt": plt, "__builtins__": __builtins__}
    exec(code, namespace)  # noqa: S102 - this IS the sandbox
    fig = plt.gcf()
    if not fig.axes:
        print("chart code drew nothing", file=sys.stderr)
        return 3
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    sys.exit(main())
