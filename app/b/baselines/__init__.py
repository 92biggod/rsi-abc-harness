"""Apply the submission CPU budget before the package imports NumPy."""
import os

threads = int(os.environ.get("BASELINE_CPU_THREADS", "1"))
if not 1 <= threads <= 8:
    raise ValueError("BASELINE_CPU_THREADS must be between 1 and 8")
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = str(threads)
