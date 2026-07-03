"""Shared subprocess/Slurm helpers for the fabric/ Python tools."""
import subprocess
import sys
import time


def run(cmd, timeout=60, **kw):
    """subprocess.run with text pipes. Non-zero exit is returned, not raised;
    a missing binary or timeout raises RuntimeError."""
    try:
        return subprocess.run(cmd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE,
                              universal_newlines=True, timeout=timeout, **kw)
    except OSError:
        raise RuntimeError("command not found: %s" % cmd[0])
    except subprocess.TimeoutExpired:
        raise RuntimeError("command timed out: %s" % " ".join(cmd))


def err_exit(tool, msg):
    sys.stderr.write("%s: error: %s\n" % (tool, msg))
    sys.exit(3)


def expand_nodes(nodelist=None, partition=None):
    """Expand a Slurm nodelist or a partition to sorted unique hostnames."""
    if nodelist:
        r = run(["scontrol", "show", "hostnames", nodelist])
        if r.returncode != 0:
            raise RuntimeError("scontrol show hostnames failed: %s"
                               % r.stderr.strip())
        nodes = r.stdout.split()
    elif partition:
        r = run(["sinfo", "-p", partition, "-N", "-h", "-o", "%N"])
        if r.returncode != 0:
            raise RuntimeError("sinfo failed: %s" % r.stderr.strip())
        nodes = r.stdout.split()
    else:
        raise RuntimeError("need a nodelist or a partition")
    nodes = sorted(set(nodes))
    if not nodes:
        raise RuntimeError("no nodes matched")
    return nodes


def submit_sbatch(script_text, args):
    """Submit a script (text, via stdin) with `sbatch --parsable <args>`.
    Returns the job id string. Raises RuntimeError on rejection."""
    r = run(["sbatch", "--parsable"] + list(args), input=script_text)
    if r.returncode != 0:
        raise RuntimeError("sbatch failed: %s" % r.stderr.strip())
    return r.stdout.strip().split(";")[0]


def wait_for_jobs(job_ids, timeout_s, poll_s=15):
    """Poll squeue until the jobs leave the queue or timeout_s elapses.
    Cancels and returns the set of job ids still queued at timeout."""
    deadline = time.time() + timeout_s
    remaining = set(job_ids)
    while remaining and time.time() < deadline:
        r = run(["squeue", "-h", "-o", "%i", "-j",
                 ",".join(sorted(remaining))], timeout=60)
        if r.returncode == 0:
            remaining = remaining & set(r.stdout.split())
        if remaining:
            time.sleep(poll_s)
    for job_id in sorted(remaining):
        run(["scancel", job_id])
    return remaining
