#!/usr/bin/env python3
"""Run a Python script with its network connections forced to IPv4.

Why: on this laptop's connection, IPv6 packets larger than about 1,300 bytes to the Hugging Face servers are
dropped without any error (checked with `ping -6 -M do -s 1300 huggingface.co`: 100% loss; IPv4 1472: 0%).
Uploads then connect, send a few kB and hang forever. Over IPv4 they work.

Xet (the Rust uploader) does not go through Python's sockets, so set HF_HUB_DISABLE_XET=1 as well:
  HF_HUB_DISABLE_XET=1 env -u PYTHONPATH ~/miniconda3/envs/gardener-train/bin/python scripts/ipv4_only.py \\
      scripts/push_dataset.py --root data/lerobot/gardener_v2 --repo-id Kimaa6/icub_gardener_v2 \\
      --description docs/datasets/gardener_v2_card.md --large --yes
"""
import runpy
import socket
import sys

_getaddrinfo = socket.getaddrinfo


def _ipv4_only(host, port, family=0, *args, **kwargs):
    return _getaddrinfo(host, port, socket.AF_INET, *args, **kwargs)


socket.getaddrinfo = _ipv4_only
sys.argv = sys.argv[1:]
runpy.run_path(sys.argv[0], run_name="__main__")
