#!/usr/bin/env python3
"""Run a trained LeRobot policy for the simulation, which lives in another conda env.

The simulation (env `gardener`) starts this script (env `gardener-train`) and talks to it over stdin and
stdout: each message is a 4-byte length and a pickled dict. It sends {"reset": True} at the start of an
episode and {"obs": {...}} once per frame, and gets back {"action": 12 numbers}. Started by
gardener/policy/client.py; you do not run it by hand.

  --n-action-steps N   run N actions of each predicted chunk before asking again (default: the checkpoint's)
  --ensemble C         temporal ensembling with this coefficient, which asks every frame (ACT paper: 0.01)
"""
import argparse
import os
import pickle
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gardener.data.instructions import code  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="a pretrained_model folder written by lerobot-train")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--n-action-steps", type=int, default=None)
    ap.add_argument("--ensemble", type=float, default=None)
    args = ap.parse_args()
    out = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    sys.stdout = sys.stderr                     # anything printed while loading must not corrupt the pipe
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.utils.control_utils import predict_action

    config = PreTrainedConfig.from_pretrained(args.checkpoint)
    if args.ensemble is not None:
        config.temporal_ensemble_coeff, config.n_action_steps = args.ensemble, 1
    elif args.n_action_steps is not None:
        config.n_action_steps = args.n_action_steps
    config.device = args.device
    policy = get_policy_class(config.type).from_pretrained(args.checkpoint, config=config)
    policy.to(args.device).eval()
    pre, post = make_pre_post_processors(policy.config, pretrained_path=args.checkpoint,
                                         preprocessor_overrides={"device_processor": {"device": args.device}})
    wants_code = "observation.environment_state" in policy.config.input_features
    print(f"policy {config.type} on {args.device}, chunk {getattr(config, 'chunk_size', '?')}, runs {config.n_action_steps} "
          f"actions per chunk, ensemble {getattr(config, 'temporal_ensemble_coeff', None)}, instruction code {wants_code}", flush=True)

    inp = sys.stdin.buffer
    while True:
        head = inp.read(4)
        if len(head) < 4:
            return
        msg = pickle.loads(inp.read(struct.unpack("<I", head)[0]))
        if msg.get("reset"):
            policy.reset()
            reply = {"ok": True}
        else:
            o = msg["obs"]
            obs = {f"observation.images.{c}": o[c] for c in ("eye", "wrist", "scene") if f"observation.images.{c}" in policy.config.input_features}
            obs["observation.state"] = np.asarray(o["state"], dtype=np.float32)
            if wants_code:
                obs["observation.environment_state"] = np.asarray(code(o["instruction"]), dtype=np.float32)
            action = predict_action(obs, policy, torch.device(args.device), pre, post, use_amp=False,
                                    task=o["instruction"], robot_type="icub_upper_body")
            reply = {"action": action.squeeze(0).float().cpu().numpy()}
        data = pickle.dumps(reply)
        out.write(struct.pack("<I", len(data)) + data)
        out.flush()


if __name__ == "__main__":
    main()
