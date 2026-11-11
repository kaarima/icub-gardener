"""The instruction as five numbers, for a policy that cannot read text (ACT).

[left pot, right pot, a little, normal, a lot]: one-hot for the pot and one-hot for the amount. The converter
saves it as observation.environment_state and the policy evaluation sends the same numbers. No heavy imports,
so both conda envs can use it. SmolVLA reads the text itself and does not need this.
"""
import re

CODE_NAMES = ["target_left", "target_right", "amount_little", "amount_normal", "amount_lot"]


def parse(text: str) -> tuple[str, str]:
    """(pot, amount) from an instruction. Amount is 'little', 'lot' or 'normal'."""
    t = text.lower()
    pot = "left" if re.search(r"\bleft\b", t) else "right" if re.search(r"\bright\b", t) else None
    if pot is None:
        raise ValueError(f"no plant named in: {text!r}")
    return pot, "little" if "little" in t else "lot" if "a lot" in t else "normal"


def code(text: str) -> list[float]:
    pot, amount = parse(text)
    return [float(pot == "left"), float(pot == "right"),
            float(amount == "little"), float(amount == "normal"), float(amount == "lot")]
