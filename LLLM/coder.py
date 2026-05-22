"""
A LLM with a role!
Coder is meant to be a generation loop for specialized coder LLLM.
The target is Qwen/Qwen2.5-Coder-0.5B-Instruct.
It shall implement here self-consistency applied to code production.
Coder can only produce self-contained, interpretable or compilable file.
For each produced code:
    - if relevant compile it.
    - executable will be run on a container.
    - optionaly static analysis will be performed.
    - produce a dict with the collected information.
    - the collected output is given back to Coder so it can score it
    (against the original instructions).
Using the produced dict for each code, we choose here the best one.
"""
