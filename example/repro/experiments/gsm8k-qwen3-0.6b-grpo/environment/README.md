# Historical training / evaluation environment

The measured runtime was Python 3.11.13, Torch 2.6.0+cu124, TRL 0.24.0, Transformers 4.57.1 and PEFT 0.17.1, with one L40S. `runtime.json` and `pip-freeze.txt` are copied from the existing runtime evidence and checked against its Git object.

The code snapshots contain their own historical project requirements. No environment was installed or GPU run executed during this documentation review. OS binaries, device drivers and external download availability are not recreated by this capsule. Do not substitute the new dsh vLLM 0.25 environment and call it the same historical configuration.
