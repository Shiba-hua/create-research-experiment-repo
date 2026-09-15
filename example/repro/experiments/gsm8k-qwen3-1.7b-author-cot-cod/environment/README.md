# Historical author-prompt inference environment

The measured runtime used the separate PRoot userspace with Torch 2.8.0+cu128, vLLM 0.10.2, Transformers 4.57.1, BF16, V1 InprocClient and a single L40S. Recorded pip freeze, wheel inventory, installation and namespace evidence are included with source hashes. `namespace-run-hostpaths.sh` records the historical binding wrapper; it is not an installation command for the current host.

`historical-project-requirements.txt` is a copy of the repository's training requirements, not this inference runtime's lock. Its Torch 2.6 entry must not replace the actual Torch 2.8 requirement. Original data/model paths and the external author-source directory must be rebound explicitly. This capsule does not include the PRoot rootfs, driver binaries, model weights or private author responses, and no new runtime was tested in this review.
