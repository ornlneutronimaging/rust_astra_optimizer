# ASTRA Optimizer

Standalone GUI to tune ASTRA reconstruction parameters on a pre-processed
CT checkpoint (the HDF5 written by `rust_ct_reconstruction`: attenuation
data with `/angles_rad` and `/center_of_rotation`). Uses algotom's
`astra_reconstruction` wrapper, like the notebook's `test_reconstruction`.

## Workflow

1. Open a checkpoint (command-line argument or the 📂 button).
2. Pick two slices on the projection view (red and cyan lines).
3. Adjust the parameters — the **method** (FBP, SIRT, SART, CGLS, BP) with
   its **iterations** (iterative methods) or **filter** (FBP) in the open
   section; the GPU toggle, circle mask ratio, FFT padding and the center
   of rotation behind the password-locked **Advanced** section.
4. **▶ Evaluate** reconstructs the two selected slices through the real
   `astra` (from the `all_ct_reconstruction_development` pixi environment)
   and shows them side by side — astra reconstructs each sinogram row
   independently, so this takes seconds. Every run lands in the
   **Run history** with slice thumbnails (hover to enlarge); its `use`
   buttons restore the parameters of a previous run.
5. **💾 Save** writes `astra_fbp_config` (JSON, with the astra method
   string such as `SIRT_CUDA`) into the checkpoint's `/metadata`;
   `rust_ct_reconstruction` restores it automatically and later ASTRA
   reconstructions use these parameters.

Defaults follow the notebook: `SIRT_CUDA`, 300 iterations, `hann` filter,
mask ratio 1.0, automatic padding, center seeded from the checkpoint. When
no GPU is visible, astra falls back to the CPU implementation on its own.

## Running

```bash
./launch_astra_optimizer.sh [checkpoint.h5]
```

Requires a graphical session; the launch script rebuilds when sources
changed. `--called-from-app` additionally prints the saved JSON on stdout
for a driving application.
