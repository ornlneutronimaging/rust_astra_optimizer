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
4. Choose the **Test data** resolution: the checkpoint's own (the
   default), or the projections n×n block-averaged first (2x2, 3x3, 4x4,
   6x6 — the same block mean as the pre-processing rebin step). Smaller
   sinograms reconstruct faster; the slices are coarser. This is a speed
   shortcut for the test only: the parameters are saved in the
   checkpoint's pixel units and the full reconstruction runs on the
   checkpoint as is (only the center of rotation is converted for the
   rebinned test run, following the pixel grid like the pre-processing
   rebin does).
5. **▶ Evaluate** reconstructs the two selected slices through the real
   `astra` (from the `all_ct_reconstruction_development` pixi environment)
   and shows them side by side — astra reconstructs each sinogram row
   independently, so this takes seconds. Every run lands in the
   **Run history** with slice thumbnails (hover to enlarge); its `use`
   buttons restore the parameters (and test resolution) of a previous run.
6. **💾 Save** writes `astra_fbp_config` (JSON, with the astra method
   string such as `SIRT_CUDA`) into the checkpoint's `/metadata`;
   `rust_ct_reconstruction` restores it automatically and later ASTRA
   reconstructions use these parameters.

Defaults follow the notebook: `SIRT_CUDA`, 300 iterations, `hann` filter,
mask ratio 1.0, automatic padding, center seeded from the checkpoint. When
no GPU is visible, astra falls back to the CPU implementation on its own.

## Center of rotation and tilt

The projection view draws the **center of rotation**: the checkpoint's
value as a dashed orange line and, once the center is changed in the
Advanced section, the current one as a solid green line, with a readout
of both values and the move between them (`↺ checkpoint value` goes
back). Drag the center field or use the arrow keys; holding Shift moves
it 10× faster. The center is an absolute column (px) and defaults to the
checkpoint's `/center_of_rotation` (the middle of the detector when the
file carries none).

Under the projection, **Tilt correction** lists what the checkpoint
records: the pre-processing step of `rust_ct_reconstruction`
(`tilt_correction`) and every run of the standalone tool
(`metadata/tilt_center_of_rotation`, JSON records), plus the
pre-processing rebin when there was one.

**🎯 Open the tilt & center-of-rotation tool** launches
`rust_tilt_center_of_rotation` on the checkpoint itself (the more robust
estimators: sub-pixel 0°/180° registration, all-pairs consensus, gridrec
test slices). Applying & saving there rewrites the checkpoint's
projections and center of rotation; when the tool closes, this window
detects the new correction record, reloads the file and re-seeds the
center from the new value — save the parameters to keep it. Closing the
tool without applying changes nothing.

## Running

```bash
./launch_astra_optimizer.sh [checkpoint.h5]
```

Requires a graphical session; the launch script rebuilds when sources
changed. `--called-from-app` additionally prints the saved JSON on stdout
for a driving application.
