//! ASTRA parameters, the test reconstruction of the two selected slices
//! (through algotom's `astra_reconstruction` wrapper in the
//! `all_ct_reconstruction_development` pixi environment), and saving the
//! parameters back into the checkpoint HDF5.

use nectar::combine::{LoadedStack, Projection};
use nectar::crop::{read_npy, write_npy};
use nectar::rebin::{rebin_center, rebin_projection, rebinned_size};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};

/// The interpreter of the pixi environment that has astra + algotom installed.
pub const ASTRA_PYTHON: &str =
    "/SNS/VENUS/shared/software/git/all_ct_reconstruction_development/.pixi/envs/default/bin/python";

/// The name of the config saved into `/metadata`, matching the main
/// application's `<algorithm key>_config` convention.
pub const CONFIG_NAME: &str = "astra_fbp_config";

/// Sinogram rows reconstructed per selected slice: astra reconstructs each
/// row independently, so a single row per line is shipped (a rebinned test
/// run cuts `BAND × n` rows and block-averages them down to `BAND`).
pub const BAND: usize = 1;

/// The n×n rebin factors the test reconstruction can run on (1 = the
/// checkpoint's own resolution).
pub const TEST_REBIN_FACTORS: [usize; 5] = [1, 2, 3, 4, 6];

/// Largest n×n test-rebin factor that is useful for a stack of `height`
/// rows: the band must still hold [`BAND`] rebinned slices.
pub fn max_test_rebin(height: usize) -> usize {
    TEST_REBIN_FACTORS
        .iter()
        .copied()
        .filter(|n| BAND * n <= height.max(1))
        .max()
        .unwrap_or(1)
}

/// The astra algorithms exposed by algotom's wrapper; each exists as a CPU
/// and a `_CUDA` variant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AstraMethod {
    Fbp,
    Sirt,
    Sart,
    Cgls,
    Bp,
}

impl AstraMethod {
    pub const ALL: [AstraMethod; 5] = [
        AstraMethod::Fbp,
        AstraMethod::Sirt,
        AstraMethod::Sart,
        AstraMethod::Cgls,
        AstraMethod::Bp,
    ];

    pub fn label(self) -> &'static str {
        match self {
            AstraMethod::Fbp => "FBP — filtered back projection",
            AstraMethod::Sirt => "SIRT — simultaneous iterative",
            AstraMethod::Sart => "SART — simultaneous algebraic",
            AstraMethod::Cgls => "CGLS — conjugate gradient",
            AstraMethod::Bp => "BP — plain back projection",
        }
    }

    fn base_name(self) -> &'static str {
        match self {
            AstraMethod::Fbp => "FBP",
            AstraMethod::Sirt => "SIRT",
            AstraMethod::Sart => "SART",
            AstraMethod::Cgls => "CGLS",
            AstraMethod::Bp => "BP",
        }
    }

    /// The `method` string passed to algotom, e.g. `SIRT_CUDA`.
    pub fn astra_name(self, gpu: bool) -> String {
        if gpu {
            format!("{}_CUDA", self.base_name())
        } else {
            self.base_name().to_owned()
        }
    }

    fn from_astra_name(name: &str) -> Option<(Self, bool)> {
        let (base, gpu) = match name.strip_suffix("_CUDA") {
            Some(base) => (base, true),
            None => (name, false),
        };
        let method = Self::ALL
            .into_iter()
            .find(|m| m.base_name() == base)?;
        Some((method, gpu))
    }

    /// FBP and BP ignore `num_iter`; the others ignore the filter.
    pub fn is_iterative(self) -> bool {
        !matches!(self, AstraMethod::Fbp | AstraMethod::Bp)
    }
}

/// The FBP filters listed by algotom.
pub const FILTERS: [&str; 6] = ["ram-lak", "hamming", "hann", "lanczos", "kaiser", "parzen"];

/// The ASTRA parameters used by the notebook's `test_reconstruction`
/// (algotom `astra_reconstruction`), with its defaults.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AstraParams {
    /// Reconstruction algorithm (notebook default: SIRT).
    pub method: AstraMethod,
    /// Use the `_CUDA` variant (the notebook's `SIRT_CUDA`).
    pub gpu: bool,
    /// Iterations for the iterative methods (notebook: 300).
    pub num_iter: i64,
    /// FBP filter (notebook: hann); index into `FILTERS`.
    pub filter: usize,
    /// Radius ratio of the circle mask applied to the reconstruction (1.0).
    pub ratio: f64,
    /// FFT edge padding in pixels; negative = let algotom choose.
    pub pad: i64,
    /// Center of rotation in pixels (column of the sinogram).
    pub center: f64,
}

impl Default for AstraParams {
    fn default() -> Self {
        Self {
            method: AstraMethod::Sirt,
            gpu: true,
            num_iter: 300,
            filter: 2, // hann
            ratio: 1.0,
            pad: -1,
            center: 0.0,
        }
    }
}

impl AstraParams {
    /// Defaults seeded from the stack: the saved `astra_fbp_config` when the
    /// checkpoint carries one, otherwise the stack's center of rotation
    /// (falling back to the middle of the detector).
    pub fn from_stack(stack: &LoadedStack) -> Self {
        if let Some((_, json)) = stack
            .metadata
            .iter()
            .find(|(name, _)| name == CONFIG_NAME)
            && let Some(params) = Self::from_json(json)
        {
            return params;
        }
        let mut params = Self::default();
        if let Some(first) = stack.sample.first() {
            params.center = stack
                .center_of_rotation
                .unwrap_or(first.width as f64 / 2.0);
        }
        params
    }

    /// The saved form matches the notebook's `astra_reconstruction` call:
    /// the method carries the `_CUDA` suffix and `pad` is null for "auto".
    pub fn to_json(&self) -> String {
        serde_json::json!({
            "method": self.method.astra_name(self.gpu),
            "num_iter": self.num_iter,
            "filter_name": FILTERS[self.filter.min(FILTERS.len() - 1)],
            "ratio": self.ratio,
            "pad": if self.pad < 0 {
                serde_json::Value::Null
            } else {
                serde_json::Value::from(self.pad)
            },
            "center": self.center,
        })
        .to_string()
    }

    pub fn from_json(text: &str) -> Option<Self> {
        let doc: serde_json::Value = serde_json::from_str(text).ok()?;
        let mut params = Self::default();
        if let Some((method, gpu)) = doc
            .get("method")
            .and_then(|v| v.as_str())
            .and_then(AstraMethod::from_astra_name)
        {
            params.method = method;
            params.gpu = gpu;
        }
        if let Some(v) = doc.get("num_iter").and_then(|v| v.as_i64()) {
            params.num_iter = v;
        }
        if let Some(name) = doc.get("filter_name").and_then(|v| v.as_str())
            && let Some(i) = FILTERS.iter().position(|f| *f == name)
        {
            params.filter = i;
        }
        if let Some(v) = doc.get("ratio").and_then(|v| v.as_f64()) {
            params.ratio = v;
        }
        params.pad = match doc.get("pad") {
            Some(v) if v.is_null() => -1,
            Some(v) => v.as_i64().unwrap_or(-1),
            None => -1,
        };
        if let Some(v) = doc.get("center").and_then(|v| v.as_f64()) {
            params.center = v;
        }
        Some(params)
    }

    pub fn describe(&self) -> String {
        let mut text = self.method.astra_name(self.gpu);
        if self.method.is_iterative() {
            text.push_str(&format!(", {} iter", self.num_iter));
        } else {
            text.push_str(&format!(
                ", {} filter",
                FILTERS[self.filter.min(FILTERS.len() - 1)]
            ));
        }
        text.push_str(&format!(", ratio {:.2}", self.ratio));
        if self.pad >= 0 {
            text.push_str(&format!(", pad {}", self.pad));
        }
        text.push_str(&format!(", center {:.2}", self.center));
        text
    }
}

const ASTRA_SCRIPT: &str = r#"
import json
import sys

import numpy as np
import algotom.rec.reconstruction as rec

sino_file, spec_file, out_file = sys.argv[1:4]
with open(spec_file) as f:
    spec = json.load(f)
sino = np.load(sino_file)  # (n_angles, n_selected_slices, width)
angles = np.array(spec["angles_rad"], dtype=np.float32)
p = spec["params"]
slices = []
for row in spec["rows"]:
    img = rec.astra_reconstruction(
        np.ascontiguousarray(sino[:, row, :]),
        p["center"],
        angles=angles,
        ratio=p["ratio"],
        method=p["method"],
        num_iter=int(p["num_iter"]),
        filter_name=p["filter_name"],
        pad=p["pad"],
        apply_log=False,
        ncore=1,
    )
    slices.append(np.array(img, dtype=np.float32))
np.save(out_file, np.stack(slices))
"#;

/// One test reconstruction of the two slices on a background thread;
/// resolves to the two reconstructed slices
/// `(height, width, values0, values1, seconds)`.
pub struct ReconJob {
    rx: Receiver<Result<(usize, usize, Vec<f32>, Vec<f32>, f64), String>>,
}

impl ReconJob {
    pub fn start(
        stack: Arc<LoadedStack>,
        top_slice: usize,
        bottom_slice: usize,
        params: AstraParams,
        test_rebin: usize,
    ) -> Self {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let started = std::time::Instant::now();
            let result = run_recon(&stack, top_slice, bottom_slice, params, test_rebin).map(
                |(h, w, top, bottom)| (h, w, top, bottom, started.elapsed().as_secs_f64()),
            );
            let _ = tx.send(result);
        });
        Self { rx }
    }

    pub fn poll(&mut self) -> Option<Result<(usize, usize, Vec<f32>, Vec<f32>, f64), String>> {
        self.rx.try_recv().ok()
    }
}

fn scratch_dir(stack: &LoadedStack) -> Result<PathBuf, String> {
    let base = stack
        .path
        .parent()
        .filter(|p| p.is_dir())
        .map(Path::to_path_buf)
        .unwrap_or_else(std::env::temp_dir);
    let dir = base.join(format!(".astra_optimizer_{}", std::process::id()));
    if std::fs::create_dir_all(&dir).is_ok() {
        return Ok(dir);
    }
    let dir = std::env::temp_dir().join(format!("astra_optimizer_{}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    Ok(dir)
}

/// The rows of one projection feeding a test band around `line`: `BAND`
/// rows at the checkpoint's resolution, `BAND × rebin` rows when the band
/// is rebinned first (so it still holds `BAND` slices afterwards).
fn band_rows(line: usize, height: usize, rebin: usize) -> (usize, usize) {
    let rows = BAND * rebin.max(1);
    let start = line
        .saturating_sub(rows / 2)
        .min(height.saturating_sub(rows));
    (start, start + rows)
}

/// The two test bands of one projection, stacked (top band first), at the
/// test resolution: cut out of the full projection and, for a rebin factor
/// above 1, block-averaged n×n like the pre-processing rebin step.
fn extract_bands(p: &Projection, bands: [(usize, usize); 2], rebin: usize) -> Vec<f32> {
    let w = p.width;
    let mut out = Vec::with_capacity(2 * BAND * w / rebin.max(1));
    for (a, b) in bands {
        let rows = &p.mean[a * w..b * w];
        if rebin <= 1 {
            out.extend_from_slice(rows);
        } else {
            let band = Projection {
                name: String::new(),
                run_number: None,
                angle_deg: None,
                n_images_used: 1,
                height: b - a,
                width: w,
                mean: rows.to_vec(),
                total_counts: 0.0,
            };
            out.extend_from_slice(&rebin_projection(&band, rebin).mean);
        }
    }
    out
}

fn run_recon(
    stack: &LoadedStack,
    top_slice: usize,
    bottom_slice: usize,
    params: AstraParams,
    test_rebin: usize,
) -> Result<(usize, usize, Vec<f32>, Vec<f32>), String> {
    let first = stack
        .sample
        .first()
        .ok_or("no projections in the stack")?;
    let (w, h, n) = (first.width, first.height, stack.sample.len());
    let angles: Vec<f64> = stack
        .sample
        .iter()
        .map(|p| p.angle_deg.map(|a| a.to_radians()))
        .collect::<Option<Vec<f64>>>()
        .ok_or("some projections carry no angle — the reconstruction needs all of them")?;
    // The test runs on n×n rebinned data when asked: smaller sinograms
    // reconstruct faster, at a coarser resolution. The parameters keep the
    // checkpoint's pixel units; only the center of rotation follows the
    // pixel grid onto the rebinned width (like the pre-processing rebin).
    let rebin = test_rebin.clamp(1, max_test_rebin(h));
    let (rw, _) = rebinned_size(w, h, rebin);
    let mut test_params = params;
    test_params.center = rebin_center(params.center, rebin);

    // One BAND-row band around each selected line; the middle row is
    // reconstructed.
    let bands = [
        band_rows(top_slice.min(h - 1), h, rebin),
        band_rows(bottom_slice.min(h - 1), h, rebin),
    ];

    let dir = scratch_dir(stack)?;
    let sino_npy = dir.join("sino.npy");
    let spec_file = dir.join("spec.json");
    let out_npy = dir.join("recon.npy");
    let script = dir.join("astra_run.py");
    let cleanup = || {
        for f in [&sino_npy, &spec_file, &out_npy, &script] {
            let _ = std::fs::remove_file(f);
        }
        let _ = std::fs::remove_dir(&dir);
    };
    let run = || -> Result<(usize, usize, Vec<f32>, Vec<f32>), String> {
        // Astra reconstructs each sinogram row independently, so only the
        // two selected bands are shipped (at the test resolution).
        let mut volume = Vec::with_capacity(n * 2 * BAND * rw);
        for p in &stack.sample {
            volume.extend_from_slice(&extract_bands(p, bands, rebin));
        }
        write_npy(&sino_npy, &[n, 2 * BAND, rw], volume.chunks(2 * BAND * rw))?;
        let spec = serde_json::json!({
            "angles_rad": angles,
            "rows": [BAND / 2, BAND + BAND / 2],
            "params": serde_json::from_str::<serde_json::Value>(&test_params.to_json()).expect("params json"),
        });
        std::fs::write(&spec_file, spec.to_string())
            .map_err(|e| format!("write {}: {e}", spec_file.display()))?;
        std::fs::write(&script, ASTRA_SCRIPT)
            .map_err(|e| format!("write {}: {e}", script.display()))?;
        let output = std::process::Command::new(ASTRA_PYTHON)
            .arg(&script)
            .arg(&sino_npy)
            .arg(&spec_file)
            .arg(&out_npy)
            .output()
            .map_err(|e| format!("cannot launch {ASTRA_PYTHON}: {e}"))?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let tail: Vec<&str> = stderr.trim().lines().rev().take(4).collect();
            let tail: Vec<&str> = tail.into_iter().rev().collect();
            return Err(format!(
                "astra failed ({}): {}",
                output.status,
                tail.join(" | ")
            ));
        }
        let (shape, values) = read_npy(&out_npy)?;
        let [count, rh, rw] = shape.as_slice() else {
            return Err(format!("astra returned shape {shape:?}, expected 3-D"));
        };
        if *count != 2 {
            return Err(format!("astra returned {count} slices, expected 2"));
        }
        let (top, bottom) = values.split_at(rh * rw);
        Ok((*rh, *rw, top.to_vec(), bottom.to_vec()))
    };
    let result = run();
    cleanup();
    result
}

/// The standalone tilt & center-of-rotation tool's correction records in a
/// stack's metadata: one JSON object per applied correction, oldest first.
pub fn tilt_tool_records(metadata: &[(String, String)]) -> Vec<String> {
    metadata
        .iter()
        .find(|(name, _)| name == "tilt_center_of_rotation")
        .map(|(_, value)| {
            value
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

/// The checkpoint's geometry without loading the projections: its
/// `/center_of_rotation` and the tilt tool's correction records. Used to
/// tell whether the tool changed the file before reloading gigabytes.
pub fn checkpoint_geometry(path: &Path) -> Result<(Option<f64>, Vec<String>), String> {
    use hdf5_metno::types::VarLenUnicode;
    let file = hdf5_metno::File::open(path)
        .map_err(|e| format!("cannot open {}: {e}", path.display()))?;
    let cor = file
        .dataset("center_of_rotation")
        .and_then(|ds| ds.read_scalar::<f64>())
        .ok();
    let records = file
        .group("metadata")
        .and_then(|g| g.dataset("tilt_center_of_rotation"))
        .and_then(|ds| ds.read_scalar::<VarLenUnicode>())
        .map(|v| {
            v.as_str()
                .lines()
                .filter(|l| !l.trim().is_empty())
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default();
    Ok((cor, records))
}

/// Write (or replace) the `astra_fbp_config` JSON in the checkpoint's
/// `/metadata` group, where the main application reads it back.
pub fn save_params(path: &Path, params: &AstraParams) -> Result<(), String> {
    use hdf5_metno::types::VarLenUnicode;
    let file = hdf5_metno::File::open_rw(path)
        .map_err(|e| format!("cannot open {} for writing: {e}", path.display()))?;
    let metadata = match file.group("metadata") {
        Ok(group) => group,
        Err(_) => file
            .create_group("metadata")
            .map_err(|e| format!("create metadata group: {e}"))?,
    };
    if metadata.dataset(CONFIG_NAME).is_ok() {
        metadata
            .unlink(CONFIG_NAME)
            .map_err(|e| format!("replace {CONFIG_NAME}: {e}"))?;
    }
    let value: VarLenUnicode = params.to_json().parse().unwrap_or_default();
    metadata
        .new_dataset::<VarLenUnicode>()
        .create(CONFIG_NAME)
        .and_then(|ds| ds.write_scalar(&value))
        .map_err(|e| format!("write {CONFIG_NAME}: {e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn params_json_roundtrip() {
        let params = AstraParams {
            method: AstraMethod::Fbp,
            gpu: false,
            num_iter: 42,
            filter: 4,
            ratio: 0.8,
            pad: 120,
            center: 251.75,
        };
        let back = AstraParams::from_json(&params.to_json()).unwrap();
        assert_eq!(back, params);
        // The saved form carries the astra method string and a null pad
        // when it is automatic, like the notebook.
        let doc: serde_json::Value = serde_json::from_str(&params.to_json()).unwrap();
        assert_eq!(doc["method"], "FBP");
        assert_eq!(doc["filter_name"], "kaiser");
        let auto = AstraParams { pad: -1, gpu: true, ..params };
        let doc: serde_json::Value = serde_json::from_str(&auto.to_json()).unwrap();
        assert!(doc["pad"].is_null());
        assert_eq!(doc["method"], "FBP_CUDA");
        assert_eq!(AstraParams::from_json(&auto.to_json()).unwrap().pad, -1);
        assert!(AstraParams::from_json("nope").is_none());
    }

    #[test]
    fn rebinned_center_follows_the_pixel_grid() {
        // No rebin: unchanged.
        assert_eq!(rebin_center(979.96, 1), 979.96);
        // The detector center stays the detector center of the rebinned
        // image (4096 wide: 2047.5 -> 1023.5 at 2x2).
        assert!((rebin_center(2047.5, 2) - 1023.5).abs() < 1e-9);
        // A general center: (c + 0.5) / n - 0.5.
        assert!((rebin_center(979.96, 2) - ((979.96 + 0.5) / 2.0 - 0.5)).abs() < 1e-9);
        assert!((rebin_center(979.96, 4) - ((979.96 + 0.5) / 4.0 - 0.5)).abs() < 1e-9);
    }

    #[test]
    fn band_rows_hold_band_slices_after_rebin() {
        // 2x2: 2 rows around row 100 -> rebinned to one slice.
        assert_eq!(band_rows(100, 2048, 2), (99, 101));
        assert_eq!(band_rows(100, 2048, 4), (98, 102));
        // Full resolution: the row itself.
        assert_eq!(band_rows(100, 2048, 1), (100, 101));
        // Clamped at the edges.
        assert_eq!(band_rows(0, 2048, 3), (0, 3));
        assert_eq!(band_rows(2047, 2048, 1), (2047, 2048));
        assert_eq!(band_rows(2047, 2048, 6), (2042, 2048));
        // Tiny stacks limit the useful factor.
        assert_eq!(max_test_rebin(5), 4);
        assert_eq!(max_test_rebin(2), 2);
        assert_eq!(max_test_rebin(1), 1);
        assert_eq!(max_test_rebin(2048), 6);
    }

    #[test]
    fn extract_bands_rebins_each_band() {
        // 4 px wide, 8 rows: row r holds the value r everywhere.
        let mut mean = Vec::new();
        for r in 0..8 {
            mean.extend(std::iter::repeat_n(r as f32, 4));
        }
        let p = Projection {
            name: "p".into(),
            run_number: None,
            angle_deg: Some(0.0),
            n_images_used: 1,
            height: 8,
            width: 4,
            mean,
            total_counts: 0.0,
        };
        // Full resolution: the rows verbatim.
        let full = extract_bands(&p, [(0, 1), (6, 7)], 1);
        assert_eq!(full, vec![0.0, 0.0, 0.0, 0.0, 6.0, 6.0, 6.0, 6.0]);
        // 2x2: rows (0,1) -> 0.5, rows (4,5) -> 4.5; 2 px wide.
        let reb = extract_bands(&p, [(0, 2), (4, 6)], 2);
        assert_eq!(reb, vec![0.5, 0.5, 4.5, 4.5]);
        // 4x4: rows 0..4 -> 1.5; 1 px wide.
        let reb = extract_bands(&p, [(0, 4), (4, 8)], 4);
        assert_eq!(reb, vec![1.5, 5.5]);
    }
}
