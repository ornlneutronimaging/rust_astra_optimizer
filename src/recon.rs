//! ASTRA parameters, the test reconstruction of the two selected slices
//! (through algotom's `astra_reconstruction` wrapper in the
//! `all_ct_reconstruction_development` pixi environment), and saving the
//! parameters back into the checkpoint HDF5.

use ct_reconstruction::combine::LoadedStack;
use ct_reconstruction::crop::{read_npy, write_npy};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};

/// The interpreter of the pixi environment that has astra + algotom installed.
pub const ASTRA_PYTHON: &str =
    "/SNS/VENUS/shared/software/git/all_ct_reconstruction_development/.pixi/envs/default/bin/python";

/// The name of the config saved into `/metadata`, matching the main
/// application's `<algorithm key>_config` convention.
pub const CONFIG_NAME: &str = "astra_fbp_config";

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
    ) -> Self {
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let started = std::time::Instant::now();
            let result = run_recon(&stack, top_slice, bottom_slice, params).map(
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

fn run_recon(
    stack: &LoadedStack,
    top_slice: usize,
    bottom_slice: usize,
    params: AstraParams,
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
    let top_slice = top_slice.min(h - 1);
    let bottom_slice = bottom_slice.min(h - 1);

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
        // two selected rows are shipped.
        let mut volume = Vec::with_capacity(n * 2 * w);
        for p in &stack.sample {
            for row in [top_slice, bottom_slice] {
                volume.extend_from_slice(&p.mean[row * w..(row + 1) * w]);
            }
        }
        write_npy(&sino_npy, &[n, 2, w], volume.chunks(2 * w))?;
        let spec = serde_json::json!({
            "angles_rad": angles,
            "rows": [0, 1],
            "params": serde_json::from_str::<serde_json::Value>(&params.to_json()).expect("params json"),
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
}
