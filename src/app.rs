//! The optimizer UI: pick two slices on the projection view, run a test
//! ASTRA reconstruction of those two slices, tune the parameters, repeat —
//! then save the parameters into the checkpoint HDF5.

use crate::recon::{AstraMethod, AstraParams, CONFIG_NAME, FILTERS, ReconJob, save_params};
use ct_reconstruction::combine::{LoadJob, LoadedStack};
use egui::{Color32, RichText};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

/// SHA-256 of the advanced-parameters password (same gate as the marimo
/// notebook and the main application's admin section).
const ADVANCED_PASSWORD_SHA256: &str =
    "b8b22aedc372aa891df895be9a7626e6d9ddc6d39ba85d202ca68de8c52ad782";

/// Imaging team logo (same asset and placement as the other rust
/// applications) and the ASTRA toolbox logo, both embedded in the binary
/// and shown at the right end of the toolbar.
const IMAGING_LOGO_BYTES: &[u8] = include_bytes!("../logos/ImagingLogo.png");
const ASTRA_LOGO_BYTES: &[u8] = include_bytes!("../logos/astra_logo.png");
const LOGO_MAX_HEIGHT: f32 = 36.0;

fn load_logo(ctx: &egui::Context, name: &str, bytes: &[u8]) -> Option<egui::TextureHandle> {
    let img = image::load_from_memory(bytes).ok()?;
    let rgba = img.to_rgba8();
    let size = [rgba.width() as usize, rgba.height() as usize];
    let pixels = rgba.into_raw();
    let color_image = egui::ColorImage::from_rgba_unmultiplied(size, &pixels);
    Some(ctx.load_texture(name, color_image, egui::TextureOptions::LINEAR))
}

fn password_matches(input: &str) -> bool {
    let digest = Sha256::digest(input.as_bytes());
    let hex: String = digest.iter().map(|b| format!("{b:02x}")).collect();
    hex == ADVANCED_PASSWORD_SHA256
}

/// One entry of the run history.
struct HistoryEntry {
    params: AstraParams,
    top_slice: usize,
    bottom_slice: usize,
    seconds: f64,
    /// Downsampled copies of the two reconstructed slices, kept so past runs
    /// can be previewed side by side when choosing the parameters.
    thumb_size: (usize, usize),
    top_thumb: Vec<f32>,
    bottom_thumb: Vec<f32>,
    tex: Option<(egui::TextureHandle, egui::TextureHandle)>,
}

pub struct OptimizerApp {
    called_from_app: bool,

    stack: Option<Arc<LoadedStack>>,
    load_job: Option<LoadJob>,
    load_error: Option<String>,

    // Slice selection on the projection view.
    top_slice: usize,
    bottom_slice: usize,
    preview_frame: usize,
    preview_tex: Option<((usize, usize), egui::TextureHandle)>,

    // Parameters.
    params: AstraParams,
    advanced_unlocked: bool,
    advanced_password: String,
    advanced_error: Option<String>,

    // Test reconstruction.
    recon_job: Option<ReconJob>,
    /// Last result: (height, width, top slice, bottom slice, seconds).
    result: Option<(usize, usize, Vec<f32>, Vec<f32>, f64)>,
    result_tex: Option<(egui::TextureHandle, egui::TextureHandle)>,
    recon_error: Option<String>,
    history: Vec<HistoryEntry>,

    // Saving into the HDF5.
    save_status: Option<Result<String, String>>,

    /// Imaging team + tool logos, loaded into textures on the first frame.
    logo_tex: Option<Vec<egui::TextureHandle>>,

    status: String,
}

impl OptimizerApp {
    pub fn new(input: Option<PathBuf>, called_from_app: bool) -> Self {
        let mut app = Self {
            called_from_app,
            stack: None,
            load_job: None,
            load_error: None,
            top_slice: 0,
            bottom_slice: 0,
            preview_frame: 0,
            preview_tex: None,
            params: AstraParams::default(),
            advanced_unlocked: false,
            advanced_password: String::new(),
            advanced_error: None,
            recon_job: None,
            result: None,
            result_tex: None,
            recon_error: None,
            history: Vec::new(),
            save_status: None,
            logo_tex: None,
            status: "Open a pre-processed checkpoint HDF5 to begin.".to_owned(),
        };
        if let Some(path) = input {
            app.start_load(path);
        }
        app
    }

    fn start_load(&mut self, path: PathBuf) {
        self.status = format!("Loading {}…", path.display());
        self.load_error = None;
        self.load_job = Some(LoadJob::start(path));
    }

    fn adopt_stack(&mut self, stack: LoadedStack) {
        let h = stack.sample.first().map(|p| p.height).unwrap_or(1);
        self.top_slice = h / 3;
        self.bottom_slice = (2 * h) / 3;
        self.preview_frame = 0;
        self.preview_tex = None;
        self.result = None;
        self.result_tex = None;
        self.history.clear();
        self.save_status = None;
        self.params = AstraParams::from_stack(&stack);
        let restored = stack
            .metadata
            .iter()
            .any(|(name, _)| name == CONFIG_NAME);
        self.status = format!(
            "{} — {} projections{}",
            stack.path.display(),
            stack.sample.len(),
            if restored {
                " — saved ASTRA parameters restored"
            } else {
                ""
            }
        );
        self.stack = Some(Arc::new(stack));
    }

    /// Stride-downsample a w×h image so its longest side is at most `max`.
    fn downsample(values: &[f32], w: usize, h: usize, max: usize) -> (Vec<f32>, usize, usize) {
        let stride = (w.max(h) / max).max(1);
        let (sw, sh) = (w.div_ceil(stride), h.div_ceil(stride));
        let mut small = Vec::with_capacity(sw * sh);
        for y in (0..h).step_by(stride) {
            for x in (0..w).step_by(stride) {
                small.push(values[y * w + x]);
            }
        }
        (small, sw, sh)
    }

    fn grayscale_texture(
        ctx: &egui::Context,
        name: &str,
        values: &[f32],
        w: usize,
        h: usize,
    ) -> egui::TextureHandle {
        let (mut lo, mut hi) = (f32::MAX, f32::MIN);
        for v in values {
            lo = lo.min(*v);
            hi = hi.max(*v);
        }
        let span = (hi - lo).max(1e-6);
        let pixels: Vec<Color32> = values
            .iter()
            .map(|v| Color32::from_gray((((v - lo) / span) * 255.0) as u8))
            .collect();
        ctx.load_texture(
            name.to_owned(),
            egui::ColorImage {
                size: [w, h],
                source_size: egui::vec2(w as f32, h as f32),
                pixels,
            },
            egui::TextureOptions::LINEAR,
        )
    }

    fn projection_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let Some(stack) = self.stack.clone() else {
            return;
        };
        let Some(first) = stack.sample.first() else {
            return;
        };
        let (w, h, n) = (first.width, first.height, stack.sample.len());
        ui.label(RichText::new("Slice selection").strong());
        self.preview_frame = self.preview_frame.min(n - 1);
        ui.horizontal(|ui| {
            ui.add(egui::Slider::new(&mut self.preview_frame, 0..=n - 1).text("projection"));
            let p = &stack.sample[self.preview_frame];
            ui.label(
                RichText::new(match p.angle_deg {
                    Some(a) => format!("{a:.2}°"),
                    None => String::new(),
                })
                .weak()
                .size(11.0),
            );
        });
        ui.add(egui::Slider::new(&mut self.top_slice, 0..=h - 1).text("top slice (red)"));
        ui.add(
            egui::Slider::new(&mut self.bottom_slice, 0..=h - 1).text("bottom slice (cyan)"),
        );

        let key = (Arc::as_ptr(&stack) as usize, self.preview_frame);
        if self.preview_tex.as_ref().map(|(k, _)| *k) != Some(key) {
            let p = &stack.sample[self.preview_frame];
            let (small, sw, sh) = Self::downsample(&p.mean, p.width, p.height, 512);
            let tex = Self::grayscale_texture(ctx, "projection", &small, sw, sh);
            self.preview_tex = Some((key, tex));
        }
        if let Some((_, tex)) = &self.preview_tex {
            let size = tex.size_vec2();
            let scale = (420.0 / size.x.max(size.y)).min(2.0);
            let response =
                ui.add(egui::Image::from_texture(tex).fit_to_exact_size(size * scale));
            let rect = response.rect;
            let painter = ui.painter_at(rect);
            let y_of =
                |row: usize| rect.top() + (row as f32 / h as f32) * rect.height();
            for (row, color) in [
                (self.top_slice, Color32::from_rgb(255, 90, 80)),
                (self.bottom_slice, Color32::from_rgb(110, 230, 230)),
            ] {
                painter.line_segment(
                    [
                        egui::pos2(rect.left(), y_of(row)),
                        egui::pos2(rect.right(), y_of(row)),
                    ],
                    egui::Stroke::new(1.5, color),
                );
            }
            let _ = w;
        }
        ui.label(
            RichText::new(
                "the test reconstruction runs on the two marked slices only (astra \
                 reconstructs each slice independently)",
            )
            .weak()
            .size(11.0),
        );
    }

    fn params_panel(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("ASTRA parameters").strong());
        egui::ComboBox::from_label("method")
            .selected_text(self.params.method.label())
            .show_ui(ui, |ui| {
                for method in AstraMethod::ALL {
                    ui.selectable_value(&mut self.params.method, method, method.label());
                }
            });
        if self.params.method.is_iterative() {
            ui.add(egui::Slider::new(&mut self.params.num_iter, 1..=500).text("iterations"))
                .on_hover_text(
                    "more iterations improve convergence but increase the reconstruction time",
                );
        } else {
            egui::ComboBox::from_label("filter")
                .selected_text(FILTERS[self.params.filter.min(FILTERS.len() - 1)])
                .show_ui(ui, |ui| {
                    for (i, name) in FILTERS.iter().enumerate() {
                        ui.selectable_value(&mut self.params.filter, i, *name);
                    }
                });
        }

        egui::CollapsingHeader::new(RichText::new("🔒 Advanced").strong())
            .default_open(false)
            .show(ui, |ui| {
                if !self.advanced_unlocked {
                    ui.horizontal(|ui| {
                        ui.label("Password:");
                        let edit = ui.add(
                            egui::TextEdit::singleline(&mut self.advanced_password)
                                .password(true)
                                .desired_width(140.0),
                        );
                        let entered =
                            edit.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
                        if ui.button("Unlock").clicked() || entered {
                            if password_matches(&self.advanced_password) {
                                self.advanced_unlocked = true;
                                self.advanced_error = None;
                            } else {
                                self.advanced_error = Some("wrong password".to_owned());
                            }
                            self.advanced_password.clear();
                        }
                    });
                    if let Some(e) = &self.advanced_error {
                        ui.colored_label(ui.visuals().error_fg_color, e);
                    }
                    return;
                }
                ui.checkbox(&mut self.params.gpu, "run on the GPU (CUDA)")
                    .on_hover_text(
                        "uses the _CUDA variant of the algorithm — much faster, but needs \
                         a GPU on the node",
                    );
                ui.add(
                    egui::Slider::new(&mut self.params.ratio, 0.0..=2.0)
                        .step_by(0.05)
                        .text("circle mask ratio"),
                )
                .on_hover_text(
                    "radius of the circle mask applied to the reconstructed image, \
                     relative to the image size; 1.0 inscribes the image",
                );
                ui.horizontal(|ui| {
                    ui.label("FFT padding (pixels):");
                    ui.add(
                        egui::DragValue::new(&mut self.params.pad).range(-1..=1000),
                    )
                    .on_hover_text(
                        "padding that reduces the FFT edge artifacts; -1 lets algotom \
                         choose automatically",
                    );
                    if self.params.pad < 0 {
                        ui.label(RichText::new("auto").weak());
                    }
                });
                let width = self
                    .stack
                    .as_ref()
                    .and_then(|s| s.sample.first())
                    .map(|p| p.width as f64)
                    .unwrap_or(512.0);
                ui.horizontal(|ui| {
                    ui.label("center of rotation (pixels):");
                    ui.add(
                        egui::DragValue::new(&mut self.params.center)
                            .speed(0.01)
                            .range(0.0..=width),
                    )
                    .on_hover_text(
                        "column of the rotation axis in the sinogram, seeded from the \
                         checkpoint's center of rotation",
                    );
                });
            });
    }

    fn results_panel(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        if let Some(job) = &mut self.recon_job {
            match job.poll() {
                Some(Ok((rh, rw, top, bottom, seconds))) => {
                    let (top_thumb, tw, th) = Self::downsample(&top, rw, rh, 512);
                    let (bottom_thumb, ..) = Self::downsample(&bottom, rw, rh, 512);
                    self.history.push(HistoryEntry {
                        params: self.params,
                        top_slice: self.top_slice,
                        bottom_slice: self.bottom_slice,
                        seconds,
                        thumb_size: (tw, th),
                        top_thumb,
                        bottom_thumb,
                        tex: None,
                    });
                    self.result_tex = Some((
                        Self::grayscale_texture(ctx, "recon_top", &top, rw, rh),
                        Self::grayscale_texture(ctx, "recon_bottom", &bottom, rw, rh),
                    ));
                    self.result = Some((rh, rw, top, bottom, seconds));
                    self.recon_error = None;
                    self.recon_job = None;
                    self.status = format!("Reconstruction done in {seconds:.1} s.");
                }
                Some(Err(e)) => {
                    self.recon_error = Some(e);
                    self.recon_job = None;
                    self.status = "Reconstruction failed.".to_owned();
                }
                None => {
                    ui.horizontal(|ui| {
                        ui.spinner();
                        ui.label("astra is reconstructing the two test slices…");
                    });
                    ctx.request_repaint_after(Duration::from_millis(300));
                }
            }
        }

        let busy = self.recon_job.is_some();
        ui.horizontal(|ui| {
            let evaluate = egui::Button::new(
                RichText::new("▶ Evaluate the reconstruction of the selected slices")
                    .size(16.0)
                    .strong()
                    .color(Color32::WHITE),
            )
            .fill(Color32::from_rgb(230, 126, 0))
            .min_size(egui::vec2(0.0, 36.0));
            if ui
                .add_enabled(self.stack.is_some() && !busy, evaluate)
                .clicked()
            {
                let stack = self.stack.clone().expect("checked above");
                self.recon_error = None;
                self.status = "Running astra…".to_owned();
                self.recon_job = Some(ReconJob::start(
                    stack,
                    self.top_slice,
                    self.bottom_slice,
                    self.params,
                ));
            }
        });
        if let Some(e) = &self.recon_error {
            ui.colored_label(ui.visuals().error_fg_color, e);
        }

        if let (Some((rh, rw, .., seconds)), Some((top_tex, bottom_tex))) =
            (&self.result, &self.result_tex)
        {
            ui.label(
                RichText::new(format!(
                    "reconstructed {rh}x{rw} slices in {seconds:.1} s — {}",
                    self.params.describe()
                ))
                .strong(),
            );
            ui.columns(2, |cols| {
                for (col, tex, label, row) in [
                    (0usize, top_tex, "top slice", self.top_slice),
                    (1, bottom_tex, "bottom slice", self.bottom_slice),
                ] {
                    let ui = &mut cols[col];
                    ui.label(
                        RichText::new(format!("{label} (row {row})"))
                            .strong()
                            .size(13.0),
                    );
                    let size = tex.size_vec2();
                    let width = (ui.available_width() - 12.0).clamp(200.0, 460.0);
                    ui.add(
                        egui::Image::from_texture(tex)
                            .fit_to_exact_size(egui::vec2(width, width * size.y / size.x)),
                    );
                }
            });
        }

        if !self.history.is_empty() {
            ui.add_space(6.0);
            egui::CollapsingHeader::new(RichText::new("Run history").strong())
                .default_open(false)
                .show(ui, |ui| {
                    let mut restore = None;
                    for (i, entry) in self.history.iter_mut().enumerate().rev() {
                        let (tw, th) = entry.thumb_size;
                        let (top_tex, bottom_tex) = entry.tex.get_or_insert_with(|| {
                            (
                                Self::grayscale_texture(
                                    ctx,
                                    &format!("hist_top_{i}"),
                                    &entry.top_thumb,
                                    tw,
                                    th,
                                ),
                                Self::grayscale_texture(
                                    ctx,
                                    &format!("hist_bottom_{i}"),
                                    &entry.bottom_thumb,
                                    tw,
                                    th,
                                ),
                            )
                        });
                        ui.horizontal(|ui| {
                            if ui.button("use").clicked() {
                                restore = Some(entry.params);
                            }
                            for (tex, which) in
                                [(&*top_tex, "top"), (&*bottom_tex, "bottom")]
                            {
                                let size = tex.size_vec2();
                                let thumb_h = 96.0;
                                ui.add(egui::Image::from_texture(tex).fit_to_exact_size(
                                    egui::vec2(thumb_h * size.x / size.y, thumb_h),
                                ))
                                .on_hover_ui(|ui| {
                                    ui.label(
                                        RichText::new(format!("#{} — {which} slice", i + 1))
                                            .strong(),
                                    );
                                    let big = 420.0;
                                    ui.add(egui::Image::from_texture(tex).fit_to_exact_size(
                                        egui::vec2(big, big * size.y / size.x),
                                    ));
                                });
                            }
                            ui.label(
                                RichText::new(format!(
                                    "#{} — rows {}/{} — {} — {:.1} s",
                                    i + 1,
                                    entry.top_slice,
                                    entry.bottom_slice,
                                    entry.params.describe(),
                                    entry.seconds
                                ))
                                .size(12.0),
                            );
                        });
                    }
                    if let Some(params) = restore {
                        self.params = params;
                    }
                });
        }
    }

    /// Save the parameters into the checkpoint, record the outcome in
    /// `save_status`, and report success.
    fn save_params_and_report(&mut self) -> bool {
        let path = self.stack.as_ref().expect("stack checked").path.clone();
        let result = save_params(&path, &self.params)
            .map(|()| format!("{CONFIG_NAME} saved into {}", path.display()));
        if result.is_ok() && self.called_from_app {
            println!("{}", self.params.to_json());
        }
        let ok = result.is_ok();
        self.save_status = Some(result);
        ok
    }
}

impl eframe::App for OptimizerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        if let Some(job) = &mut self.load_job {
            match job.poll() {
                Some(Ok(stack)) => {
                    self.adopt_stack(stack);
                    self.load_job = None;
                }
                Some(Err(e)) => {
                    self.load_error = Some(e);
                    self.load_job = None;
                    self.status = "Loading failed.".to_owned();
                }
                None => ctx.request_repaint_after(Duration::from_millis(300)),
            }
        }

        egui::Panel::top("toolbar").show(ui, |ui| {
            ui.horizontal(|ui| {
                if ui.button("📂 Open a checkpoint HDF5…").clicked() {
                    let mut dialog = rfd::FileDialog::new()
                        .set_title("Select a pre-processed checkpoint HDF5")
                        .add_filter("HDF5", &["h5", "hdf5"]);
                    if let Some(dir) = self
                        .stack
                        .as_ref()
                        .and_then(|s| s.path.parent())
                        .filter(|p| p.is_dir())
                    {
                        dialog = dialog.set_directory(dir);
                    }
                    if let Some(path) = dialog.pick_file() {
                        self.start_load(path);
                    }
                }
                ui.label(RichText::new(&self.status).weak());
                let logos = self.logo_tex.get_or_insert_with(|| {
                    [
                        ("imaging_logo", IMAGING_LOGO_BYTES),
                        ("astra_logo", ASTRA_LOGO_BYTES),
                    ]
                    .into_iter()
                    .filter_map(|(name, bytes)| load_logo(&ctx, name, bytes))
                    .collect()
                });
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    for tex in logos.iter() {
                        ui.add(egui::Image::from_texture(tex).max_height(LOGO_MAX_HEIGHT));
                    }
                    ui.separator();
                    crate::theme::toggle_button(ui);
                });
            });
        });
        if self.stack.is_some() {
            egui::Panel::bottom("actions").show(ui, |ui| {
                let ready = self.recon_job.is_none();
                ui.add_space(6.0);
                ui.horizontal(|ui| {
                    let save = egui::Button::new(
                        RichText::new("💾 Save the parameters into the HDF5")
                            .size(16.0)
                            .strong()
                            .color(Color32::WHITE),
                    )
                    .fill(Color32::from_rgb(46, 125, 50))
                    .min_size(egui::vec2(0.0, 36.0));
                    if ui
                        .add_enabled(ready, save)
                        .on_hover_text(
                            "writes astra_fbp_config into the checkpoint so later \
                             reconstructions use these parameters",
                        )
                        .clicked()
                    {
                        self.save_params_and_report();
                    }
                    let ret = egui::Button::new(
                        RichText::new("↩ Return to the main application")
                            .size(16.0)
                            .strong()
                            .color(Color32::WHITE),
                    )
                    .fill(Color32::from_rgb(21, 101, 192))
                    .min_size(egui::vec2(0.0, 36.0));
                    if ui
                        .add_enabled(ready, ret)
                        .on_hover_text("saves the parameters into the HDF5 and closes this tool")
                        .clicked()
                        && self.save_params_and_report()
                    {
                        ui.ctx().send_viewport_cmd(egui::ViewportCommand::Close);
                    }
                    match &self.save_status {
                        Some(Ok(msg)) => {
                            // The pale green is only legible on the dark background;
                            // the light theme needs a deeper one.
                            let ok_color = if ui.visuals().dark_mode {
                                Color32::from_rgb(120, 200, 120)
                            } else {
                                Color32::from_rgb(27, 118, 51)
                            };
                            ui.colored_label(ok_color, msg);
                        }
                        Some(Err(e)) => {
                            ui.colored_label(ui.visuals().error_fg_color, e);
                        }
                        None => {}
                    }
                });
                ui.add_space(6.0);
            });
        }
        egui::CentralPanel::default().show(ui, |ui| {
            if self.load_job.is_some() {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("loading the stack…");
                });
                return;
            }
            if let Some(e) = &self.load_error {
                ui.colored_label(ui.visuals().error_fg_color, e);
            }
            if self.stack.is_none() {
                return;
            }
            egui::ScrollArea::vertical()
                .auto_shrink([false, false])
                .show(ui, |ui| {
                    ui.columns(2, |cols| {
                        self.projection_panel(&mut cols[0], &ctx);
                        self.params_panel(&mut cols[1]);
                    });
                    ui.separator();
                    self.results_panel(ui, &ctx);
                });
        });
    }
}
