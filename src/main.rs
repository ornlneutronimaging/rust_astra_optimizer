//! ASTRA optimizer — tune the ASTRA reconstruction parameters on two test
//! slices of a pre-processed CT checkpoint (the HDF5 written by
//! rust_ct_reconstruction), then save them back into the file so the full
//! reconstruction uses them.

use std::path::PathBuf;
use astra_optimizer::app::OptimizerApp;

const USAGE: &str = "\
astra_optimizer — tune ASTRA parameters on two test slices

USAGE:
  astra_optimizer [OPTIONS] [CHECKPOINT.h5]

ARGS:
  CHECKPOINT.h5   A pre-processed checkpoint written by rust_ct_reconstruction
                  (attenuation data with angles and center of rotation). When
                  omitted, the file can be opened from within the application.

OPTIONS:
  --called-from-app   Driven by another application: saving the parameters
                      also prints the astra_config JSON on stdout
  -h, --help          Show this help
";

fn main() -> eframe::Result<()> {
    let mut input = None;
    let mut called_from_app = false;
    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            "--called-from-app" | "--called_from_app" => called_from_app = true,
            s if s.starts_with('-') => {
                eprintln!("Error: unknown option: {s}\n\n{USAGE}");
                std::process::exit(2);
            }
            s => input = Some(PathBuf::from(s)),
        }
    }

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1400.0, 950.0])
            .with_title("ASTRA Optimizer"),
        ..Default::default()
    };
    eframe::run_native(
        "ASTRA Optimizer",
        native_options,
        Box::new(move |cc| {
            cc.egui_ctx.set_theme(astra_optimizer::theme::load());
            Ok(Box::new(OptimizerApp::new(input, called_from_app)))
        }),
    )
}
