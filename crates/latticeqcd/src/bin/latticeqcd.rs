use latticeqcd::{run_lqcd, Params};
use std::{env, error::Error, process};

fn main() {
    let mut args = env::args();
    let program = args.next().unwrap_or_else(|| "latticeqcd".to_owned());
    let Some(path) = args.next() else {
        eprintln!("usage: {program} PATH");
        process::exit(2);
    };
    if args.next().is_some() {
        eprintln!("usage: {program} PATH");
        process::exit(2);
    }

    let result = Params::from_file(&path).and_then(|params| {
        run_lqcd(&params).map_err(|failure| {
            eprintln!("{failure}");
            let mut source = failure.source();
            while let Some(error) = source {
                eprintln!("caused by: {error}");
                source = error.source();
            }
            process::exit(1);
        })
    });
    match result {
        Ok(report) => println!(
            "completed_updates={} accepted={} rejected={} measurements={} flows={} outputs={}",
            report.completed_updates,
            report.accepted_updates,
            report.rejected_updates,
            report.measurements.len(),
            report.flows.len(),
            report.published_paths.len(),
        ),
        Err(error) => {
            eprintln!("{error}");
            let mut source = error.source();
            while let Some(error) = source {
                eprintln!("caused by: {error}");
                source = error.source();
            }
            process::exit(1);
        }
    }
}
