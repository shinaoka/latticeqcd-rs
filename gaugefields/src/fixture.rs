use crate::{GaugeError, GaugeLinkTensor, GaugeLinks, LatticeShape4};
use npyz::Order;
use num_complex::Complex64;
use serde::Deserialize;
use serde_json::Value;
use std::{fs, path::Path};
use tenferro_tensor::Tensor;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FixtureMetadata {
    pub nc: usize,
    pub lattice: [usize; 4],
    pub beta: f64,
    pub expected_observables: Value,
    pub gaugefields_jl_version: String,
    pub gaugefields_jl_commit: String,
    #[serde(default)]
    pub reference_bits: Option<Vec<Vec<[u64; 2]>>>,
}

#[derive(Debug)]
pub struct Fixture {
    links: GaugeLinks,
    metadata: FixtureMetadata,
}
impl Fixture {
    pub fn links(&self) -> &GaugeLinks {
        &self.links
    }
    pub fn metadata(&self) -> &FixtureMetadata {
        &self.metadata
    }
}

pub fn load_fixture(directory: impl AsRef<Path>) -> Result<Fixture, GaugeError> {
    let directory = directory.as_ref();
    let meta_path = directory.join("metadata.json");
    let metadata: FixtureMetadata =
        serde_json::from_slice(&fs::read(&meta_path).map_err(|source| GaugeError::Io {
            path: meta_path.clone(),
            source,
        })?)
        .map_err(GaugeError::Metadata)?;
    if metadata.nc == 0 {
        return Err(GaugeError::MetadataMismatch {
            detail: "NC must be positive".into(),
        });
    }
    let lattice = LatticeShape4::new(metadata.lattice)?;
    let expected: Vec<u64> = std::iter::once(metadata.nc as u64)
        .chain(std::iter::once(metadata.nc as u64))
        .chain(metadata.lattice.map(|n| n as u64))
        .collect();
    let mut links = Vec::with_capacity(4);
    for mu in 0..4 {
        let path = directory.join(format!("u{mu}.npy"));
        let bytes = fs::read(&path).map_err(|source| GaugeError::Io { path, source })?;
        let npy = npyz::NpyFile::new(&bytes[..]).map_err(|e| GaugeError::Npy {
            mu,
            detail: e.to_string(),
        })?;
        if npy.order() != Order::Fortran {
            return Err(GaugeError::NpyOrder { mu });
        }
        if npy.shape().len() != 6 {
            return Err(GaugeError::NpyRank {
                mu,
                found: npy.shape().len(),
            });
        }
        if npy.shape() != expected {
            return Err(GaugeError::MetadataMismatch {
                detail: format!(
                    "mu {mu} shape {:?}, metadata expects {expected:?}",
                    npy.shape()
                ),
            });
        }
        let values = npy
            .into_vec::<Complex64>()
            .map_err(|e| GaugeError::NpyDType {
                mu,
                detail: e.to_string(),
            })?;
        let shape = expected.iter().map(|&n| n as usize).collect::<Vec<_>>();
        let tensor = Tensor::from_vec_col_major(shape, values)
            .map_err(|e| GaugeError::Tensor(e.to_string()))?;
        links.push(GaugeLinkTensor::new(tensor, lattice)?);
    }
    let links: [GaugeLinkTensor; 4] =
        links.try_into().map_err(|_| GaugeError::MetadataMismatch {
            detail: "expected four directions".into(),
        })?;
    Ok(Fixture {
        links: GaugeLinks::new(links)?,
        metadata,
    })
}
