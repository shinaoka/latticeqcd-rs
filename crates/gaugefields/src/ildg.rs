//! Minimal ILDG 1.1 LIME I/O for one host-resident SU(3) configuration.
//!
//! The binary order follows Gaugefields.jl v0.7.2 at commit
//! `9e5719970770f4497405a856315c90bef7f74449`,
//! `src/output/ildg_format.jl::{save_binarydata,load_binarydata!}`. The LIME
//! framing and XML validation are implemented independently from that source;
//! checksum records are skipped and not verified.

use crate::{site_index, GaugeError, GaugeLinkTensor, GaugeLinks, HostGaugeLinks, LatticeShape4};
use num_complex::Complex64;
use quick_xml::{events::Event, Reader};
use std::{
    fs::File,
    io::{self, Cursor, Read, Write},
    path::{Path, PathBuf},
};
use tenferro_tensor::TypedTensor;

const LIME_MAGIC: u32 = 0x4567_89ab;
const LIME_VERSION: u16 = 1;
const LIME_MB: u16 = 0x8000;
const LIME_ME: u16 = 0x4000;
const LIME_HEADER_BYTES: usize = 144;
const MAX_XML_BYTES: u64 = 1024 * 1024;
const FORMAT_TYPE: &[u8] = b"ildg-format";
const BINARY_TYPE: &[u8] = b"ildg-binary-data";

/// Reads one minimally framed ILDG 1.1 SU(3), Float64 configuration.
///
/// The reader accepts one LIME message containing one `ildg-format` and one
/// `ildg-binary-data` record. Unknown records, XML comments, whitespace, and
/// XML namespaces are tolerated. Checksum records are streamed past without
/// verification.
///
/// # Errors
///
/// Returns a typed error for I/O, malformed LIME/XML, unsupported metadata,
/// truncation, duplicate or split records, invalid dimensions, length
/// mismatches, non-finite components, or trailing data.
///
/// # Examples
///
/// ```
/// use gaugefields::{cold_su3, read_ildg, write_ildg, LatticeShape4};
///
/// let path = std::env::temp_dir().join(format!("gaugefields-doctest-{}", std::process::id()));
/// let links = cold_su3(LatticeShape4::new([1, 1, 1, 1])?)?;
/// write_ildg(&path, &links)?;
/// let loaded = read_ildg(&path)?;
/// assert_eq!(loaded.host_view()?.link(0, 0)?.trace().re, 3.0);
/// std::fs::remove_file(path)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn read_ildg(path: impl AsRef<Path>) -> Result<GaugeLinks, GaugeError> {
    let path = path.as_ref().to_path_buf();
    let file = File::open(&path).map_err(|source| GaugeError::IldgIo {
        path: path.clone(),
        source,
    })?;
    read_ildg_stream(file, &path)
}

/// Writes one canonical ILDG 1.1 LIME message.
///
/// The message contains one `ildg-format` record followed by one unsplit
/// `ildg-binary-data` record. Values are big-endian IEEE Float64 in
/// `t,z,y,x,mu,row,column,real/imag` order. Input validation, including the
/// host/SU(3) boundary, dimensions, allocation bounds, and finiteness, is
/// completed before the destination is created or truncated. Checksums are not
/// emitted.
///
/// # Errors
///
/// Returns a typed error for invalid input or I/O. An I/O failure after file
/// creation may leave a partial destination file.
///
/// # Examples
///
/// ```
/// use gaugefields::{cold_su3, read_ildg, write_ildg, LatticeShape4};
///
/// let path = std::env::temp_dir().join(format!("gaugefields-doctest-write-{}", std::process::id()));
/// let links = cold_su3(LatticeShape4::new([1, 1, 1, 1])?)?;
/// write_ildg(&path, &links)?;
/// assert_eq!(read_ildg(&path)?.lattice(), links.lattice());
/// std::fs::remove_file(path)?;
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
pub fn write_ildg(path: impl AsRef<Path>, links: &GaugeLinks) -> Result<(), GaugeError> {
    let path = path.as_ref().to_path_buf();
    let view = links.host_view()?;
    let binary_length = checked_binary_length(view.lattice())?;
    validate_finite(&view)?;
    let xml = format_xml(view.lattice());

    let file = File::create(&path).map_err(|source| GaugeError::IldgIo {
        path: path.clone(),
        source,
    })?;
    write_ildg_stream(file, &path, &view, &xml, binary_length)
}

fn io_error(path: &Path, source: io::Error) -> GaugeError {
    GaugeError::IldgIo {
        path: path.to_path_buf(),
        source,
    }
}

struct IldgReader<R> {
    reader: R,
    path: PathBuf,
}

impl<R: Read> IldgReader<R> {
    fn new(reader: R, path: &Path) -> Self {
        Self {
            reader,
            path: path.to_path_buf(),
        }
    }

    fn read_exact(&mut self, bytes: &mut [u8], detail: &'static str) -> Result<(), GaugeError> {
        self.reader.read_exact(bytes).map_err(|source| {
            if source.kind() == io::ErrorKind::UnexpectedEof {
                GaugeError::IldgFormat { detail }
            } else {
                io_error(&self.path, source)
            }
        })
    }

    fn read_binary_exact(
        &mut self,
        bytes: &mut [u8],
        detail: &'static str,
    ) -> Result<(), GaugeError> {
        self.reader.read_exact(bytes).map_err(|source| {
            if source.kind() == io::ErrorKind::UnexpectedEof {
                GaugeError::IldgPayload { detail }
            } else {
                io_error(&self.path, source)
            }
        })
    }

    fn read_header(&mut self) -> Result<Option<RecordHeader>, GaugeError> {
        let mut raw = [0_u8; LIME_HEADER_BYTES];
        let mut first = [0_u8; 1];
        let count = self
            .reader
            .read(&mut first)
            .map_err(|source| io_error(&self.path, source))?;
        if count == 0 {
            return Ok(None);
        }
        raw[0] = first[0];
        self.read_exact(&mut raw[1..], "truncated LIME header")?;
        Ok(Some(parse_header(raw)?))
    }

    fn skip(&mut self, mut length: u64, detail: &'static str) -> Result<(), GaugeError> {
        let mut scratch = [0_u8; 8192];
        while length != 0 {
            let count = usize::try_from(length.min(scratch.len() as u64))
                .map_err(|_| GaugeError::IldgPayload { detail })?;
            self.read_exact(&mut scratch[..count], detail)?;
            length -= count as u64;
        }
        Ok(())
    }

    fn padding(&mut self, length: u64) -> Result<(), GaugeError> {
        let count = ((8 - (length % 8)) % 8) as usize;
        let mut padding = [0_u8; 8];
        self.read_exact(&mut padding[..count], "truncated LIME padding")
    }

    fn trailing_byte(&mut self) -> Result<bool, GaugeError> {
        let mut byte = [0_u8; 1];
        let count = self
            .reader
            .read(&mut byte)
            .map_err(|source| io_error(&self.path, source))?;
        Ok(count != 0)
    }
}

#[derive(Clone, Copy)]
struct RecordHeader {
    flags: u16,
    length: u64,
    kind: RecordKind,
}

#[derive(Clone, Copy)]
enum RecordKind {
    Format,
    Binary,
    Other,
}

fn parse_header(raw: [u8; LIME_HEADER_BYTES]) -> Result<RecordHeader, GaugeError> {
    let magic = u32::from_be_bytes([raw[0], raw[1], raw[2], raw[3]]);
    if magic != LIME_MAGIC {
        return Err(GaugeError::IldgFormat {
            detail: "invalid LIME magic",
        });
    }
    let version = u16::from_be_bytes([raw[4], raw[5]]);
    if version != LIME_VERSION {
        return Err(GaugeError::IldgFormat {
            detail: "unsupported LIME version",
        });
    }
    let flags = u16::from_be_bytes([raw[6], raw[7]]);
    if flags & !(LIME_MB | LIME_ME) != 0 {
        return Err(GaugeError::IldgFormat {
            detail: "reserved LIME flags are set",
        });
    }
    let length = u64::from_be_bytes([
        raw[8], raw[9], raw[10], raw[11], raw[12], raw[13], raw[14], raw[15],
    ]);
    let kind = parse_record_type(&raw[16..])?;
    Ok(RecordHeader {
        flags,
        length,
        kind,
    })
}

fn parse_record_type(raw: &[u8]) -> Result<RecordKind, GaugeError> {
    let end = raw
        .iter()
        .position(|&byte| byte == 0)
        .ok_or(GaugeError::IldgFormat {
            detail: "LIME record type is not NUL padded",
        })?;
    if end == 0 || raw[end + 1..].iter().any(|&byte| byte != 0) {
        return Err(GaugeError::IldgFormat {
            detail: "invalid LIME record type padding",
        });
    }
    let name = &raw[..end];
    if !name.iter().all(u8::is_ascii) {
        return Err(GaugeError::IldgFormat {
            detail: "LIME record type is not ASCII",
        });
    }
    Ok(if name == FORMAT_TYPE {
        RecordKind::Format
    } else if name == BINARY_TYPE {
        RecordKind::Binary
    } else {
        RecordKind::Other
    })
}

fn read_ildg_stream<R: Read>(reader: R, path: &Path) -> Result<GaugeLinks, GaugeError> {
    let mut reader = IldgReader::new(reader, path);
    let mut started = false;
    let mut metadata = None;
    let mut binary = None;

    loop {
        let header = match reader.read_header()? {
            Some(header) => header,
            None => {
                return Err(GaugeError::IldgFormat {
                    detail: if started {
                        "LIME message has no end"
                    } else {
                        "missing LIME message"
                    },
                });
            }
        };

        if !started {
            if header.flags & LIME_MB == 0 {
                return Err(GaugeError::IldgFormat {
                    detail: "LIME message does not begin with MB",
                });
            }
            started = true;
        } else if header.flags & LIME_MB != 0 {
            return Err(GaugeError::IldgFormat {
                detail: "duplicate LIME message begin",
            });
        }

        match header.kind {
            RecordKind::Format => {
                if metadata.is_some() {
                    return Err(GaugeError::IldgFormat {
                        detail: "duplicate or split ildg-format record",
                    });
                }
                let xml = read_xml_record(&mut reader, header.length)?;
                metadata = Some(parse_metadata(&xml)?);
            }
            RecordKind::Binary => {
                if metadata.is_none() {
                    return Err(GaugeError::IldgFormat {
                        detail: "ildg-binary-data precedes ildg-format",
                    });
                }
                if binary.is_some() {
                    return Err(GaugeError::IldgFormat {
                        detail: "duplicate or split ildg-binary-data record",
                    });
                }
                let lattice = metadata_lattice(metadata.as_ref())?;
                let value = read_binary_record(&mut reader, header.length, lattice)?;
                binary = Some(value);
            }
            RecordKind::Other => {
                reader.skip(header.length, "truncated LIME metadata record")?;
                reader.padding(header.length)?;
            }
        }

        if header.flags & LIME_ME != 0 {
            if reader.trailing_byte()? {
                return Err(GaugeError::IldgFormat {
                    detail: "trailing bytes after LIME message",
                });
            }
            break;
        }
    }
    if metadata.is_none() {
        return Err(GaugeError::IldgFormat {
            detail: "missing ildg-format record",
        });
    }
    binary.ok_or(GaugeError::IldgFormat {
        detail: "missing ildg-binary-data record",
    })
}

fn read_xml_record<R: Read>(
    reader: &mut IldgReader<R>,
    length: u64,
) -> Result<Vec<u8>, GaugeError> {
    if length > MAX_XML_BYTES {
        return Err(GaugeError::IldgXml {
            detail: "ildg-format XML exceeds the size limit",
        });
    }
    let length = usize::try_from(length).map_err(|_| GaugeError::IldgXml {
        detail: "ildg-format XML length overflows usize",
    })?;
    let mut xml = vec![0_u8; length];
    reader.read_exact(&mut xml, "truncated ildg-format XML")?;
    reader.padding(length as u64)?;
    Ok(xml)
}

#[derive(Default)]
struct Metadata {
    version: Option<String>,
    field: Option<String>,
    precision: Option<String>,
    lx: Option<String>,
    ly: Option<String>,
    lz: Option<String>,
    lt: Option<String>,
}

#[derive(Clone, Copy)]
enum KnownField {
    Version,
    Field,
    Precision,
    Lx,
    Ly,
    Lz,
    Lt,
}

fn known_field(name: &[u8]) -> Option<KnownField> {
    Some(match name {
        b"version" => KnownField::Version,
        b"field" => KnownField::Field,
        b"precision" => KnownField::Precision,
        b"lx" => KnownField::Lx,
        b"ly" => KnownField::Ly,
        b"lz" => KnownField::Lz,
        b"lt" => KnownField::Lt,
        _ => return None,
    })
}

impl Metadata {
    fn has(&self, field: KnownField) -> bool {
        match field {
            KnownField::Version => self.version.is_some(),
            KnownField::Field => self.field.is_some(),
            KnownField::Precision => self.precision.is_some(),
            KnownField::Lx => self.lx.is_some(),
            KnownField::Ly => self.ly.is_some(),
            KnownField::Lz => self.lz.is_some(),
            KnownField::Lt => self.lt.is_some(),
        }
    }

    fn set(&mut self, field: KnownField, value: String) {
        match field {
            KnownField::Version => self.version = Some(value),
            KnownField::Field => self.field = Some(value),
            KnownField::Precision => self.precision = Some(value),
            KnownField::Lx => self.lx = Some(value),
            KnownField::Ly => self.ly = Some(value),
            KnownField::Lz => self.lz = Some(value),
            KnownField::Lt => self.lt = Some(value),
        }
    }
}

fn parse_metadata(xml: &[u8]) -> Result<Metadata, GaugeError> {
    let mut reader = Reader::from_reader(Cursor::new(xml));
    let mut buffer = Vec::new();
    let mut stack: Vec<Vec<u8>> = Vec::new();
    let mut metadata = Metadata::default();
    let mut active: Option<(KnownField, String)> = None;
    let mut root_seen = false;
    let mut root_closed = false;

    loop {
        let event = reader
            .read_event_into(&mut buffer)
            .map_err(|_| GaugeError::IldgXml {
                detail: "malformed XML",
            })?;
        match event {
            Event::Start(start) => {
                let name = start.name().as_ref().to_vec();
                let local = start.local_name();
                if stack.is_empty() {
                    if root_seen || root_closed || local.as_ref() != b"ildgFormat" {
                        return Err(GaugeError::IldgXml {
                            detail: "XML root must be ildgFormat",
                        });
                    }
                    root_seen = true;
                } else if stack.len() == 1 {
                    if let Some(field) = known_field(local.as_ref()) {
                        if metadata.has(field) {
                            return Err(GaugeError::IldgXml {
                                detail: "duplicate ILDG XML field",
                            });
                        }
                        active = Some((field, String::new()));
                    }
                } else if active.is_some() {
                    return Err(GaugeError::IldgXml {
                        detail: "known ILDG XML field is nested",
                    });
                }
                stack.push(name);
            }
            Event::Empty(empty) => {
                let local = empty.local_name();
                if stack.is_empty() {
                    if root_seen || root_closed || local.as_ref() != b"ildgFormat" {
                        return Err(GaugeError::IldgXml {
                            detail: "XML root must be ildgFormat",
                        });
                    }
                    return Err(GaugeError::IldgXml {
                        detail: "ildgFormat root cannot be empty",
                    });
                }
                if stack.len() == 1 {
                    if let Some(field) = known_field(local.as_ref()) {
                        if metadata.has(field) {
                            return Err(GaugeError::IldgXml {
                                detail: "duplicate ILDG XML field",
                            });
                        }
                        metadata.set(field, String::new());
                    }
                } else if active.is_some() {
                    return Err(GaugeError::IldgXml {
                        detail: "known ILDG XML field is nested",
                    });
                }
            }
            Event::Text(text) => {
                let value = text.unescape().map_err(|_| GaugeError::IldgXml {
                    detail: "invalid XML text",
                })?;
                if let Some((_, contents)) = active.as_mut() {
                    contents.push_str(&value);
                } else if (stack.is_empty() || root_closed || stack.len() == 1)
                    && !value.trim().is_empty()
                {
                    return Err(GaugeError::IldgXml {
                        detail: "unexpected XML text",
                    });
                }
            }
            Event::CData(text) => {
                let value = text.decode().map_err(|_| GaugeError::IldgXml {
                    detail: "invalid XML text",
                })?;
                if let Some((_, contents)) = active.as_mut() {
                    contents.push_str(&value);
                } else if (stack.is_empty() || root_closed || stack.len() == 1)
                    && !value.trim().is_empty()
                {
                    return Err(GaugeError::IldgXml {
                        detail: "unexpected XML text",
                    });
                }
            }
            Event::End(end) => {
                let matches = stack
                    .last()
                    .is_some_and(|name| name.as_slice() == end.name().as_ref());
                if !matches {
                    return Err(GaugeError::IldgXml {
                        detail: "mismatched XML element",
                    });
                }
                let direct_known = stack.len() == 2 && active.is_some();
                stack.pop();
                if direct_known {
                    let (field, value) = active.take().ok_or(GaugeError::IldgXml {
                        detail: "missing XML field state",
                    })?;
                    metadata.set(field, value.trim().to_owned());
                }
                if stack.is_empty() {
                    root_closed = true;
                }
            }
            Event::Comment(_) | Event::Decl(_) | Event::PI(_) => {}
            Event::Eof => {
                if !root_seen || !root_closed || !stack.is_empty() {
                    return Err(GaugeError::IldgXml {
                        detail: "truncated XML document",
                    });
                }
                break;
            }
            Event::DocType(_) => {
                return Err(GaugeError::IldgXml {
                    detail: "unsupported XML construct",
                });
            }
        }
        buffer.clear();
    }

    Ok(metadata)
}

fn required<'a>(value: Option<&'a String>, detail: &'static str) -> Result<&'a str, GaugeError> {
    value
        .map(String::as_str)
        .ok_or(GaugeError::IldgXml { detail })
}

fn parse_dimension(value: &str) -> Result<usize, GaugeError> {
    if value.is_empty() || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err(GaugeError::IldgXml {
            detail: "ILDG dimension is not a decimal integer",
        });
    }
    let value = value.parse::<usize>().map_err(|_| GaugeError::IldgXml {
        detail: "ILDG dimension overflows usize",
    })?;
    if value == 0 {
        return Err(GaugeError::IldgXml {
            detail: "ILDG dimension is not positive",
        });
    }
    Ok(value)
}

fn metadata_lattice(metadata: Option<&Metadata>) -> Result<LatticeShape4, GaugeError> {
    let metadata = metadata.ok_or(GaugeError::IldgFormat {
        detail: "missing ildg-format metadata",
    })?;
    if required(metadata.version.as_ref(), "missing ILDG version")? != "1.0" {
        return Err(GaugeError::IldgXml {
            detail: "unsupported ILDG format version",
        });
    }
    if required(metadata.field.as_ref(), "missing ILDG field")? != "su3gauge" {
        return Err(GaugeError::IldgXml {
            detail: "unsupported ILDG field",
        });
    }
    if required(metadata.precision.as_ref(), "missing ILDG precision")? != "64" {
        return Err(GaugeError::IldgXml {
            detail: "unsupported ILDG precision",
        });
    }
    let extents = [
        parse_dimension(required(metadata.lx.as_ref(), "missing lx")?)?,
        parse_dimension(required(metadata.ly.as_ref(), "missing ly")?)?,
        parse_dimension(required(metadata.lz.as_ref(), "missing lz")?)?,
        parse_dimension(required(metadata.lt.as_ref(), "missing lt")?)?,
    ];
    LatticeShape4::new(extents).map_err(|_| GaugeError::IldgXml {
        detail: "ILDG dimensions overflow the lattice volume",
    })
}

fn checked_value_count(lattice: LatticeShape4) -> Result<usize, GaugeError> {
    let values = 9usize
        .checked_mul(lattice.nv())
        .ok_or(GaugeError::AllocationOverflow)?;
    let bytes = values
        .checked_mul(std::mem::size_of::<Complex64>())
        .ok_or(GaugeError::AllocationOverflow)?;
    if bytes > isize::MAX as usize {
        return Err(GaugeError::AllocationOverflow);
    }
    Ok(values)
}

fn checked_binary_length(lattice: LatticeShape4) -> Result<u64, GaugeError> {
    let values = checked_value_count(lattice)?;
    let components = values
        .checked_mul(4)
        .and_then(|n| n.checked_mul(2))
        .ok_or(GaugeError::AllocationOverflow)?;
    let bytes = components
        .checked_mul(std::mem::size_of::<f64>())
        .ok_or(GaugeError::AllocationOverflow)?;
    if bytes > isize::MAX as usize {
        return Err(GaugeError::AllocationOverflow);
    }
    u64::try_from(bytes).map_err(|_| GaugeError::AllocationOverflow)
}

fn read_binary_record<R: Read>(
    reader: &mut IldgReader<R>,
    length: u64,
    lattice: LatticeShape4,
) -> Result<GaugeLinks, GaugeError> {
    let expected = checked_binary_length(lattice)?;
    if length != expected {
        return Err(GaugeError::IldgPayload {
            detail: "ildg-binary-data length does not match XML dimensions",
        });
    }
    let value_count = checked_value_count(lattice)?;
    let mut values: [Vec<Complex64>; 4] = std::array::from_fn(|_| Vec::with_capacity(value_count));
    let [nx, ny, nz, nt] = lattice.extents();
    for t in 0..nt {
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let site = site_index([x, y, z, t], lattice)?;
                    let mut blocks = [Complex64::default(); 9];
                    for (mu, direction_values) in values.iter_mut().enumerate() {
                        blocks.fill(Complex64::default());
                        for row in 0..3 {
                            for column in 0..3 {
                                let real = read_f64(reader, mu, site, row * 6 + column * 2)?;
                                let imag = read_f64(reader, mu, site, row * 6 + column * 2 + 1)?;
                                blocks[row + 3 * column] = Complex64::new(real, imag);
                            }
                        }
                        direction_values.extend_from_slice(&blocks);
                    }
                }
            }
        }
    }
    reader.padding(length)?;

    let [nx, ny, nz, nt] = lattice.extents();
    let shape = vec![3, 3, nx, ny, nz, nt];
    let mut tensors = Vec::with_capacity(4);
    for values in values {
        tensors.push(
            TypedTensor::from_vec_col_major(shape.clone(), values)
                .map_err(|error| GaugeError::Tensor(error.to_string()))?,
        );
    }
    let tensors: [TypedTensor<Complex64>; 4] = tensors
        .try_into()
        .map_err(|_| GaugeError::Tensor("ILDG requires four tensors".into()))?;
    let links: [GaugeLinkTensor; 4] = tensors
        .into_iter()
        .map(|tensor| GaugeLinkTensor::from_typed(tensor, lattice))
        .collect::<Result<Vec<_>, _>>()?
        .try_into()
        .map_err(|_| GaugeError::Tensor("ILDG requires four link tensors".into()))?;
    GaugeLinks::new(links)
}

fn read_f64<R: Read>(
    reader: &mut IldgReader<R>,
    direction: usize,
    site: usize,
    component: usize,
) -> Result<f64, GaugeError> {
    let mut bytes = [0_u8; 8];
    reader.read_binary_exact(&mut bytes, "truncated ildg-binary-data")?;
    let value = f64::from_bits(u64::from_be_bytes(bytes));
    if !value.is_finite() {
        return Err(GaugeError::IldgNonFinite {
            direction,
            site,
            component,
        });
    }
    Ok(value)
}

fn validate_finite(view: &HostGaugeLinks<'_>) -> Result<(), GaugeError> {
    for mu in 0..4 {
        for site in 0..view.lattice().nv() {
            let matrix = view.link(mu, site)?;
            for (component, value) in matrix.as_array().iter().enumerate() {
                if !value.re.is_finite() {
                    return Err(GaugeError::IldgNonFinite {
                        direction: mu,
                        site,
                        component: component * 2,
                    });
                }
                if !value.im.is_finite() {
                    return Err(GaugeError::IldgNonFinite {
                        direction: mu,
                        site,
                        component: component * 2 + 1,
                    });
                }
            }
        }
    }
    Ok(())
}

fn format_xml(lattice: LatticeShape4) -> Vec<u8> {
    let [lx, ly, lz, lt] = lattice.extents();
    format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<ildgFormat xmlns=\"http://www.lqcd.org/ildg\">\n  <version>1.0</version>\n  <field>su3gauge</field>\n  <precision>64</precision>\n  <lx>{lx}</lx>\n  <ly>{ly}</ly>\n  <lz>{lz}</lz>\n  <lt>{lt}</lt>\n</ildgFormat>\n"
    )
    .into_bytes()
}

fn write_ildg_stream<W: Write>(
    mut writer: W,
    path: &Path,
    view: &HostGaugeLinks<'_>,
    xml: &[u8],
    binary_length: u64,
) -> Result<(), GaugeError> {
    write_record(&mut writer, path, LIME_MB, FORMAT_TYPE, xml)?;
    write_header(&mut writer, path, LIME_ME, BINARY_TYPE, binary_length)?;
    let [nx, ny, nz, nt] = view.lattice().extents();
    for t in 0..nt {
        for z in 0..nz {
            for y in 0..ny {
                for x in 0..nx {
                    let site = site_index([x, y, z, t], view.lattice())?;
                    for mu in 0..4 {
                        let matrix = view.link(mu, site)?;
                        for row in 0..3 {
                            for column in 0..3 {
                                let value = matrix[(row, column)];
                                writer
                                    .write_all(&value.re.to_bits().to_be_bytes())
                                    .map_err(|source| io_error(path, source))?;
                                writer
                                    .write_all(&value.im.to_bits().to_be_bytes())
                                    .map_err(|source| io_error(path, source))?;
                            }
                        }
                    }
                }
            }
        }
    }
    write_padding(&mut writer, path, binary_length)
}

fn write_record<W: Write>(
    writer: &mut W,
    path: &Path,
    flags: u16,
    record_type: &[u8],
    payload: &[u8],
) -> Result<(), GaugeError> {
    let length = u64::try_from(payload.len()).map_err(|_| GaugeError::AllocationOverflow)?;
    write_header(writer, path, flags, record_type, length)?;
    writer
        .write_all(payload)
        .map_err(|source| io_error(path, source))?;
    write_padding(writer, path, length)
}

fn write_header<W: Write>(
    writer: &mut W,
    path: &Path,
    flags: u16,
    record_type: &[u8],
    length: u64,
) -> Result<(), GaugeError> {
    if record_type.is_empty() || record_type.len() >= 128 || !record_type.iter().all(u8::is_ascii) {
        return Err(GaugeError::IldgFormat {
            detail: "invalid LIME record type",
        });
    }
    let mut header = [0_u8; LIME_HEADER_BYTES];
    header[..4].copy_from_slice(&LIME_MAGIC.to_be_bytes());
    header[4..6].copy_from_slice(&LIME_VERSION.to_be_bytes());
    header[6..8].copy_from_slice(&flags.to_be_bytes());
    header[8..16].copy_from_slice(&length.to_be_bytes());
    header[16..16 + record_type.len()].copy_from_slice(record_type);
    writer
        .write_all(&header)
        .map_err(|source| io_error(path, source))
}

fn write_padding<W: Write>(writer: &mut W, path: &Path, length: u64) -> Result<(), GaugeError> {
    let count = ((8 - (length % 8)) % 8) as usize;
    writer
        .write_all(&[0_u8; 8][..count])
        .map_err(|source| io_error(path, source))
}

#[cfg(test)]
mod tests;
