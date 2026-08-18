use super::*;
use crate::{cold_su3, Mat3};
use num_complex::Complex64;
use std::io::Cursor;
use std::path::Path;

fn memory_path() -> &'static Path {
    Path::new("<memory>")
}

fn memory_bytes(links: &GaugeLinks) -> Vec<u8> {
    let view = links.host_view().unwrap();
    let length = checked_binary_length(view.lattice()).unwrap();
    let xml = format_xml(view.lattice());
    let mut output = Vec::new();
    write_ildg_stream(&mut output, memory_path(), &view, &xml, length).unwrap();
    output
}

fn read_memory(bytes: &[u8]) -> Result<GaugeLinks, GaugeError> {
    read_ildg_stream(Cursor::new(bytes), memory_path())
}

fn record(flags: u16, kind: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut output = Vec::new();
    write_header(
        &mut output,
        memory_path(),
        flags,
        kind,
        payload.len() as u64,
    )
    .unwrap();
    output.extend_from_slice(payload);
    output.extend_from_slice(&[0_u8; 8][..((8 - payload.len() as u64 % 8) % 8) as usize]);
    output
}

fn xml(extents: [usize; 4]) -> Vec<u8> {
    format_xml(LatticeShape4::new(extents).unwrap())
}

fn message(xml: &[u8], binary: &[u8]) -> Vec<u8> {
    let mut bytes = record(LIME_MB, FORMAT_TYPE, xml);
    bytes.extend(record(LIME_ME, BINARY_TYPE, binary));
    bytes
}

#[test]
fn cold_and_nontrivial_roundtrip_are_exact() {
    let cold = cold_su3(LatticeShape4::new([1, 1, 1, 1]).unwrap()).unwrap();
    let loaded = read_memory(&memory_bytes(&cold)).unwrap();
    for mu in 0..4 {
        for site in 0..1 {
            assert_eq!(
                loaded.host_view().unwrap().link(mu, site).unwrap(),
                cold.host_view().unwrap().link(mu, site).unwrap()
            );
        }
    }

    let lattice = LatticeShape4::new([2, 2, 1, 2]).unwrap();
    let mut links = cold_su3(lattice).unwrap();
    for mu in 0..4 {
        let mut matrix = Mat3::identity();
        matrix[(0, 1)] = Complex64::new(mu as f64 + 0.25, -0.5);
        crate::store_link(&mut links, mu, mu % lattice.nv(), matrix).unwrap();
    }
    let loaded = read_memory(&memory_bytes(&links)).unwrap();
    for mu in 0..4 {
        for site in 0..lattice.nv() {
            assert_eq!(
                loaded.host_view().unwrap().link(mu, site).unwrap(),
                links.host_view().unwrap().link(mu, site).unwrap()
            );
        }
    }
}

#[test]
fn reader_preserves_ildg_matrix_color_direction_and_site_order() {
    let mut payload = Vec::new();
    for t in 0..2 {
        for z in 0..2 {
            for y in 0..2 {
                for x in 0..2 {
                    let site = x + 2 * (y + 2 * (z + 2 * t));
                    for mu in 0..4 {
                        for row in 0..3 {
                            for column in 0..3 {
                                let bits =
                                    (1_000_000 * mu + 10_000 * site + 100 * column + row) as f64;
                                payload.extend_from_slice(&bits.to_bits().to_be_bytes());
                                payload.extend_from_slice(&(-bits - 0.5).to_bits().to_be_bytes());
                            }
                        }
                    }
                }
            }
        }
    }
    let bytes = message(&xml([2, 2, 2, 2]), &payload);
    let links = read_memory(&bytes).unwrap();
    let matrix = links.host_view().unwrap().link(3, 7).unwrap();
    assert_eq!(matrix[(2, 1)], Complex64::new(3_070_102.0, -3_070_102.5));
}

#[test]
fn unknown_metadata_and_xml_are_tolerated() {
    let base = String::from_utf8(xml([1, 1, 1, 1])).unwrap();
    let xml = base.replace(
        "</ildgFormat>",
        "<metadata><version>ignored</version><nested><lx>ignored</lx></nested></metadata><!-- okay --></ildgFormat>",
    );
    let links = cold_su3(LatticeShape4::new([1, 1, 1, 1]).unwrap()).unwrap();
    let view = links.host_view().unwrap();
    let length = checked_binary_length(view.lattice()).unwrap();
    let mut bytes = record(LIME_MB, FORMAT_TYPE, xml.as_bytes());
    let mut binary = Vec::new();
    write_ildg_stream(
        &mut binary,
        memory_path(),
        &view,
        &format_xml(view.lattice()),
        length,
    )
    .unwrap();
    let base_length = format_xml(view.lattice()).len();
    let header_start = 144 + base_length.div_ceil(8) * 8;
    bytes.extend_from_slice(&binary[header_start..]);
    assert_eq!(read_memory(&bytes).unwrap().lattice(), links.lattice());
}

#[test]
fn namespaced_xml_and_nonzero_padding_are_accepted() {
    let base = String::from_utf8(xml([1, 1, 1, 1])).unwrap();
    let namespaced = base
        .replace(
            "<ildgFormat xmlns=\"http://www.lqcd.org/ildg\">",
            "<ildg:ildgFormat xmlns:ildg=\"urn:example\">",
        )
        .replace("</ildgFormat>", "</ildg:ildgFormat>")
        .replace("<version>", "<ildg:version>")
        .replace("</version>", "</ildg:version>")
        .replace("<field>", "<ildg:field>")
        .replace("</field>", "</ildg:field>")
        .replace("<precision>", "<ildg:precision>")
        .replace("</precision>", "</ildg:precision>")
        .replace("<lx>", "<ildg:lx>")
        .replace("</lx>", "</ildg:lx>")
        .replace("<ly>", "<ildg:ly>")
        .replace("</ly>", "</ildg:ly>")
        .replace("<lz>", "<ildg:lz>")
        .replace("</lz>", "</ildg:lz>")
        .replace("<lt>", "<ildg:lt>")
        .replace("</lt>", "</ildg:lt>");
    let links = cold_su3(LatticeShape4::new([1, 1, 1, 1]).unwrap()).unwrap();
    let view = links.host_view().unwrap();
    let length = checked_binary_length(view.lattice()).unwrap();
    let mut bytes = record(LIME_MB, FORMAT_TYPE, namespaced.as_bytes());
    let mut binary = Vec::new();
    write_ildg_stream(
        &mut binary,
        memory_path(),
        &view,
        &format_xml(view.lattice()),
        length,
    )
    .unwrap();
    let base_length = format_xml(view.lattice()).len();
    let header_start = 144 + base_length.div_ceil(8) * 8;
    bytes.extend_from_slice(&binary[header_start..]);
    let padding_start = 144 + namespaced.len();
    let padding_len = (8 - namespaced.len() % 8) % 8;
    if padding_len != 0 {
        bytes[padding_start..padding_start + padding_len].fill(0xa5);
    }
    assert_eq!(read_memory(&bytes).unwrap().lattice(), links.lattice());
}

#[test]
fn malformed_lime_header_and_message_sequences_are_rejected() {
    assert!(matches!(
        read_memory(&[]),
        Err(GaugeError::IldgFormat { .. })
    ));

    let links = cold_su3(LatticeShape4::new([1, 1, 1, 1]).unwrap()).unwrap();
    let valid = memory_bytes(&links);
    let cases = [
        ("magic", 0usize, 0_u8),
        ("version", 5, 0_u8),
        ("flags", 7, 1_u8),
    ];
    for (name, offset, value) in cases {
        let mut bytes = valid.clone();
        bytes[offset] = value;
        assert!(
            matches!(read_memory(&bytes), Err(GaugeError::IldgFormat { .. })),
            "{name}"
        );
    }

    let mut no_mb = valid.clone();
    no_mb[6] &= !0x80;
    assert!(matches!(
        read_memory(&no_mb),
        Err(GaugeError::IldgFormat { .. })
    ));

    let mut duplicate_mb = valid.clone();
    duplicate_mb[144 + xml([1, 1, 1, 1]).len().div_ceil(8) * 8 + 6] |= 0x80;
    assert!(matches!(
        read_memory(&duplicate_mb),
        Err(GaugeError::IldgFormat { .. })
    ));

    let mut no_me = valid.clone();
    let second_header = 144 + xml([1, 1, 1, 1]).len().div_ceil(8) * 8;
    no_me[second_header + 6] &= !0x40;
    assert!(matches!(
        read_memory(&no_me),
        Err(GaugeError::IldgFormat { .. })
    ));

    let mut trailing = valid.clone();
    trailing.push(1);
    assert!(matches!(
        read_memory(&trailing),
        Err(GaugeError::IldgFormat { .. })
    ));

    let mut second_message = valid.clone();
    second_message.extend_from_slice(&valid);
    assert!(matches!(
        read_memory(&second_message),
        Err(GaugeError::IldgFormat { .. })
    ));
}

#[test]
fn malformed_record_types_and_truncation_are_rejected() {
    let links = cold_su3(LatticeShape4::new([1, 1, 1, 1]).unwrap()).unwrap();
    let valid = memory_bytes(&links);
    for bytes in [valid[..10].to_vec(), valid[..valid.len() - 1].to_vec()] {
        assert!(matches!(
            read_memory(&bytes),
            Err(GaugeError::IldgFormat { .. }) | Err(GaugeError::IldgPayload { .. })
        ));
    }

    let mut bad_type_padding = valid.clone();
    bad_type_padding[16 + FORMAT_TYPE.len() + 1] = 1;
    assert!(matches!(
        read_memory(&bad_type_padding),
        Err(GaugeError::IldgFormat { .. })
    ));

    let mut unknown_type = valid.clone();
    unknown_type[16 + FORMAT_TYPE.len()] = b'x';
    assert!(matches!(
        read_memory(&unknown_type),
        Err(GaugeError::IldgFormat { .. })
    ));

    let mut non_ascii = valid.clone();
    non_ascii[16] = 0xff;
    assert!(matches!(
        read_memory(&non_ascii),
        Err(GaugeError::IldgFormat { .. })
    ));

    let mut empty_type = valid.clone();
    empty_type[16] = 0;
    assert!(matches!(
        read_memory(&empty_type),
        Err(GaugeError::IldgFormat { .. })
    ));

    let mut unknown_truncated = record(LIME_MB, b"unknown", &[1, 2, 3]);
    unknown_truncated.truncate(144 + 2);
    assert!(matches!(
        read_memory(&unknown_truncated),
        Err(GaugeError::IldgFormat { .. })
    ));
}

#[test]
fn duplicate_split_and_missing_required_records_are_rejected() {
    let one = xml([1, 1, 1, 1]);
    let mut duplicate_format = record(LIME_MB, FORMAT_TYPE, &one);
    duplicate_format.extend(record(0, FORMAT_TYPE, &one));
    duplicate_format.extend(record(LIME_ME, BINARY_TYPE, &[0_u8; 576]));
    assert!(matches!(
        read_memory(&duplicate_format),
        Err(GaugeError::IldgFormat { .. })
    ));

    let mut split_binary = record(LIME_MB, FORMAT_TYPE, &one);
    split_binary.extend(record(0, BINARY_TYPE, &[0_u8; 288]));
    split_binary.extend(record(LIME_ME, BINARY_TYPE, &[0_u8; 288]));
    assert!(matches!(
        read_memory(&split_binary),
        Err(GaugeError::IldgFormat { .. }) | Err(GaugeError::IldgPayload { .. })
    ));

    let before = record(LIME_MB, BINARY_TYPE, &[]);
    assert!(matches!(
        read_memory(&before),
        Err(GaugeError::IldgFormat { .. })
    ));

    let only_format = record(LIME_MB | LIME_ME, FORMAT_TYPE, &one);
    assert!(matches!(
        read_memory(&only_format),
        Err(GaugeError::IldgFormat { .. })
    ));

    let only_binary = record(LIME_MB | LIME_ME, BINARY_TYPE, &[0_u8; 576]);
    assert!(matches!(
        read_memory(&only_binary),
        Err(GaugeError::IldgFormat { .. })
    ));

    let unknown_only = record(LIME_MB | LIME_ME, b"unknown", &[]);
    assert!(matches!(
        read_memory(&unknown_only),
        Err(GaugeError::IldgFormat { .. })
    ));
}

#[test]
fn malformed_xml_and_payload_return_typed_errors() {
    let cases = [
        String::from_utf8(xml([1, 1, 1, 1]))
            .unwrap()
            .replace("<lt>1</lt>", "<lt>0</lt>"),
        String::from_utf8(xml([1, 1, 1, 1]))
            .unwrap()
            .replace("<field>su3gauge</field>", "<field>su2gauge</field>"),
        String::from_utf8(xml([1, 1, 1, 1]))
            .unwrap()
            .replace("<lx>1</lx>", "<lx>1</lx><lx>1</lx>"),
        String::from_utf8(xml([1, 1, 1, 1]))
            .unwrap()
            .replace("</ildgFormat>", ""),
        String::from_utf8(xml([1, 1, 1, 1])).unwrap().replace(
            "<version>1.0</version>",
            "<version><nested>1.0</nested></version>",
        ),
        String::from_utf8(xml([1, 1, 1, 1]))
            .unwrap()
            .replace("<ildgFormat", "<wrongRoot"),
    ];
    for xml in cases {
        let bytes = message(xml.as_bytes(), &[0_u8; 576]);
        assert!(matches!(
            read_memory(&bytes),
            Err(GaugeError::IldgXml { .. })
        ));
    }

    let mut cdata_after_root = xml([1, 1, 1, 1]);
    cdata_after_root.extend_from_slice(b"<![CDATA[not allowed here]]>");
    let bytes = message(&cdata_after_root, &[0_u8; 576]);
    assert!(matches!(
        read_memory(&bytes),
        Err(GaugeError::IldgXml { .. })
    ));

    let direct_text =
        String::from_utf8(xml([1, 1, 1, 1]))
            .unwrap()
            .replacen(">\n", ">not allowed\n", 1);
    let bytes = message(direct_text.as_bytes(), &[0_u8; 576]);
    assert!(matches!(
        read_memory(&bytes),
        Err(GaugeError::IldgXml { .. })
    ));

    let mut short = record(LIME_MB, FORMAT_TYPE, &xml([1, 1, 1, 1]));
    short.extend(record(LIME_ME, BINARY_TYPE, &[0_u8; 8]));
    assert!(matches!(
        read_memory(&short),
        Err(GaugeError::IldgPayload { .. })
    ));

    let mut bad_length = record(LIME_MB, FORMAT_TYPE, &xml([1, 1, 1, 1]));
    bad_length.extend(record(LIME_ME, BINARY_TYPE, &[0_u8; 575]));
    assert!(matches!(
        read_memory(&bad_length),
        Err(GaugeError::IldgPayload { .. })
    ));

    let format_xml = xml([1, 1, 1, 1]);
    let format = record(LIME_MB, FORMAT_TYPE, &format_xml);
    assert_ne!(format_xml.len() % 8, 0);
    let truncated_format = &format[..format.len() - 1];
    let mut bytes = truncated_format.to_vec();
    bytes.extend(record(LIME_ME, BINARY_TYPE, &[0_u8; 576]));
    assert!(matches!(
        read_memory(&bytes),
        Err(GaugeError::IldgFormat { .. })
    ));

    let xml_payload = xml([1, 1, 1, 1]);
    let mut truncated_xml = Vec::new();
    write_header(
        &mut truncated_xml,
        memory_path(),
        LIME_MB,
        FORMAT_TYPE,
        xml_payload.len() as u64 + 1,
    )
    .unwrap();
    truncated_xml.extend_from_slice(&xml_payload);
    assert!(matches!(
        read_memory(&truncated_xml),
        Err(GaugeError::IldgFormat { .. })
    ));

    let mut truncated_binary = record(LIME_MB, FORMAT_TYPE, &xml([1, 1, 1, 1]));
    let mut binary_header = Vec::new();
    write_header(&mut binary_header, memory_path(), LIME_ME, BINARY_TYPE, 576).unwrap();
    binary_header.extend_from_slice(&[0_u8; 575]);
    truncated_binary.extend_from_slice(&binary_header);
    assert!(matches!(
        read_memory(&truncated_binary),
        Err(GaugeError::IldgPayload { .. })
    ));

    let mut nonfinite = record(LIME_MB, FORMAT_TYPE, &xml([1, 1, 1, 1]));
    let mut payload = vec![0_u8; 576];
    payload[..8].copy_from_slice(&f64::NAN.to_bits().to_be_bytes());
    nonfinite.extend(record(LIME_ME, BINARY_TYPE, &payload));
    assert!(matches!(
        read_memory(&nonfinite),
        Err(GaugeError::IldgNonFinite { .. })
    ));

    let mut infinity = record(LIME_MB, FORMAT_TYPE, &xml([1, 1, 1, 1]));
    let mut payload = vec![0_u8; 576];
    payload[8..16].copy_from_slice(&f64::INFINITY.to_bits().to_be_bytes());
    infinity.extend(record(LIME_ME, BINARY_TYPE, &payload));
    assert!(matches!(
        read_memory(&infinity),
        Err(GaugeError::IldgNonFinite { .. })
    ));
}

#[test]
fn xml_size_and_dimension_overflow_are_rejected_before_allocation() {
    let oversized = vec![b'x'; (MAX_XML_BYTES + 1) as usize];
    let mut bytes = record(LIME_MB, FORMAT_TYPE, &oversized);
    bytes.extend(record(LIME_ME, BINARY_TYPE, &[]));
    assert!(matches!(
        read_memory(&bytes),
        Err(GaugeError::IldgXml { .. })
    ));

    let too_large = format!(
        "<ildgFormat><version>1.0</version><field>su3gauge</field><precision>64</precision><lx>{}</lx><ly>1</ly><lz>1</lz><lt>1</lt></ildgFormat>",
        usize::MAX
    );
    let bytes = message(too_large.as_bytes(), &[]);
    assert!(matches!(
        read_memory(&bytes),
        Err(GaugeError::AllocationOverflow)
    ));

    let volume_overflow = format!(
        "<ildgFormat><version>1.0</version><field>su3gauge</field><precision>64</precision><lx>{}</lx><ly>2</ly><lz>1</lz><lt>1</lt></ildgFormat>",
        usize::MAX
    );
    let bytes = message(volume_overflow.as_bytes(), &[]);
    assert!(matches!(
        read_memory(&bytes),
        Err(GaugeError::IldgXml { .. })
    ));

    if usize::BITS >= 64 {
        let extent = (isize::MAX as usize) / (9 * 4 * 2 * std::mem::size_of::<f64>()) + 1;
        let lattice = LatticeShape4::new([extent, 1, 1, 1]).unwrap();
        assert!(matches!(
            checked_binary_length(lattice),
            Err(GaugeError::AllocationOverflow)
        ));
    }
}

#[test]
fn missing_xml_fields_and_unsupported_values_are_rejected() {
    let base = String::from_utf8(xml([1, 1, 1, 1])).unwrap();
    for field in ["version", "field", "precision", "lx", "ly", "lz", "lt"] {
        let open = format!("<{field}>");
        let close = format!("</{field}>");
        let missing = base.replace(
            &format!(
                "{open}{}{}",
                if field == "field" {
                    "su3gauge"
                } else if field == "version" {
                    "1.0"
                } else if field == "precision" {
                    "64"
                } else {
                    "1"
                },
                close
            ),
            "",
        );
        let bytes = message(missing.as_bytes(), &[0_u8; 576]);
        assert!(
            matches!(read_memory(&bytes), Err(GaugeError::IldgXml { .. })),
            "missing {field}"
        );
    }

    for replacement in [
        ("<version>1.0</version>", "<version>1.1</version>"),
        ("<field>su3gauge</field>", "<field>su4gauge</field>"),
        ("<precision>64</precision>", "<precision>32</precision>"),
        ("<lx>1</lx>", "<lx>+1</lx>"),
    ] {
        let unsupported = base.replace(replacement.0, replacement.1);
        let bytes = message(unsupported.as_bytes(), &[0_u8; 576]);
        assert!(matches!(
            read_memory(&bytes),
            Err(GaugeError::IldgXml { .. })
        ));
    }
}
