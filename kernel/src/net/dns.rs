//! DNS Resolver
//!
//! Domain Name System client for hostname resolution.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use spin::Mutex;

use super::tcp::SocketAddr;
use super::udp;
use super::{Ipv4Address, NetError};

/// DNS port
pub const DNS_PORT: u16 = 53;

/// DNS record types
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordType {
    A = 1,     // IPv4 address
    NS = 2,    // Name server
    CNAME = 5, // Canonical name
    SOA = 6,   // Start of authority
    PTR = 12,  // Pointer (reverse lookup)
    MX = 15,   // Mail exchange
    TXT = 16,  // Text record
    AAAA = 28, // IPv6 address
    SRV = 33,  // Service record
}

/// DNS query class
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RecordClass {
    IN = 1, // Internet
    CS = 2, // CSNET
    CH = 3, // CHAOS
    HS = 4, // Hesiod
}

/// DNS response code
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResponseCode {
    NoError = 0,
    FormatError = 1,
    ServerFailure = 2,
    NameError = 3, // NXDOMAIN
    NotImplemented = 4,
    Refused = 5,
}

/// DNS header
#[repr(C, packed)]
#[derive(Clone, Copy)]
struct DnsHeader {
    id: u16,
    flags: u16,
    qdcount: u16, // Question count
    ancount: u16, // Answer count
    nscount: u16, // Authority count
    arcount: u16, // Additional count
}

impl DnsHeader {
    fn new_query(id: u16) -> Self {
        Self {
            id: id.to_be(),
            flags: 0x0100_u16.to_be(), // Standard query, recursion desired
            qdcount: 1_u16.to_be(),
            ancount: 0,
            nscount: 0,
            arcount: 0,
        }
    }

    fn is_response(&self) -> bool {
        (u16::from_be(self.flags) & 0x8000) != 0
    }

    fn response_code(&self) -> ResponseCode {
        match u16::from_be(self.flags) & 0x000F {
            0 => ResponseCode::NoError,
            1 => ResponseCode::FormatError,
            2 => ResponseCode::ServerFailure,
            3 => ResponseCode::NameError,
            4 => ResponseCode::NotImplemented,
            5 => ResponseCode::Refused,
            _ => ResponseCode::ServerFailure,
        }
    }

    fn answer_count(&self) -> u16 {
        u16::from_be(self.ancount)
    }
}

/// DNS resource record
#[derive(Clone, Debug)]
pub struct DnsRecord {
    pub name: String,
    pub record_type: RecordType,
    pub class: RecordClass,
    pub ttl: u32,
    pub data: DnsRecordData,
}

/// DNS record data
#[derive(Clone, Debug)]
pub enum DnsRecordData {
    A(Ipv4Address),
    AAAA([u8; 16]),
    CNAME(String),
    MX {
        priority: u16,
        exchange: String,
    },
    TXT(String),
    NS(String),
    PTR(String),
    SRV {
        priority: u16,
        weight: u16,
        port: u16,
        target: String,
    },
    Unknown(Vec<u8>),
}

/// DNS cache entry
struct CacheEntry {
    records: Vec<DnsRecord>,
    expires_at: u64,
}

/// DNS cache
static DNS_CACHE: Mutex<BTreeMap<String, CacheEntry>> = Mutex::new(BTreeMap::new());

/// DNS servers
static DNS_SERVERS: Mutex<Vec<Ipv4Address>> = Mutex::new(Vec::new());

/// Transaction ID counter
static NEXT_ID: Mutex<u16> = Mutex::new(1);

/// Get next transaction ID
fn next_id() -> u16 {
    let mut id = NEXT_ID.lock();
    let current = *id;
    *id = id.wrapping_add(1);
    current
}

/// Set DNS servers
pub fn set_servers(servers: Vec<Ipv4Address>) {
    *DNS_SERVERS.lock() = servers;
}

/// Add DNS server
pub fn add_server(server: Ipv4Address) {
    DNS_SERVERS.lock().push(server);
}

/// Get DNS servers
pub fn get_servers() -> Vec<Ipv4Address> {
    DNS_SERVERS.lock().clone()
}

/// Encode domain name in DNS format
fn encode_name(name: &str) -> Vec<u8> {
    let mut encoded = Vec::new();

    for label in name.split('.') {
        if label.is_empty() {
            continue;
        }
        encoded.push(label.len() as u8);
        encoded.extend_from_slice(label.as_bytes());
    }

    encoded.push(0); // Root label
    encoded
}

/// Decode domain name from DNS message
fn decode_name(data: &[u8], offset: &mut usize) -> Result<String, NetError> {
    let mut name = String::new();
    let mut jumped = false;
    let mut jump_offset = 0;

    loop {
        if *offset >= data.len() {
            return Err(NetError::InvalidData);
        }

        let len = data[*offset];

        // Check for pointer
        if (len & 0xC0) == 0xC0 {
            if *offset + 1 >= data.len() {
                return Err(NetError::InvalidData);
            }

            if !jumped {
                jump_offset = *offset + 2;
            }

            let pointer = ((len as usize & 0x3F) << 8) | data[*offset + 1] as usize;
            *offset = pointer;
            jumped = true;
            continue;
        }

        if len == 0 {
            *offset += 1;
            break;
        }

        *offset += 1;

        if *offset + len as usize > data.len() {
            return Err(NetError::InvalidData);
        }

        if !name.is_empty() {
            name.push('.');
        }

        let label = core::str::from_utf8(&data[*offset..*offset + len as usize])
            .map_err(|_| NetError::InvalidData)?;
        name.push_str(label);

        *offset += len as usize;
    }

    if jumped {
        *offset = jump_offset;
    }

    Ok(name)
}

/// Build DNS query packet
fn build_query(name: &str, record_type: RecordType) -> Vec<u8> {
    let id = next_id();
    let header = DnsHeader::new_query(id);

    let mut packet = Vec::with_capacity(512);

    // Header
    let header_bytes: [u8; 12] = unsafe { core::mem::transmute(header) };
    packet.extend_from_slice(&header_bytes);

    // Question
    packet.extend(encode_name(name));
    packet.extend_from_slice(&(record_type as u16).to_be_bytes());
    packet.extend_from_slice(&(RecordClass::IN as u16).to_be_bytes());

    packet
}

/// Parse DNS response
fn parse_response(data: &[u8]) -> Result<Vec<DnsRecord>, NetError> {
    if data.len() < 12 {
        return Err(NetError::InvalidData);
    }

    let header: DnsHeader = unsafe { core::ptr::read_unaligned(data.as_ptr() as *const DnsHeader) };

    if !header.is_response() {
        return Err(NetError::InvalidData);
    }

    if header.response_code() != ResponseCode::NoError {
        return Err(NetError::HostNotFound);
    }

    let mut offset = 12;

    // Skip questions
    let qdcount = u16::from_be(header.qdcount);
    for _ in 0..qdcount {
        decode_name(data, &mut offset)?;
        offset += 4; // Type + Class
    }

    // Parse answers
    let mut records = Vec::new();
    let ancount = header.answer_count();

    for _ in 0..ancount {
        let name = decode_name(data, &mut offset)?;

        if offset + 10 > data.len() {
            break;
        }

        let rtype = u16::from_be_bytes([data[offset], data[offset + 1]]);
        let rclass = u16::from_be_bytes([data[offset + 2], data[offset + 3]]);
        let ttl = u32::from_be_bytes([
            data[offset + 4],
            data[offset + 5],
            data[offset + 6],
            data[offset + 7],
        ]);
        let rdlength = u16::from_be_bytes([data[offset + 8], data[offset + 9]]) as usize;
        offset += 10;

        if offset + rdlength > data.len() {
            break;
        }

        let record_type = match rtype {
            1 => RecordType::A,
            2 => RecordType::NS,
            5 => RecordType::CNAME,
            6 => RecordType::SOA,
            12 => RecordType::PTR,
            15 => RecordType::MX,
            16 => RecordType::TXT,
            28 => RecordType::AAAA,
            33 => RecordType::SRV,
            _ => {
                offset += rdlength;
                continue;
            }
        };

        let rdata = &data[offset..offset + rdlength];
        let record_data = match record_type {
            RecordType::A => {
                if rdlength >= 4 {
                    DnsRecordData::A(Ipv4Address([rdata[0], rdata[1], rdata[2], rdata[3]]))
                } else {
                    DnsRecordData::Unknown(rdata.to_vec())
                }
            }
            RecordType::AAAA => {
                if rdlength >= 16 {
                    let mut addr = [0u8; 16];
                    addr.copy_from_slice(&rdata[..16]);
                    DnsRecordData::AAAA(addr)
                } else {
                    DnsRecordData::Unknown(rdata.to_vec())
                }
            }
            RecordType::CNAME | RecordType::NS | RecordType::PTR => {
                let mut name_offset = offset;
                match decode_name(data, &mut name_offset) {
                    Ok(cname) => match record_type {
                        RecordType::CNAME => DnsRecordData::CNAME(cname),
                        RecordType::NS => DnsRecordData::NS(cname),
                        RecordType::PTR => DnsRecordData::PTR(cname),
                        _ => unreachable!(),
                    },
                    Err(_) => DnsRecordData::Unknown(rdata.to_vec()),
                }
            }
            RecordType::MX => {
                if rdlength >= 2 {
                    let priority = u16::from_be_bytes([rdata[0], rdata[1]]);
                    let mut name_offset = offset + 2;
                    match decode_name(data, &mut name_offset) {
                        Ok(exchange) => DnsRecordData::MX { priority, exchange },
                        Err(_) => DnsRecordData::Unknown(rdata.to_vec()),
                    }
                } else {
                    DnsRecordData::Unknown(rdata.to_vec())
                }
            }
            RecordType::TXT => {
                let txt = String::from_utf8_lossy(rdata).to_string();
                DnsRecordData::TXT(txt)
            }
            RecordType::SRV => {
                if rdlength >= 6 {
                    let priority = u16::from_be_bytes([rdata[0], rdata[1]]);
                    let weight = u16::from_be_bytes([rdata[2], rdata[3]]);
                    let port = u16::from_be_bytes([rdata[4], rdata[5]]);
                    let mut name_offset = offset + 6;
                    match decode_name(data, &mut name_offset) {
                        Ok(target) => DnsRecordData::SRV {
                            priority,
                            weight,
                            port,
                            target,
                        },
                        Err(_) => DnsRecordData::Unknown(rdata.to_vec()),
                    }
                } else {
                    DnsRecordData::Unknown(rdata.to_vec())
                }
            }
            _ => DnsRecordData::Unknown(rdata.to_vec()),
        };

        records.push(DnsRecord {
            name,
            record_type,
            class: RecordClass::IN,
            ttl,
            data: record_data,
        });

        offset += rdlength;
    }

    Ok(records)
}

/// Resolve hostname to IPv4 address
pub fn resolve(hostname: &str) -> Result<Ipv4Address, NetError> {
    // Check cache first
    {
        let cache = DNS_CACHE.lock();
        if let Some(entry) = cache.get(hostname) {
            for record in &entry.records {
                if let DnsRecordData::A(addr) = record.data {
                    return Ok(addr);
                }
            }
        }
    }

    // Get DNS server
    let servers = DNS_SERVERS.lock();
    let server = servers
        .first()
        .cloned()
        .unwrap_or(Ipv4Address::new(8, 8, 8, 8));
    drop(servers);

    // Build query
    let query = build_query(hostname, RecordType::A);

    // Send query via UDP
    let local = SocketAddr::new(Ipv4Address::UNSPECIFIED, 0);
    let remote = SocketAddr::new(server, DNS_PORT);

    udp::bind(local)?;
    udp::send_to(local, remote, &query)?;

    // Wait for response (with timeout)
    // In a real implementation, this would use async/await or polling
    let response = alloc::vec![0u8; 512]; // Placeholder

    // Parse response
    let records = parse_response(&response)?;

    // Cache results
    if !records.is_empty() {
        let ttl = records.first().map(|r| r.ttl).unwrap_or(300);
        let mut cache = DNS_CACHE.lock();
        cache.insert(
            String::from(hostname),
            CacheEntry {
                records: records.clone(),
                expires_at: 0, // Would use system time + TTL
            },
        );
    }

    // Return first A record
    for record in records {
        if let DnsRecordData::A(addr) = record.data {
            return Ok(addr);
        }
    }

    Err(NetError::HostNotFound)
}

/// Resolve hostname to all IPv4 addresses
pub fn resolve_all(hostname: &str) -> Result<Vec<Ipv4Address>, NetError> {
    let mut addresses = Vec::new();

    // Check cache
    {
        let cache = DNS_CACHE.lock();
        if let Some(entry) = cache.get(hostname) {
            for record in &entry.records {
                if let DnsRecordData::A(addr) = record.data {
                    addresses.push(addr);
                }
            }
            if !addresses.is_empty() {
                return Ok(addresses);
            }
        }
    }

    // Query DNS server
    let servers = DNS_SERVERS.lock();
    let server = servers
        .first()
        .cloned()
        .unwrap_or(Ipv4Address::new(8, 8, 8, 8));
    drop(servers);

    let query = build_query(hostname, RecordType::A);

    // Send and receive (simplified)
    let local = SocketAddr::new(Ipv4Address::UNSPECIFIED, 0);
    let remote = SocketAddr::new(server, DNS_PORT);

    udp::bind(local)?;
    udp::send_to(local, remote, &query)?;

    // Parse response (placeholder)
    let response = alloc::vec![0u8; 512];
    let records = parse_response(&response)?;

    for record in records {
        if let DnsRecordData::A(addr) = record.data {
            addresses.push(addr);
        }
    }

    if addresses.is_empty() {
        Err(NetError::HostNotFound)
    } else {
        Ok(addresses)
    }
}

/// Reverse DNS lookup
pub fn reverse_lookup(addr: Ipv4Address) -> Result<String, NetError> {
    // Build reverse lookup name
    let name = alloc::format!(
        "{}.{}.{}.{}.in-addr.arpa",
        addr.0[3],
        addr.0[2],
        addr.0[1],
        addr.0[0]
    );

    let query = build_query(&name, RecordType::PTR);

    let servers = DNS_SERVERS.lock();
    let server = servers
        .first()
        .cloned()
        .unwrap_or(Ipv4Address::new(8, 8, 8, 8));
    drop(servers);

    let local = SocketAddr::new(Ipv4Address::UNSPECIFIED, 0);
    let remote = SocketAddr::new(server, DNS_PORT);

    udp::bind(local)?;
    udp::send_to(local, remote, &query)?;

    // Parse response (placeholder)
    let response = alloc::vec![0u8; 512];
    let records = parse_response(&response)?;

    for record in records {
        if let DnsRecordData::PTR(hostname) = record.data {
            return Ok(hostname);
        }
    }

    Err(NetError::HostNotFound)
}

/// Clear DNS cache
pub fn clear_cache() {
    DNS_CACHE.lock().clear();
}

/// Initialize DNS resolver
pub fn init() {
    // Add default DNS servers
    add_server(Ipv4Address::new(8, 8, 8, 8)); // Google DNS
    add_server(Ipv4Address::new(8, 8, 4, 4)); // Google DNS secondary
    add_server(Ipv4Address::new(1, 1, 1, 1)); // Cloudflare DNS

    crate::kprintln!("  DNS resolver initialized");
}
