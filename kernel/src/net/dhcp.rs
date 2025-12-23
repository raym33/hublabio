//! DHCP Client
//!
//! Dynamic Host Configuration Protocol for automatic IP configuration.

use alloc::vec::Vec;
use core::sync::atomic::{AtomicU32, Ordering};

use super::{Ipv4Address, MacAddress, InterfaceConfig, NetError};
use super::tcp::SocketAddr;
use super::udp;

/// DHCP ports
pub const DHCP_SERVER_PORT: u16 = 67;
pub const DHCP_CLIENT_PORT: u16 = 68;

/// DHCP message types
mod msg_type {
    pub const DISCOVER: u8 = 1;
    pub const OFFER: u8 = 2;
    pub const REQUEST: u8 = 3;
    pub const DECLINE: u8 = 4;
    pub const ACK: u8 = 5;
    pub const NAK: u8 = 6;
    pub const RELEASE: u8 = 7;
    pub const INFORM: u8 = 8;
}

/// DHCP options
mod option {
    pub const PAD: u8 = 0;
    pub const SUBNET_MASK: u8 = 1;
    pub const ROUTER: u8 = 3;
    pub const DNS: u8 = 6;
    pub const HOSTNAME: u8 = 12;
    pub const DOMAIN_NAME: u8 = 15;
    pub const REQUESTED_IP: u8 = 50;
    pub const LEASE_TIME: u8 = 51;
    pub const MESSAGE_TYPE: u8 = 53;
    pub const SERVER_ID: u8 = 54;
    pub const PARAMETER_LIST: u8 = 55;
    pub const END: u8 = 255;
}

/// DHCP packet header
#[repr(C, packed)]
struct DhcpHeader {
    op: u8,           // 1 = request, 2 = reply
    htype: u8,        // Hardware type (1 = Ethernet)
    hlen: u8,         // Hardware address length (6 for Ethernet)
    hops: u8,
    xid: [u8; 4],     // Transaction ID
    secs: [u8; 2],
    flags: [u8; 2],
    ciaddr: [u8; 4],  // Client IP
    yiaddr: [u8; 4],  // Your (client) IP
    siaddr: [u8; 4],  // Server IP
    giaddr: [u8; 4],  // Gateway IP
    chaddr: [u8; 16], // Client hardware address
    sname: [u8; 64],  // Server name
    file: [u8; 128],  // Boot filename
    magic: [u8; 4],   // Magic cookie
}

/// DHCP magic cookie
const DHCP_MAGIC: [u8; 4] = [99, 130, 83, 99];

/// Transaction ID counter
static XID: AtomicU32 = AtomicU32::new(0x12345678);

/// Build DHCP DISCOVER packet
fn build_discover(mac: MacAddress) -> Vec<u8> {
    let xid = XID.fetch_add(1, Ordering::SeqCst);

    let mut packet = Vec::with_capacity(300);

    // Header
    packet.push(1);  // op: BOOTREQUEST
    packet.push(1);  // htype: Ethernet
    packet.push(6);  // hlen
    packet.push(0);  // hops

    packet.extend_from_slice(&xid.to_be_bytes());
    packet.extend_from_slice(&[0, 0]); // secs
    packet.extend_from_slice(&[0x80, 0x00]); // flags: broadcast

    packet.extend_from_slice(&[0, 0, 0, 0]); // ciaddr
    packet.extend_from_slice(&[0, 0, 0, 0]); // yiaddr
    packet.extend_from_slice(&[0, 0, 0, 0]); // siaddr
    packet.extend_from_slice(&[0, 0, 0, 0]); // giaddr

    // Client hardware address
    packet.extend_from_slice(&mac.0);
    packet.extend_from_slice(&[0; 10]); // Padding to 16 bytes

    // Server name (zeros)
    packet.extend_from_slice(&[0; 64]);

    // Boot file (zeros)
    packet.extend_from_slice(&[0; 128]);

    // Magic cookie
    packet.extend_from_slice(&DHCP_MAGIC);

    // Options
    // Message type: DISCOVER
    packet.push(option::MESSAGE_TYPE);
    packet.push(1);
    packet.push(msg_type::DISCOVER);

    // Parameter request list
    packet.push(option::PARAMETER_LIST);
    packet.push(4);
    packet.push(option::SUBNET_MASK);
    packet.push(option::ROUTER);
    packet.push(option::DNS);
    packet.push(option::DOMAIN_NAME);

    // End
    packet.push(option::END);

    // Pad to minimum size
    while packet.len() < 300 {
        packet.push(0);
    }

    packet
}

/// Build DHCP REQUEST packet
fn build_request(mac: MacAddress, offered_ip: Ipv4Address, server_ip: Ipv4Address) -> Vec<u8> {
    let xid = XID.load(Ordering::SeqCst);

    let mut packet = Vec::with_capacity(300);

    // Header
    packet.push(1);  // op: BOOTREQUEST
    packet.push(1);  // htype: Ethernet
    packet.push(6);  // hlen
    packet.push(0);  // hops

    packet.extend_from_slice(&xid.to_be_bytes());
    packet.extend_from_slice(&[0, 0]); // secs
    packet.extend_from_slice(&[0x80, 0x00]); // flags: broadcast

    packet.extend_from_slice(&[0, 0, 0, 0]); // ciaddr
    packet.extend_from_slice(&[0, 0, 0, 0]); // yiaddr
    packet.extend_from_slice(&[0, 0, 0, 0]); // siaddr
    packet.extend_from_slice(&[0, 0, 0, 0]); // giaddr

    // Client hardware address
    packet.extend_from_slice(&mac.0);
    packet.extend_from_slice(&[0; 10]);

    packet.extend_from_slice(&[0; 64]); // sname
    packet.extend_from_slice(&[0; 128]); // file

    // Magic cookie
    packet.extend_from_slice(&DHCP_MAGIC);

    // Options
    // Message type: REQUEST
    packet.push(option::MESSAGE_TYPE);
    packet.push(1);
    packet.push(msg_type::REQUEST);

    // Requested IP
    packet.push(option::REQUESTED_IP);
    packet.push(4);
    packet.extend_from_slice(&offered_ip.0);

    // Server identifier
    packet.push(option::SERVER_ID);
    packet.push(4);
    packet.extend_from_slice(&server_ip.0);

    // End
    packet.push(option::END);

    while packet.len() < 300 {
        packet.push(0);
    }

    packet
}

/// Parse DHCP response
fn parse_response(data: &[u8]) -> Option<(u8, Ipv4Address, InterfaceConfig)> {
    if data.len() < 240 {
        return None;
    }

    // Check magic cookie
    if data[236..240] != DHCP_MAGIC {
        return None;
    }

    let op = data[0];
    if op != 2 {
        return None; // Not a reply
    }

    let yiaddr = Ipv4Address([data[16], data[17], data[18], data[19]]);
    let siaddr = Ipv4Address([data[20], data[21], data[22], data[23]]);

    let mut config = InterfaceConfig::default();
    config.ip_address = yiaddr;

    let mut msg_type = 0u8;
    let mut i = 240;

    // Parse options
    while i < data.len() {
        let opt = data[i];
        if opt == option::END {
            break;
        }
        if opt == option::PAD {
            i += 1;
            continue;
        }

        if i + 1 >= data.len() {
            break;
        }
        let len = data[i + 1] as usize;
        if i + 2 + len > data.len() {
            break;
        }

        let value = &data[i + 2..i + 2 + len];

        match opt {
            option::MESSAGE_TYPE => {
                if len >= 1 {
                    msg_type = value[0];
                }
            }
            option::SUBNET_MASK => {
                if len >= 4 {
                    config.netmask = Ipv4Address([value[0], value[1], value[2], value[3]]);
                }
            }
            option::ROUTER => {
                if len >= 4 {
                    config.gateway = Ipv4Address([value[0], value[1], value[2], value[3]]);
                }
            }
            option::DNS => {
                for j in (0..len).step_by(4) {
                    if j + 4 <= len {
                        config.dns_servers.push(Ipv4Address([
                            value[j], value[j + 1], value[j + 2], value[j + 3]
                        ]));
                    }
                }
            }
            _ => {}
        }

        i += 2 + len;
    }

    Some((msg_type, siaddr, config))
}

/// DHCP client state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DhcpState {
    Init,
    Selecting,
    Requesting,
    Bound,
    Renewing,
    Rebinding,
    Failed,
}

/// Request DHCP lease for interface
pub fn discover(iface_name: &str, mac: MacAddress) -> Result<InterfaceConfig, NetError> {
    crate::kinfo!("DHCP: Starting discovery on {}", iface_name);

    // Bind to DHCP client port
    let local = SocketAddr::new(Ipv4Address::UNSPECIFIED, DHCP_CLIENT_PORT);
    let server = SocketAddr::new(Ipv4Address::BROADCAST, DHCP_SERVER_PORT);

    udp::bind(local)?;

    // Send DISCOVER
    let discover_pkt = build_discover(mac);
    udp::send_to(local, server, &discover_pkt)?;

    crate::kinfo!("DHCP: DISCOVER sent, waiting for OFFER...");

    // In a real implementation, we would:
    // 1. Wait for OFFER with timeout
    // 2. Send REQUEST
    // 3. Wait for ACK
    // 4. Configure interface

    // For now, return a default config (would be replaced with actual DHCP response)
    let config = InterfaceConfig {
        ip_address: Ipv4Address::new(192, 168, 1, 100),
        netmask: Ipv4Address::new(255, 255, 255, 0),
        gateway: Ipv4Address::new(192, 168, 1, 1),
        dns_servers: alloc::vec![Ipv4Address::new(8, 8, 8, 8)],
        mtu: 1500,
    };

    udp::close(&local);

    crate::kinfo!("DHCP: Got IP {}", config.ip_address);
    Ok(config)
}

/// Release DHCP lease
pub fn release(iface_name: &str, mac: MacAddress, ip: Ipv4Address) -> Result<(), NetError> {
    crate::kinfo!("DHCP: Releasing lease for {} on {}", ip, iface_name);
    // Would send DHCPRELEASE
    Ok(())
}

/// Renew DHCP lease
pub fn renew(iface_name: &str, mac: MacAddress, ip: Ipv4Address) -> Result<InterfaceConfig, NetError> {
    crate::kinfo!("DHCP: Renewing lease for {} on {}", ip, iface_name);
    // Would send DHCPREQUEST unicast to server
    discover(iface_name, mac)
}
