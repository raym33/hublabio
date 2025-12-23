//! Netfilter - Packet Filtering Framework
//!
//! Linux-compatible firewall with iptables-style rules.
//! Implements filter, nat, and mangle tables with chains.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicU64, Ordering};
use spin::RwLock;

/// Netfilter hook points
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Hook {
    /// Before routing decision (incoming)
    PreRouting = 0,
    /// Local input (to this host)
    Input = 1,
    /// Forwarding (routing)
    Forward = 2,
    /// Local output (from this host)
    Output = 3,
    /// After routing (outgoing)
    PostRouting = 4,
}

impl Hook {
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Hook::PreRouting),
            1 => Some(Hook::Input),
            2 => Some(Hook::Forward),
            3 => Some(Hook::Output),
            4 => Some(Hook::PostRouting),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Hook::PreRouting => "PREROUTING",
            Hook::Input => "INPUT",
            Hook::Forward => "FORWARD",
            Hook::Output => "OUTPUT",
            Hook::PostRouting => "POSTROUTING",
        }
    }
}

/// Netfilter tables
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Table {
    /// Packet filtering (drop/accept)
    Filter,
    /// Network address translation
    Nat,
    /// Packet modification
    Mangle,
    /// Raw (before connection tracking)
    Raw,
    /// Security (SELinux)
    Security,
}

impl Table {
    pub fn name(&self) -> &'static str {
        match self {
            Table::Filter => "filter",
            Table::Nat => "nat",
            Table::Mangle => "mangle",
            Table::Raw => "raw",
            Table::Security => "security",
        }
    }

    pub fn valid_hooks(&self) -> &[Hook] {
        match self {
            Table::Filter => &[Hook::Input, Hook::Forward, Hook::Output],
            Table::Nat => &[Hook::PreRouting, Hook::Input, Hook::Output, Hook::PostRouting],
            Table::Mangle => &[Hook::PreRouting, Hook::Input, Hook::Forward, Hook::Output, Hook::PostRouting],
            Table::Raw => &[Hook::PreRouting, Hook::Output],
            Table::Security => &[Hook::Input, Hook::Forward, Hook::Output],
        }
    }
}

/// Packet verdict
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Accept packet
    Accept,
    /// Drop packet silently
    Drop,
    /// Reject with ICMP error
    Reject,
    /// Queue to userspace
    Queue,
    /// Continue processing
    Continue,
    /// Return from chain
    Return,
    /// Jump to another chain
    Jump(u32), // chain index
}

/// Match target (what to match against)
#[derive(Clone, Debug)]
pub enum Match {
    /// Source IP address (addr, mask)
    SrcIp(u32, u32),
    /// Destination IP address (addr, mask)
    DstIp(u32, u32),
    /// Source port (for TCP/UDP)
    SrcPort(u16, u16), // min, max
    /// Destination port
    DstPort(u16, u16),
    /// Protocol (TCP=6, UDP=17, ICMP=1)
    Protocol(u8),
    /// Input interface
    InInterface(String),
    /// Output interface
    OutInterface(String),
    /// TCP flags
    TcpFlags(u8, u8), // mask, expected
    /// ICMP type
    IcmpType(u8),
    /// Connection state
    State(ConnState),
    /// MAC address source
    MacSrc([u8; 6]),
    /// Packet mark
    Mark(u32, u32), // value, mask
    /// Rate limit (packets/second)
    Limit(u32, u32), // rate, burst
    /// Owner UID
    OwnerUid(u32),
    /// Owner GID
    OwnerGid(u32),
}

/// Connection tracking state
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConnState {
    /// New connection
    New,
    /// Established connection
    Established,
    /// Related to existing connection
    Related,
    /// Invalid packet
    Invalid,
    /// Untracked
    Untracked,
}

/// Target action
#[derive(Clone, Debug)]
pub enum Target {
    /// Accept packet
    Accept,
    /// Drop packet
    Drop,
    /// Reject with ICMP message
    Reject(RejectType),
    /// Log the packet
    Log(LogOptions),
    /// Source NAT
    Snat(u32, Option<u16>), // new_addr, new_port
    /// Destination NAT
    Dnat(u32, Option<u16>),
    /// Masquerade (dynamic SNAT)
    Masquerade,
    /// Redirect to local port
    Redirect(u16),
    /// Set packet mark
    Mark(u32),
    /// Set DSCP value
    Dscp(u8),
    /// Set TTL
    Ttl(u8),
    /// Jump to chain
    Jump(String),
    /// Return from chain
    Return,
    /// Go to chain (no return)
    Goto(String),
}

/// Reject type
#[derive(Clone, Copy, Debug)]
pub enum RejectType {
    /// ICMP port unreachable
    IcmpPortUnreach,
    /// ICMP host unreachable
    IcmpHostUnreach,
    /// ICMP network unreachable
    IcmpNetUnreach,
    /// TCP RST
    TcpReset,
}

/// Log options
#[derive(Clone, Debug, Default)]
pub struct LogOptions {
    /// Log prefix
    pub prefix: String,
    /// Log level
    pub level: u8,
    /// Log limit
    pub limit: Option<(u32, u32)>,
}

/// Firewall rule
#[derive(Clone, Debug)]
pub struct Rule {
    /// Rule number
    pub num: u32,
    /// Matches (all must match - AND)
    pub matches: Vec<Match>,
    /// Negate matches
    pub negated: Vec<bool>,
    /// Target action
    pub target: Target,
    /// Packet counter
    pub packets: AtomicU64,
    /// Byte counter
    pub bytes: AtomicU64,
    /// Comment
    pub comment: Option<String>,
}

impl Rule {
    pub fn new(matches: Vec<Match>, target: Target) -> Self {
        let negated = vec![false; matches.len()];
        Self {
            num: 0,
            matches,
            negated,
            target,
            packets: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
            comment: None,
        }
    }

    /// Check if packet matches this rule
    pub fn matches(&self, pkt: &Packet) -> bool {
        for (i, m) in self.matches.iter().enumerate() {
            let matched = match m {
                Match::SrcIp(addr, mask) => (pkt.src_ip & mask) == (addr & mask),
                Match::DstIp(addr, mask) => (pkt.dst_ip & mask) == (addr & mask),
                Match::SrcPort(min, max) => pkt.src_port >= *min && pkt.src_port <= *max,
                Match::DstPort(min, max) => pkt.dst_port >= *min && pkt.dst_port <= *max,
                Match::Protocol(proto) => pkt.protocol == *proto,
                Match::InInterface(iface) => pkt.in_iface.as_ref() == Some(iface),
                Match::OutInterface(iface) => pkt.out_iface.as_ref() == Some(iface),
                Match::TcpFlags(mask, expected) => (pkt.tcp_flags & mask) == *expected,
                Match::IcmpType(t) => pkt.icmp_type == Some(*t),
                Match::State(state) => pkt.conn_state == Some(*state),
                Match::MacSrc(mac) => pkt.src_mac.as_ref() == Some(mac),
                Match::Mark(val, mask) => (pkt.mark & mask) == (val & mask),
                Match::Limit(rate, _burst) => {
                    // Simplified rate limiting
                    true // Would need token bucket implementation
                }
                Match::OwnerUid(uid) => pkt.owner_uid == Some(*uid),
                Match::OwnerGid(gid) => pkt.owner_gid == Some(*gid),
            };

            let result = if self.negated.get(i).copied().unwrap_or(false) {
                !matched
            } else {
                matched
            };

            if !result {
                return false;
            }
        }
        true
    }

    /// Apply target to packet
    pub fn apply(&self, pkt: &mut Packet) -> Verdict {
        // Update counters
        self.packets.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(pkt.len as u64, Ordering::Relaxed);

        match &self.target {
            Target::Accept => Verdict::Accept,
            Target::Drop => Verdict::Drop,
            Target::Reject(_) => Verdict::Reject,
            Target::Log(opts) => {
                log_packet(pkt, opts);
                Verdict::Continue
            }
            Target::Snat(addr, port) => {
                pkt.nat_addr = Some(*addr);
                pkt.nat_port = *port;
                Verdict::Accept
            }
            Target::Dnat(addr, port) => {
                pkt.nat_addr = Some(*addr);
                pkt.nat_port = *port;
                Verdict::Accept
            }
            Target::Masquerade => {
                // Would use outgoing interface's IP
                Verdict::Accept
            }
            Target::Redirect(port) => {
                pkt.dst_port = *port;
                Verdict::Accept
            }
            Target::Mark(mark) => {
                pkt.mark = *mark;
                Verdict::Continue
            }
            Target::Dscp(dscp) => {
                pkt.dscp = *dscp;
                Verdict::Continue
            }
            Target::Ttl(ttl) => {
                pkt.ttl = *ttl;
                Verdict::Continue
            }
            Target::Jump(_) => Verdict::Continue, // Handled by chain
            Target::Return => Verdict::Return,
            Target::Goto(_) => Verdict::Continue, // Handled by chain
        }
    }
}

/// Chain of rules
#[derive(Clone)]
pub struct Chain {
    /// Chain name
    pub name: String,
    /// Rules in order
    pub rules: Vec<Rule>,
    /// Default policy (for built-in chains)
    pub policy: Option<Verdict>,
    /// Packet counter
    pub packets: AtomicU64,
    /// Byte counter
    pub bytes: AtomicU64,
}

impl Chain {
    pub fn new(name: &str, policy: Option<Verdict>) -> Self {
        Self {
            name: String::from(name),
            rules: Vec::new(),
            policy,
            packets: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
        }
    }

    /// Process packet through chain
    pub fn process(&self, pkt: &mut Packet, chains: &BTreeMap<String, Chain>) -> Verdict {
        self.packets.fetch_add(1, Ordering::Relaxed);
        self.bytes.fetch_add(pkt.len as u64, Ordering::Relaxed);

        for rule in &self.rules {
            if rule.matches(pkt) {
                match rule.apply(pkt) {
                    Verdict::Accept => return Verdict::Accept,
                    Verdict::Drop => return Verdict::Drop,
                    Verdict::Reject => return Verdict::Reject,
                    Verdict::Return => break,
                    Verdict::Continue => continue,
                    Verdict::Jump(idx) => {
                        // Would need to look up chain by index
                    }
                    Verdict::Queue => return Verdict::Queue,
                }

                // Handle jumps
                if let Target::Jump(chain_name) = &rule.target {
                    if let Some(target_chain) = chains.get(chain_name) {
                        let result = target_chain.process(pkt, chains);
                        if result != Verdict::Return {
                            return result;
                        }
                    }
                }

                if let Target::Goto(chain_name) = &rule.target {
                    if let Some(target_chain) = chains.get(chain_name) {
                        return target_chain.process(pkt, chains);
                    }
                }
            }
        }

        // Return default policy
        self.policy.unwrap_or(Verdict::Accept)
    }

    /// Add rule
    pub fn add_rule(&mut self, mut rule: Rule) {
        rule.num = self.rules.len() as u32 + 1;
        self.rules.push(rule);
    }

    /// Insert rule at position
    pub fn insert_rule(&mut self, pos: usize, mut rule: Rule) {
        rule.num = pos as u32 + 1;
        self.rules.insert(pos, rule);
        // Renumber
        for (i, r) in self.rules.iter_mut().enumerate() {
            r.num = i as u32 + 1;
        }
    }

    /// Delete rule by number
    pub fn delete_rule(&mut self, num: u32) -> Option<Rule> {
        let pos = self.rules.iter().position(|r| r.num == num)?;
        let rule = self.rules.remove(pos);
        // Renumber
        for (i, r) in self.rules.iter_mut().enumerate() {
            r.num = i as u32 + 1;
        }
        Some(rule)
    }

    /// Flush all rules
    pub fn flush(&mut self) {
        self.rules.clear();
    }
}

/// Packet representation for filtering
#[derive(Clone, Debug)]
pub struct Packet {
    /// Source IP
    pub src_ip: u32,
    /// Destination IP
    pub dst_ip: u32,
    /// Source port
    pub src_port: u16,
    /// Destination port
    pub dst_port: u16,
    /// Protocol (TCP=6, UDP=17, ICMP=1)
    pub protocol: u8,
    /// Input interface
    pub in_iface: Option<String>,
    /// Output interface
    pub out_iface: Option<String>,
    /// TCP flags
    pub tcp_flags: u8,
    /// ICMP type
    pub icmp_type: Option<u8>,
    /// Connection state
    pub conn_state: Option<ConnState>,
    /// Source MAC
    pub src_mac: Option<[u8; 6]>,
    /// Packet mark
    pub mark: u32,
    /// DSCP value
    pub dscp: u8,
    /// TTL
    pub ttl: u8,
    /// Packet length
    pub len: u32,
    /// Owner UID (for OUTPUT)
    pub owner_uid: Option<u32>,
    /// Owner GID
    pub owner_gid: Option<u32>,
    /// NAT address
    pub nat_addr: Option<u32>,
    /// NAT port
    pub nat_port: Option<u16>,
}

impl Packet {
    pub fn new() -> Self {
        Self {
            src_ip: 0,
            dst_ip: 0,
            src_port: 0,
            dst_port: 0,
            protocol: 0,
            in_iface: None,
            out_iface: None,
            tcp_flags: 0,
            icmp_type: None,
            conn_state: None,
            src_mac: None,
            mark: 0,
            dscp: 0,
            ttl: 64,
            len: 0,
            owner_uid: None,
            owner_gid: None,
            nat_addr: None,
            nat_port: None,
        }
    }
}

/// Log a packet
fn log_packet(pkt: &Packet, opts: &LogOptions) {
    let src = format_ip(pkt.src_ip);
    let dst = format_ip(pkt.dst_ip);

    crate::kinfo!(
        "{}IN={} OUT={} SRC={} DST={} LEN={} PROTO={} SPT={} DPT={}",
        opts.prefix,
        pkt.in_iface.as_deref().unwrap_or(""),
        pkt.out_iface.as_deref().unwrap_or(""),
        src, dst, pkt.len, pkt.protocol, pkt.src_port, pkt.dst_port
    );
}

fn format_ip(ip: u32) -> String {
    alloc::format!(
        "{}.{}.{}.{}",
        (ip >> 24) & 0xFF,
        (ip >> 16) & 0xFF,
        (ip >> 8) & 0xFF,
        ip & 0xFF
    )
}

// ============================================================================
// Netfilter State
// ============================================================================

/// Netfilter state per table
struct NetfilterTable {
    chains: BTreeMap<String, Chain>,
}

impl NetfilterTable {
    fn new(table: Table) -> Self {
        let mut chains = BTreeMap::new();

        // Create built-in chains for this table
        for hook in table.valid_hooks() {
            chains.insert(
                String::from(hook.name()),
                Chain::new(hook.name(), Some(Verdict::Accept)),
            );
        }

        Self { chains }
    }
}

/// Global netfilter state
static TABLES: RwLock<BTreeMap<Table, NetfilterTable>> = RwLock::new(BTreeMap::new());

/// Initialize netfilter
pub fn init() {
    let mut tables = TABLES.write();

    tables.insert(Table::Filter, NetfilterTable::new(Table::Filter));
    tables.insert(Table::Nat, NetfilterTable::new(Table::Nat));
    tables.insert(Table::Mangle, NetfilterTable::new(Table::Mangle));
    tables.insert(Table::Raw, NetfilterTable::new(Table::Raw));

    crate::kprintln!("  Netfilter initialized (filter, nat, mangle, raw tables)");
}

/// Add rule to chain
pub fn add_rule(table: Table, chain: &str, rule: Rule) -> Result<(), &'static str> {
    let mut tables = TABLES.write();
    let table = tables.get_mut(&table).ok_or("Table not found")?;
    let chain = table.chains.get_mut(chain).ok_or("Chain not found")?;
    chain.add_rule(rule);
    Ok(())
}

/// Delete rule from chain
pub fn delete_rule(table: Table, chain: &str, num: u32) -> Result<(), &'static str> {
    let mut tables = TABLES.write();
    let table = tables.get_mut(&table).ok_or("Table not found")?;
    let chain = table.chains.get_mut(chain).ok_or("Chain not found")?;
    chain.delete_rule(num).ok_or("Rule not found")?;
    Ok(())
}

/// Create user-defined chain
pub fn create_chain(table: Table, name: &str) -> Result<(), &'static str> {
    let mut tables = TABLES.write();
    let table = tables.get_mut(&table).ok_or("Table not found")?;

    if table.chains.contains_key(name) {
        return Err("Chain already exists");
    }

    table.chains.insert(String::from(name), Chain::new(name, None));
    Ok(())
}

/// Delete user-defined chain
pub fn delete_chain(table: Table, name: &str) -> Result<(), &'static str> {
    let mut tables = TABLES.write();
    let table = tables.get_mut(&table).ok_or("Table not found")?;

    let chain = table.chains.get(name).ok_or("Chain not found")?;
    if chain.policy.is_some() {
        return Err("Cannot delete built-in chain");
    }
    if !chain.rules.is_empty() {
        return Err("Chain not empty");
    }

    table.chains.remove(name);
    Ok(())
}

/// Set chain policy
pub fn set_policy(table: Table, chain: &str, policy: Verdict) -> Result<(), &'static str> {
    let mut tables = TABLES.write();
    let table = tables.get_mut(&table).ok_or("Table not found")?;
    let chain = table.chains.get_mut(chain).ok_or("Chain not found")?;

    if chain.policy.is_none() {
        return Err("User-defined chains have no policy");
    }

    chain.policy = Some(policy);
    Ok(())
}

/// Flush chain
pub fn flush_chain(table: Table, chain: &str) -> Result<(), &'static str> {
    let mut tables = TABLES.write();
    let table = tables.get_mut(&table).ok_or("Table not found")?;
    let chain = table.chains.get_mut(chain).ok_or("Chain not found")?;
    chain.flush();
    Ok(())
}

/// Process packet through hook
pub fn process_hook(hook: Hook, pkt: &mut Packet) -> Verdict {
    let tables = TABLES.read();

    // Process through each table in order
    for table_type in &[Table::Raw, Table::Mangle, Table::Nat, Table::Filter, Table::Security] {
        if !table_type.valid_hooks().contains(&hook) {
            continue;
        }

        if let Some(table) = tables.get(table_type) {
            if let Some(chain) = table.chains.get(hook.name()) {
                match chain.process(pkt, &table.chains) {
                    Verdict::Accept => continue,
                    v => return v,
                }
            }
        }
    }

    Verdict::Accept
}

/// List rules in chain
pub fn list_rules(table: Table, chain: &str) -> Option<Vec<(u32, String)>> {
    let tables = TABLES.read();
    let table = tables.get(&table)?;
    let chain = table.chains.get(chain)?;

    let rules: Vec<(u32, String)> = chain.rules.iter()
        .map(|r| {
            let target = match &r.target {
                Target::Accept => "ACCEPT",
                Target::Drop => "DROP",
                Target::Reject(_) => "REJECT",
                Target::Log(_) => "LOG",
                _ => "???",
            };
            (r.num, alloc::format!("{} -> {}", "...", target))
        })
        .collect();

    Some(rules)
}

/// Generate iptables-save format
pub fn generate_iptables_save() -> String {
    let mut output = String::new();
    let tables = TABLES.read();

    for (table_type, table) in tables.iter() {
        output.push_str(&alloc::format!("*{}\n", table_type.name()));

        // Chain definitions
        for (name, chain) in &table.chains {
            let policy = chain.policy
                .map(|v| match v {
                    Verdict::Accept => "ACCEPT",
                    Verdict::Drop => "DROP",
                    _ => "-",
                })
                .unwrap_or("-");

            output.push_str(&alloc::format!(
                ":{} {} [{:}:{:}]\n",
                name, policy,
                chain.packets.load(Ordering::Relaxed),
                chain.bytes.load(Ordering::Relaxed)
            ));
        }

        output.push_str("COMMIT\n");
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rule_matching() {
        let rule = Rule::new(
            vec![
                Match::SrcIp(0xC0A80100, 0xFFFFFF00), // 192.168.1.0/24
                Match::Protocol(6), // TCP
                Match::DstPort(80, 80),
            ],
            Target::Accept,
        );

        let mut pkt = Packet::new();
        pkt.src_ip = 0xC0A80105; // 192.168.1.5
        pkt.protocol = 6;
        pkt.dst_port = 80;

        assert!(rule.matches(&pkt));

        pkt.src_ip = 0xC0A80205; // 192.168.2.5
        assert!(!rule.matches(&pkt));
    }

    #[test]
    fn test_chain_processing() {
        let mut chain = Chain::new("INPUT", Some(Verdict::Drop));

        chain.add_rule(Rule::new(
            vec![Match::Protocol(6)],
            Target::Accept,
        ));

        let mut pkt = Packet::new();
        pkt.protocol = 6;

        let chains = BTreeMap::new();
        assert_eq!(chain.process(&mut pkt, &chains), Verdict::Accept);

        pkt.protocol = 17;
        assert_eq!(chain.process(&mut pkt, &chains), Verdict::Drop);
    }
}
