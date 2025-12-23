//! Minimal Web Browser
//!
//! A lightweight web browser for HubLab IO with basic HTML/CSS rendering.
//! Focuses on text content and essential web features.

use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

/// Browser configuration
#[derive(Clone, Debug)]
pub struct BrowserConfig {
    /// User agent string
    pub user_agent: String,
    /// Enable JavaScript (limited)
    pub javascript_enabled: bool,
    /// Enable images
    pub images_enabled: bool,
    /// Maximum page size (bytes)
    pub max_page_size: usize,
    /// Connection timeout (ms)
    pub timeout_ms: u32,
    /// Enable cookies
    pub cookies_enabled: bool,
    /// Home page URL
    pub home_url: String,
}

impl Default for BrowserConfig {
    fn default() -> Self {
        Self {
            user_agent: String::from("HubLabIO/1.0 (ARM64; AI-Native OS)"),
            javascript_enabled: false, // Start simple
            images_enabled: true,
            max_page_size: 5 * 1024 * 1024, // 5MB
            timeout_ms: 30000,
            cookies_enabled: true,
            home_url: String::from("about:blank"),
        }
    }
}

/// URL parser
#[derive(Clone, Debug)]
pub struct Url {
    /// Protocol (http, https, file, about)
    pub scheme: String,
    /// Host name
    pub host: String,
    /// Port number
    pub port: u16,
    /// Path
    pub path: String,
    /// Query string
    pub query: Option<String>,
    /// Fragment
    pub fragment: Option<String>,
}

impl Url {
    /// Parse URL string
    pub fn parse(url: &str) -> Option<Self> {
        let url = url.trim();

        // Handle special URLs
        if url.starts_with("about:") {
            return Some(Self {
                scheme: String::from("about"),
                host: String::new(),
                port: 0,
                path: url[6..].to_string(),
                query: None,
                fragment: None,
            });
        }

        // Split scheme
        let (scheme, rest) = if let Some(idx) = url.find("://") {
            (url[..idx].to_string(), &url[idx + 3..])
        } else {
            // Default to https
            (String::from("https"), url)
        };

        // Extract fragment
        let (rest, fragment) = if let Some(idx) = rest.find('#') {
            (&rest[..idx], Some(rest[idx + 1..].to_string()))
        } else {
            (rest, None)
        };

        // Extract query
        let (rest, query) = if let Some(idx) = rest.find('?') {
            (&rest[..idx], Some(rest[idx + 1..].to_string()))
        } else {
            (rest, None)
        };

        // Split host and path
        let (host_port, path) = if let Some(idx) = rest.find('/') {
            (&rest[..idx], rest[idx..].to_string())
        } else {
            (rest, String::from("/"))
        };

        // Extract port
        let (host, port) = if let Some(idx) = host_port.find(':') {
            let port_str = &host_port[idx + 1..];
            let port = port_str
                .parse()
                .unwrap_or(if scheme == "https" { 443 } else { 80 });
            (host_port[..idx].to_string(), port)
        } else {
            (
                host_port.to_string(),
                if scheme == "https" { 443 } else { 80 },
            )
        };

        Some(Self {
            scheme,
            host,
            port,
            path,
            query,
            fragment,
        })
    }

    /// Convert back to string
    pub fn to_string(&self) -> String {
        if self.scheme == "about" {
            return format!("about:{}", self.path);
        }

        let mut url = format!("{}://{}", self.scheme, self.host);

        let default_port = if self.scheme == "https" { 443 } else { 80 };
        if self.port != default_port {
            url.push_str(&format!(":{}", self.port));
        }

        url.push_str(&self.path);

        if let Some(ref q) = self.query {
            url.push('?');
            url.push_str(q);
        }

        if let Some(ref f) = self.fragment {
            url.push('#');
            url.push_str(f);
        }

        url
    }
}

/// HTTP request
#[derive(Clone, Debug)]
pub struct HttpRequest {
    /// Method (GET, POST, etc.)
    pub method: String,
    /// URL
    pub url: Url,
    /// Headers
    pub headers: BTreeMap<String, String>,
    /// Body (for POST)
    pub body: Option<Vec<u8>>,
}

impl HttpRequest {
    /// Create GET request
    pub fn get(url: Url) -> Self {
        Self {
            method: String::from("GET"),
            url,
            headers: BTreeMap::new(),
            body: None,
        }
    }

    /// Create POST request
    pub fn post(url: Url, body: Vec<u8>) -> Self {
        Self {
            method: String::from("POST"),
            url,
            headers: BTreeMap::new(),
            body: Some(body),
        }
    }

    /// Add header
    pub fn header(mut self, key: &str, value: &str) -> Self {
        self.headers.insert(key.to_string(), value.to_string());
        self
    }

    /// Serialize to bytes
    pub fn to_bytes(&self, config: &BrowserConfig) -> Vec<u8> {
        let mut request = format!(
            "{} {} HTTP/1.1\r\nHost: {}\r\nUser-Agent: {}\r\nConnection: close\r\n",
            self.method, self.url.path, self.url.host, config.user_agent
        );

        for (key, value) in &self.headers {
            request.push_str(&format!("{}: {}\r\n", key, value));
        }

        if let Some(ref body) = self.body {
            request.push_str(&format!("Content-Length: {}\r\n", body.len()));
        }

        request.push_str("\r\n");

        let mut bytes = request.into_bytes();
        if let Some(ref body) = self.body {
            bytes.extend_from_slice(body);
        }

        bytes
    }
}

/// HTTP response
#[derive(Clone, Debug)]
pub struct HttpResponse {
    /// Status code
    pub status: u16,
    /// Status text
    pub status_text: String,
    /// Headers
    pub headers: BTreeMap<String, String>,
    /// Body
    pub body: Vec<u8>,
}

impl HttpResponse {
    /// Parse HTTP response
    pub fn parse(data: &[u8]) -> Option<Self> {
        let text = core::str::from_utf8(data).ok()?;

        // Find header/body separator
        let header_end = text.find("\r\n\r\n")?;
        let header_text = &text[..header_end];
        let body_start = header_end + 4;

        // Parse status line
        let mut lines = header_text.lines();
        let status_line = lines.next()?;
        let parts: Vec<&str> = status_line.splitn(3, ' ').collect();
        if parts.len() < 2 {
            return None;
        }

        let status: u16 = parts[1].parse().ok()?;
        let status_text = parts.get(2).unwrap_or(&"").to_string();

        // Parse headers
        let mut headers = BTreeMap::new();
        for line in lines {
            if let Some(idx) = line.find(':') {
                let key = line[..idx].trim().to_lowercase();
                let value = line[idx + 1..].trim().to_string();
                headers.insert(key, value);
            }
        }

        // Extract body
        let body = data[body_start..].to_vec();

        Some(Self {
            status,
            status_text,
            headers,
            body,
        })
    }

    /// Get content type
    pub fn content_type(&self) -> Option<&str> {
        self.headers.get("content-type").map(|s| s.as_str())
    }

    /// Check if HTML
    pub fn is_html(&self) -> bool {
        self.content_type()
            .map(|ct| ct.contains("text/html"))
            .unwrap_or(false)
    }

    /// Get body as string
    pub fn body_string(&self) -> Option<String> {
        String::from_utf8(self.body.clone()).ok()
    }
}

/// HTML DOM Node
#[derive(Clone, Debug)]
pub enum DomNode {
    /// Text content
    Text(String),
    /// Element with tag, attributes, children
    Element {
        tag: String,
        attrs: BTreeMap<String, String>,
        children: Vec<DomNode>,
    },
    /// Comment
    Comment(String),
}

impl DomNode {
    /// Get text content recursively
    pub fn text_content(&self) -> String {
        match self {
            DomNode::Text(text) => text.clone(),
            DomNode::Element { children, .. } => children
                .iter()
                .map(|c| c.text_content())
                .collect::<Vec<_>>()
                .join(""),
            DomNode::Comment(_) => String::new(),
        }
    }

    /// Find elements by tag name
    pub fn find_by_tag(&self, tag_name: &str) -> Vec<&DomNode> {
        let mut results = Vec::new();
        self.find_by_tag_recursive(tag_name, &mut results);
        results
    }

    fn find_by_tag_recursive<'a>(&'a self, tag_name: &str, results: &mut Vec<&'a DomNode>) {
        if let DomNode::Element { tag, children, .. } = self {
            if tag.eq_ignore_ascii_case(tag_name) {
                results.push(self);
            }
            for child in children {
                child.find_by_tag_recursive(tag_name, results);
            }
        }
    }

    /// Get attribute value
    pub fn get_attr(&self, name: &str) -> Option<&str> {
        if let DomNode::Element { attrs, .. } = self {
            attrs.get(name).map(|s| s.as_str())
        } else {
            None
        }
    }
}

/// Simple HTML parser
pub struct HtmlParser;

impl HtmlParser {
    /// Parse HTML string into DOM
    pub fn parse(html: &str) -> DomNode {
        let mut parser = HtmlParserState::new(html);
        parser.parse_document()
    }
}

struct HtmlParserState<'a> {
    input: &'a str,
    pos: usize,
}

impl<'a> HtmlParserState<'a> {
    fn new(input: &'a str) -> Self {
        Self { input, pos: 0 }
    }

    fn parse_document(&mut self) -> DomNode {
        let children = self.parse_nodes();
        DomNode::Element {
            tag: String::from("document"),
            attrs: BTreeMap::new(),
            children,
        }
    }

    fn parse_nodes(&mut self) -> Vec<DomNode> {
        let mut nodes = Vec::new();

        while self.pos < self.input.len() {
            self.skip_whitespace();

            if self.starts_with("</") {
                break; // End tag, return to parent
            } else if self.starts_with("<!--") {
                if let Some(comment) = self.parse_comment() {
                    nodes.push(comment);
                }
            } else if self.starts_with("<") {
                if let Some(element) = self.parse_element() {
                    nodes.push(element);
                }
            } else {
                if let Some(text) = self.parse_text() {
                    nodes.push(text);
                }
            }
        }

        nodes
    }

    fn parse_element(&mut self) -> Option<DomNode> {
        if !self.consume_char('<') {
            return None;
        }

        // Parse tag name
        let tag = self.parse_tag_name()?;

        // Parse attributes
        let attrs = self.parse_attributes();

        // Self-closing or void element
        let self_closing = self.consume_str("/>")
            || matches!(
                tag.to_lowercase().as_str(),
                "br" | "hr"
                    | "img"
                    | "input"
                    | "meta"
                    | "link"
                    | "area"
                    | "base"
                    | "col"
                    | "embed"
                    | "source"
                    | "track"
                    | "wbr"
            );

        if !self_closing && !self.consume_char('>') {
            // Try to recover
            self.skip_until('>');
            self.consume_char('>');
        }

        let children = if self_closing {
            Vec::new()
        } else {
            // Parse children
            let children = self.parse_nodes();

            // Consume end tag
            self.consume_str("</");
            self.parse_tag_name();
            self.consume_char('>');

            children
        };

        Some(DomNode::Element {
            tag,
            attrs,
            children,
        })
    }

    fn parse_tag_name(&mut self) -> Option<String> {
        let start = self.pos;
        while self.pos < self.input.len() {
            let c = self.current_char()?;
            if c.is_alphanumeric() || c == '-' || c == '_' {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos > start {
            Some(self.input[start..self.pos].to_string())
        } else {
            None
        }
    }

    fn parse_attributes(&mut self) -> BTreeMap<String, String> {
        let mut attrs = BTreeMap::new();

        loop {
            self.skip_whitespace();

            if self.current_char() == Some('>') || self.current_char() == Some('/') {
                break;
            }

            if let Some(name) = self.parse_attr_name() {
                self.skip_whitespace();
                let value = if self.consume_char('=') {
                    self.skip_whitespace();
                    self.parse_attr_value()
                } else {
                    String::new()
                };
                attrs.insert(name, value);
            } else {
                break;
            }
        }

        attrs
    }

    fn parse_attr_name(&mut self) -> Option<String> {
        let start = self.pos;
        while self.pos < self.input.len() {
            let c = self.current_char()?;
            if c.is_alphanumeric() || c == '-' || c == '_' || c == ':' {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos > start {
            Some(self.input[start..self.pos].to_lowercase())
        } else {
            None
        }
    }

    fn parse_attr_value(&mut self) -> String {
        let quote = self.current_char();
        if quote == Some('"') || quote == Some('\'') {
            self.pos += 1;
            let start = self.pos;
            while self.pos < self.input.len() && self.current_char() != quote {
                self.pos += 1;
            }
            let value = self.input[start..self.pos].to_string();
            self.consume_char(quote.unwrap());
            value
        } else {
            // Unquoted value
            let start = self.pos;
            while self.pos < self.input.len() {
                let c = self.current_char().unwrap();
                if c.is_whitespace() || c == '>' || c == '/' {
                    break;
                }
                self.pos += 1;
            }
            self.input[start..self.pos].to_string()
        }
    }

    fn parse_text(&mut self) -> Option<DomNode> {
        let start = self.pos;
        while self.pos < self.input.len() && self.current_char() != Some('<') {
            self.pos += 1;
        }
        if self.pos > start {
            let text = self.input[start..self.pos].to_string();
            let text = self.decode_entities(&text);
            if !text.trim().is_empty() {
                Some(DomNode::Text(text))
            } else {
                None
            }
        } else {
            None
        }
    }

    fn parse_comment(&mut self) -> Option<DomNode> {
        if !self.consume_str("<!--") {
            return None;
        }
        let start = self.pos;
        while self.pos < self.input.len() && !self.starts_with("-->") {
            self.pos += 1;
        }
        let content = self.input[start..self.pos].to_string();
        self.consume_str("-->");
        Some(DomNode::Comment(content))
    }

    fn decode_entities(&self, text: &str) -> String {
        text.replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&amp;", "&")
            .replace("&quot;", "\"")
            .replace("&apos;", "'")
            .replace("&nbsp;", " ")
    }

    fn current_char(&self) -> Option<char> {
        self.input[self.pos..].chars().next()
    }

    fn consume_char(&mut self, expected: char) -> bool {
        if self.current_char() == Some(expected) {
            self.pos += 1;
            true
        } else {
            false
        }
    }

    fn consume_str(&mut self, s: &str) -> bool {
        if self.starts_with(s) {
            self.pos += s.len();
            true
        } else {
            false
        }
    }

    fn starts_with(&self, s: &str) -> bool {
        self.input[self.pos..].starts_with(s)
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.input.len() {
            if let Some(c) = self.current_char() {
                if c.is_whitespace() {
                    self.pos += 1;
                } else {
                    break;
                }
            } else {
                break;
            }
        }
    }

    fn skip_until(&mut self, target: char) {
        while self.pos < self.input.len() && self.current_char() != Some(target) {
            self.pos += 1;
        }
    }
}

/// Rendered line for display
#[derive(Clone, Debug)]
pub struct RenderLine {
    /// Text content
    pub text: String,
    /// Is heading
    pub is_heading: bool,
    /// Heading level (1-6)
    pub heading_level: u8,
    /// Is link
    pub is_link: bool,
    /// Link URL (if link)
    pub link_url: Option<String>,
    /// Is list item
    pub is_list_item: bool,
    /// Indentation level
    pub indent: usize,
}

/// HTML renderer (text-based)
pub struct HtmlRenderer {
    lines: Vec<RenderLine>,
    current_indent: usize,
}

impl HtmlRenderer {
    pub fn new() -> Self {
        Self {
            lines: Vec::new(),
            current_indent: 0,
        }
    }

    /// Render DOM to text lines
    pub fn render(&mut self, dom: &DomNode) -> Vec<RenderLine> {
        self.lines.clear();
        self.render_node(dom);
        self.lines.clone()
    }

    fn render_node(&mut self, node: &DomNode) {
        match node {
            DomNode::Text(text) => {
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    self.add_line(trimmed, false, 0, false, None, false);
                }
            }
            DomNode::Element {
                tag,
                attrs,
                children,
            } => {
                let tag_lower = tag.to_lowercase();

                match tag_lower.as_str() {
                    // Skip non-visual elements
                    "script" | "style" | "head" | "meta" | "link" => {}

                    // Headings
                    "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                        let level = tag_lower.chars().last().unwrap().to_digit(10).unwrap() as u8;
                        let text = self.collect_text(children);
                        self.add_line(&text, true, level, false, None, false);
                        self.add_empty_line();
                    }

                    // Paragraphs
                    "p" => {
                        self.render_children(children);
                        self.add_empty_line();
                    }

                    // Line breaks
                    "br" => {
                        self.add_empty_line();
                    }

                    // Horizontal rule
                    "hr" => {
                        self.add_line("─".repeat(40).as_str(), false, 0, false, None, false);
                    }

                    // Links
                    "a" => {
                        let href = attrs.get("href").cloned();
                        let text = self.collect_text(children);
                        self.add_line(&text, false, 0, true, href, false);
                    }

                    // Lists
                    "ul" | "ol" => {
                        self.current_indent += 2;
                        self.render_children(children);
                        self.current_indent -= 2;
                    }

                    "li" => {
                        let text = self.collect_text(children);
                        self.add_line(&format!("• {}", text), false, 0, false, None, true);
                    }

                    // Code blocks
                    "pre" | "code" => {
                        let text = self.collect_text(children);
                        for line in text.lines() {
                            self.add_line(&format!("  {}", line), false, 0, false, None, false);
                        }
                    }

                    // Blockquote
                    "blockquote" => {
                        self.current_indent += 2;
                        let text = self.collect_text(children);
                        for line in text.lines() {
                            self.add_line(&format!("│ {}", line), false, 0, false, None, false);
                        }
                        self.current_indent -= 2;
                    }

                    // Bold/Strong
                    "b" | "strong" => {
                        let text = self.collect_text(children);
                        self.add_line(&format!("**{}**", text), false, 0, false, None, false);
                    }

                    // Italic/Emphasis
                    "i" | "em" => {
                        let text = self.collect_text(children);
                        self.add_line(&format!("_{}_", text), false, 0, false, None, false);
                    }

                    // Images (show alt text)
                    "img" => {
                        let alt = attrs.get("alt").map(|s| s.as_str()).unwrap_or("[image]");
                        self.add_line(&format!("[IMG: {}]", alt), false, 0, false, None, false);
                    }

                    // Tables (basic support)
                    "table" => {
                        self.add_line("┌─ Table ─┐", false, 0, false, None, false);
                        self.render_children(children);
                        self.add_line("└──────────┘", false, 0, false, None, false);
                    }

                    "tr" => {
                        let cells: Vec<String> = children
                            .iter()
                            .filter_map(|c| {
                                if let DomNode::Element { tag, children, .. } = c {
                                    if tag == "td" || tag == "th" {
                                        return Some(self.collect_text(children));
                                    }
                                }
                                None
                            })
                            .collect();
                        self.add_line(
                            &format!("│ {} │", cells.join(" │ ")),
                            false,
                            0,
                            false,
                            None,
                            false,
                        );
                    }

                    // Forms (show input placeholders)
                    "input" => {
                        let placeholder =
                            attrs.get("placeholder").map(|s| s.as_str()).unwrap_or("");
                        let input_type = attrs.get("type").map(|s| s.as_str()).unwrap_or("text");
                        self.add_line(
                            &format!("[{}: {}]", input_type, placeholder),
                            false,
                            0,
                            false,
                            None,
                            false,
                        );
                    }

                    "button" => {
                        let text = self.collect_text(children);
                        self.add_line(&format!("[{}]", text), false, 0, false, None, false);
                    }

                    // Default: render children
                    _ => {
                        self.render_children(children);
                    }
                }
            }
            DomNode::Comment(_) => {} // Skip comments
        }
    }

    fn render_children(&mut self, children: &[DomNode]) {
        for child in children {
            self.render_node(child);
        }
    }

    fn collect_text(&self, nodes: &[DomNode]) -> String {
        nodes
            .iter()
            .map(|n| n.text_content())
            .collect::<Vec<_>>()
            .join(" ")
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ")
    }

    fn add_line(
        &mut self,
        text: &str,
        is_heading: bool,
        heading_level: u8,
        is_link: bool,
        link_url: Option<String>,
        is_list_item: bool,
    ) {
        let indent_str = " ".repeat(self.current_indent);
        self.lines.push(RenderLine {
            text: format!("{}{}", indent_str, text),
            is_heading,
            heading_level,
            is_link,
            link_url,
            is_list_item,
            indent: self.current_indent,
        });
    }

    fn add_empty_line(&mut self) {
        if !self.lines.is_empty() && !self.lines.last().unwrap().text.is_empty() {
            self.lines.push(RenderLine {
                text: String::new(),
                is_heading: false,
                heading_level: 0,
                is_link: false,
                link_url: None,
                is_list_item: false,
                indent: 0,
            });
        }
    }
}

/// Tab in browser
#[derive(Clone, Debug)]
pub struct Tab {
    /// Tab ID
    pub id: usize,
    /// Current URL
    pub url: Option<Url>,
    /// Page title
    pub title: String,
    /// Rendered content
    pub content: Vec<RenderLine>,
    /// Scroll position
    pub scroll_y: usize,
    /// Loading state
    pub loading: bool,
    /// History (for back/forward)
    pub history: Vec<Url>,
    /// History position
    pub history_pos: usize,
}

impl Tab {
    pub fn new(id: usize) -> Self {
        Self {
            id,
            url: None,
            title: String::from("New Tab"),
            content: Vec::new(),
            scroll_y: 0,
            loading: false,
            history: Vec::new(),
            history_pos: 0,
        }
    }

    /// Check if can go back
    pub fn can_go_back(&self) -> bool {
        self.history_pos > 0
    }

    /// Check if can go forward
    pub fn can_go_forward(&self) -> bool {
        self.history_pos < self.history.len().saturating_sub(1)
    }
}

/// Browser state
pub struct Browser {
    /// Configuration
    pub config: BrowserConfig,
    /// Tabs
    pub tabs: Vec<Tab>,
    /// Active tab index
    pub active_tab: usize,
    /// Tab counter
    tab_counter: usize,
    /// Cookies
    cookies: BTreeMap<String, String>,
    /// Bookmarks
    bookmarks: Vec<Bookmark>,
}

/// Bookmark
#[derive(Clone, Debug)]
pub struct Bookmark {
    /// Bookmark title
    pub title: String,
    /// URL
    pub url: String,
}

impl Browser {
    /// Create new browser
    pub fn new(config: BrowserConfig) -> Self {
        let mut browser = Self {
            config,
            tabs: Vec::new(),
            active_tab: 0,
            tab_counter: 0,
            cookies: BTreeMap::new(),
            bookmarks: Vec::new(),
        };

        // Create initial tab
        browser.new_tab();

        browser
    }

    /// Create new tab
    pub fn new_tab(&mut self) -> usize {
        let id = self.tab_counter;
        self.tab_counter += 1;
        self.tabs.push(Tab::new(id));
        self.active_tab = self.tabs.len() - 1;
        id
    }

    /// Close tab
    pub fn close_tab(&mut self, index: usize) {
        if self.tabs.len() > 1 && index < self.tabs.len() {
            self.tabs.remove(index);
            if self.active_tab >= self.tabs.len() {
                self.active_tab = self.tabs.len() - 1;
            }
        }
    }

    /// Get active tab
    pub fn active_tab(&self) -> Option<&Tab> {
        self.tabs.get(self.active_tab)
    }

    /// Get active tab mut
    pub fn active_tab_mut(&mut self) -> Option<&mut Tab> {
        self.tabs.get_mut(self.active_tab)
    }

    /// Navigate to URL
    pub fn navigate(&mut self, url_str: &str) {
        let url = match Url::parse(url_str) {
            Some(u) => u,
            None => return,
        };

        if let Some(tab) = self.active_tab_mut() {
            tab.loading = true;

            // Handle special URLs
            if url.scheme == "about" {
                let content = match url.path.as_str() {
                    "blank" => vec![],
                    "version" => vec![
                        RenderLine {
                            text: String::from("HubLab IO Browser v1.0"),
                            is_heading: true,
                            heading_level: 1,
                            is_link: false,
                            link_url: None,
                            is_list_item: false,
                            indent: 0,
                        },
                        RenderLine {
                            text: String::from("A minimal web browser for AI-native OS"),
                            is_heading: false,
                            heading_level: 0,
                            is_link: false,
                            link_url: None,
                            is_list_item: false,
                            indent: 0,
                        },
                    ],
                    _ => vec![RenderLine {
                        text: format!("Unknown page: about:{}", url.path),
                        is_heading: false,
                        heading_level: 0,
                        is_link: false,
                        link_url: None,
                        is_list_item: false,
                        indent: 0,
                    }],
                };

                tab.content = content;
                tab.title = format!("about:{}", url.path);
                tab.url = Some(url.clone());
                tab.loading = false;

                // Add to history
                tab.history.truncate(tab.history_pos + 1);
                tab.history.push(url);
                tab.history_pos = tab.history.len() - 1;
            }
            // TODO: Implement actual HTTP fetching
            // For now, mark as loaded with placeholder
        }
    }

    /// Go back in history
    pub fn go_back(&mut self) {
        if let Some(tab) = self.active_tab_mut() {
            if tab.can_go_back() {
                tab.history_pos -= 1;
                if let Some(url) = tab.history.get(tab.history_pos).cloned() {
                    // Navigate without adding to history
                    tab.url = Some(url);
                }
            }
        }
    }

    /// Go forward in history
    pub fn go_forward(&mut self) {
        if let Some(tab) = self.active_tab_mut() {
            if tab.can_go_forward() {
                tab.history_pos += 1;
                if let Some(url) = tab.history.get(tab.history_pos).cloned() {
                    tab.url = Some(url);
                }
            }
        }
    }

    /// Refresh current page
    pub fn refresh(&mut self) {
        if let Some(tab) = self.active_tab() {
            if let Some(url) = tab.url.clone() {
                self.navigate(&url.to_string());
            }
        }
    }

    /// Add bookmark
    pub fn add_bookmark(&mut self, title: &str, url: &str) {
        self.bookmarks.push(Bookmark {
            title: title.to_string(),
            url: url.to_string(),
        });
    }

    /// Get bookmarks
    pub fn bookmarks(&self) -> &[Bookmark] {
        &self.bookmarks
    }

    /// Scroll down
    pub fn scroll_down(&mut self, lines: usize) {
        if let Some(tab) = self.active_tab_mut() {
            tab.scroll_y = tab.scroll_y.saturating_add(lines);
            let max_scroll = tab.content.len().saturating_sub(20); // Assume 20 visible lines
            tab.scroll_y = tab.scroll_y.min(max_scroll);
        }
    }

    /// Scroll up
    pub fn scroll_up(&mut self, lines: usize) {
        if let Some(tab) = self.active_tab_mut() {
            tab.scroll_y = tab.scroll_y.saturating_sub(lines);
        }
    }

    /// Set cookie
    pub fn set_cookie(&mut self, domain: &str, name: &str, value: &str) {
        let key = format!("{}:{}", domain, name);
        self.cookies.insert(key, value.to_string());
    }

    /// Get cookie
    pub fn get_cookie(&self, domain: &str, name: &str) -> Option<&str> {
        let key = format!("{}:{}", domain, name);
        self.cookies.get(&key).map(|s| s.as_str())
    }

    /// Load HTML content directly (for testing)
    pub fn load_html(&mut self, html: &str, title: &str) {
        let dom = HtmlParser::parse(html);
        let mut renderer = HtmlRenderer::new();
        let content = renderer.render(&dom);

        if let Some(tab) = self.active_tab_mut() {
            tab.content = content;
            tab.title = title.to_string();
            tab.scroll_y = 0;
            tab.loading = false;
        }
    }
}

impl Default for Browser {
    fn default() -> Self {
        Self::new(BrowserConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_url_parse() {
        let url = Url::parse("https://example.com/path?query=1#section").unwrap();
        assert_eq!(url.scheme, "https");
        assert_eq!(url.host, "example.com");
        assert_eq!(url.port, 443);
        assert_eq!(url.path, "/path");
        assert_eq!(url.query, Some("query=1".to_string()));
        assert_eq!(url.fragment, Some("section".to_string()));
    }

    #[test]
    fn test_url_parse_about() {
        let url = Url::parse("about:blank").unwrap();
        assert_eq!(url.scheme, "about");
        assert_eq!(url.path, "blank");
    }

    #[test]
    fn test_html_parser() {
        let html = "<html><body><h1>Hello</h1><p>World</p></body></html>";
        let dom = HtmlParser::parse(html);

        let headings = dom.find_by_tag("h1");
        assert_eq!(headings.len(), 1);
        assert_eq!(headings[0].text_content(), "Hello");
    }

    #[test]
    fn test_browser_tabs() {
        let mut browser = Browser::default();
        assert_eq!(browser.tabs.len(), 1);

        browser.new_tab();
        assert_eq!(browser.tabs.len(), 2);
        assert_eq!(browser.active_tab, 1);

        browser.close_tab(1);
        assert_eq!(browser.tabs.len(), 1);
    }

    #[test]
    fn test_html_renderer() {
        let html = "<h1>Title</h1><p>Paragraph text</p><a href='test.html'>Link</a>";
        let dom = HtmlParser::parse(html);
        let mut renderer = HtmlRenderer::new();
        let lines = renderer.render(&dom);

        assert!(lines.iter().any(|l| l.is_heading));
        assert!(lines.iter().any(|l| l.is_link));
    }
}
