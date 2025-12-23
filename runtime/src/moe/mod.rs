//! Mixture of Real Experts (MoE-R)
//!
//! Enables multiple specialized models to collaborate on tasks.
//! Based on Jupiter's MoE-R swarm system.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;
use alloc::sync::Arc;
use spin::RwLock;

/// Expert identifier
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExpertId(pub u64);

/// Expert domain/specialization
#[derive(Clone, Debug)]
pub struct ExpertDomain {
    /// Domain name (e.g., "python", "react", "api-design")
    pub name: String,
    /// Keywords for routing
    pub keywords: Vec<String>,
    /// Confidence threshold for activation
    pub threshold: f32,
}

/// Expert configuration
#[derive(Clone, Debug)]
pub struct ExpertConfig {
    /// Unique ID
    pub id: ExpertId,
    /// Expert name
    pub name: String,
    /// Description
    pub description: String,
    /// Domain/specialization
    pub domain: ExpertDomain,
    /// Model path
    pub model_path: String,
    /// Node ID (for distributed)
    pub node_id: Option<super::distributed::NodeId>,
    /// Priority (higher = preferred)
    pub priority: u8,
}

/// Expert state
#[derive(Clone, Debug)]
pub struct ExpertState {
    pub config: ExpertConfig,
    pub loaded: bool,
    pub busy: bool,
    pub total_queries: u64,
    pub avg_latency_ms: f32,
}

/// Router for selecting experts
pub struct Router {
    /// All registered experts
    experts: BTreeMap<ExpertId, ExpertState>,
    /// Keyword to expert mapping
    keyword_index: BTreeMap<String, Vec<ExpertId>>,
    /// Routing strategy
    strategy: RoutingStrategy,
}

/// Routing strategies
#[derive(Clone, Copy, Debug)]
pub enum RoutingStrategy {
    /// Select top-k most relevant experts
    TopK(usize),
    /// Select all experts above threshold
    Threshold(f32),
    /// Route to single best expert
    Single,
    /// Use all available experts
    All,
}

impl Router {
    /// Create a new router
    pub fn new(strategy: RoutingStrategy) -> Self {
        Self {
            experts: BTreeMap::new(),
            keyword_index: BTreeMap::new(),
            strategy,
        }
    }

    /// Register an expert
    pub fn register(&mut self, config: ExpertConfig) {
        let id = config.id;

        // Index keywords
        for keyword in &config.domain.keywords {
            self.keyword_index
                .entry(keyword.to_lowercase())
                .or_default()
                .push(id);
        }

        self.experts.insert(id, ExpertState {
            config,
            loaded: false,
            busy: false,
            total_queries: 0,
            avg_latency_ms: 0.0,
        });
    }

    /// Route a query to relevant experts
    pub fn route(&self, query: &str) -> Vec<ExpertId> {
        let query_lower = query.to_lowercase();
        let mut scores: BTreeMap<ExpertId, f32> = BTreeMap::new();

        // Score experts based on keyword matches
        for (keyword, expert_ids) in &self.keyword_index {
            if query_lower.contains(keyword.as_str()) {
                for &id in expert_ids {
                    *scores.entry(id).or_insert(0.0) += 1.0;
                }
            }
        }

        // Also check domain names
        for (id, state) in &self.experts {
            if query_lower.contains(state.config.domain.name.as_str()) {
                *scores.entry(*id).or_insert(0.0) += 2.0;
            }
        }

        // Apply strategy
        let mut ranked: Vec<(ExpertId, f32)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        match self.strategy {
            RoutingStrategy::TopK(k) => {
                ranked.truncate(k);
                ranked.into_iter().map(|(id, _)| id).collect()
            }
            RoutingStrategy::Threshold(t) => {
                ranked.into_iter()
                    .filter(|(id, score)| {
                        let expert = self.experts.get(id).unwrap();
                        *score >= expert.config.domain.threshold.max(t)
                    })
                    .map(|(id, _)| id)
                    .collect()
            }
            RoutingStrategy::Single => {
                ranked.into_iter().take(1).map(|(id, _)| id).collect()
            }
            RoutingStrategy::All => {
                self.experts.keys().copied().collect()
            }
        }
    }

    /// Get expert info
    pub fn get_expert(&self, id: ExpertId) -> Option<&ExpertState> {
        self.experts.get(&id)
    }

    /// List all experts
    pub fn list_experts(&self) -> impl Iterator<Item = &ExpertState> {
        self.experts.values()
    }

    /// Mark expert as loaded
    pub fn set_loaded(&mut self, id: ExpertId, loaded: bool) {
        if let Some(state) = self.experts.get_mut(&id) {
            state.loaded = loaded;
        }
    }

    /// Mark expert as busy
    pub fn set_busy(&mut self, id: ExpertId, busy: bool) {
        if let Some(state) = self.experts.get_mut(&id) {
            state.busy = busy;
        }
    }
}

/// Response synthesizer for combining expert outputs
pub struct Synthesizer {
    /// Combination strategy
    strategy: SynthesisStrategy,
}

/// Synthesis strategies
#[derive(Clone, Copy, Debug)]
pub enum SynthesisStrategy {
    /// Concatenate all responses
    Concatenate,
    /// Use voting for discrete answers
    Voting,
    /// Weighted average by confidence
    WeightedAverage,
    /// Use best response (highest confidence)
    Best,
    /// Let final expert summarize others
    Summarize,
}

/// Expert response
#[derive(Clone, Debug)]
pub struct ExpertResponse {
    pub expert_id: ExpertId,
    pub content: String,
    pub confidence: f32,
    pub latency_ms: u64,
}

impl Synthesizer {
    /// Create a new synthesizer
    pub fn new(strategy: SynthesisStrategy) -> Self {
        Self { strategy }
    }

    /// Synthesize multiple expert responses
    pub fn synthesize(&self, responses: &[ExpertResponse]) -> String {
        if responses.is_empty() {
            return String::from("No expert responses available.");
        }

        if responses.len() == 1 {
            return responses[0].content.clone();
        }

        match self.strategy {
            SynthesisStrategy::Concatenate => {
                responses.iter()
                    .map(|r| r.content.as_str())
                    .collect::<Vec<_>>()
                    .join("\n\n---\n\n")
            }
            SynthesisStrategy::Best => {
                responses.iter()
                    .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
                    .map(|r| r.content.clone())
                    .unwrap_or_default()
            }
            SynthesisStrategy::Voting | SynthesisStrategy::WeightedAverage => {
                // For text generation, default to best
                responses.iter()
                    .max_by(|a, b| a.confidence.partial_cmp(&b.confidence).unwrap())
                    .map(|r| r.content.clone())
                    .unwrap_or_default()
            }
            SynthesisStrategy::Summarize => {
                // Build context for summarization
                let mut context = String::from("Expert responses:\n\n");
                for (i, r) in responses.iter().enumerate() {
                    context.push_str(&alloc::format!(
                        "Expert {} (confidence: {:.2}):\n{}\n\n",
                        i + 1, r.confidence, r.content
                    ));
                }
                // Return context - actual summarization would need model inference
                context
            }
        }
    }
}

/// MoE-R Swarm orchestrator
pub struct Swarm {
    /// Router for expert selection
    pub router: Router,
    /// Response synthesizer
    pub synthesizer: Synthesizer,
    /// Active experts
    active: Vec<ExpertId>,
}

impl Swarm {
    /// Create a new swarm
    pub fn new(routing: RoutingStrategy, synthesis: SynthesisStrategy) -> Self {
        Self {
            router: Router::new(routing),
            synthesizer: Synthesizer::new(synthesis),
            active: Vec::new(),
        }
    }

    /// Process a query through the swarm
    pub fn process(&mut self, query: &str) -> Vec<ExpertId> {
        let experts = self.router.route(query);
        self.active = experts.clone();
        experts
    }

    /// Get active experts for current query
    pub fn active_experts(&self) -> &[ExpertId] {
        &self.active
    }

    /// Combine expert responses
    pub fn combine(&self, responses: &[ExpertResponse]) -> String {
        self.synthesizer.synthesize(responses)
    }
}
