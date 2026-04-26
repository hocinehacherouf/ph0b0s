//! Mock implementations of `LlmAgent` and `LlmSession` for tests.
//!
//! Both types share state behind `Arc<Mutex<...>>` so a clone of the mock
//! sees the same canned-response queue and the same recorded calls. This
//! lets a test enqueue on one handle and inspect on another.
//!
//! Mutex choice: `std::sync::Mutex`. Critical sections are short
//! (`push`/`pop` on a `VecDeque`), and never held across `.await`.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use ph0b0s_core::error::LlmError;
use ph0b0s_core::llm::{
    AgentRoleKey, ChatRequest, ChatResponse, LlmAgent, LlmSession, SessionOptions,
    StructuredRequest, Usage, UserMessage,
};

// ---------------------------------------------------------------------------
// MockLlmAgent
// ---------------------------------------------------------------------------

#[derive(Default)]
struct MockLlmState {
    chat_responses: Mutex<VecDeque<Result<ChatResponse, LlmError>>>,
    structured_responses: Mutex<VecDeque<Result<serde_json::Value, LlmError>>>,
    recorded_chats: Mutex<Vec<ChatRequest>>,
    recorded_structured: Mutex<Vec<StructuredRequest>>,
    recorded_session_opts: Mutex<Vec<SessionOptions>>,
    /// Optional pre-configured session that `LlmAgent::session` clones and
    /// returns. The Arc-shared state means tests can keep an inspector
    /// handle to assert on after the session has been boxed.
    session_template: Mutex<Option<MockLlmSession>>,
}

/// Mock `LlmAgent` with canned response queues per method and full
/// recording of inputs.
///
/// `MockLlmAgent` is `Clone`; clones share state via `Arc`. This is the
/// supported way to keep an inspector handle while the runtime owns the
/// agent as `&dyn LlmAgent`.
#[derive(Clone)]
pub struct MockLlmAgent {
    model_id: String,
    role: AgentRoleKey,
    state: Arc<MockLlmState>,
}

impl MockLlmAgent {
    pub fn new() -> Self {
        Self {
            model_id: "mock".into(),
            role: AgentRoleKey::new("test"),
            state: Arc::new(MockLlmState::default()),
        }
    }

    pub fn with_role(mut self, role: impl Into<AgentRoleKey>) -> Self {
        self.role = role.into();
        self
    }

    pub fn with_model(mut self, model_id: impl Into<String>) -> Self {
        self.model_id = model_id.into();
        self
    }

    pub fn enqueue_chat_ok(&self, resp: ChatResponse) -> &Self {
        self.state
            .chat_responses
            .lock()
            .expect("chat_responses mutex poisoned")
            .push_back(Ok(resp));
        self
    }

    pub fn enqueue_chat_err(&self, err: LlmError) -> &Self {
        self.state
            .chat_responses
            .lock()
            .expect("chat_responses mutex poisoned")
            .push_back(Err(err));
        self
    }

    /// Enqueue a chat response with just `content` set (rest defaulted).
    pub fn enqueue_chat_text(&self, text: &str) -> &Self {
        self.enqueue_chat_ok(ChatResponse {
            content: text.to_owned(),
            ..Default::default()
        })
    }

    pub fn enqueue_structured_ok(&self, value: serde_json::Value) -> &Self {
        self.state
            .structured_responses
            .lock()
            .expect("structured_responses mutex poisoned")
            .push_back(Ok(value));
        self
    }

    pub fn enqueue_structured_err(&self, err: LlmError) -> &Self {
        self.state
            .structured_responses
            .lock()
            .expect("structured_responses mutex poisoned")
            .push_back(Err(err));
        self
    }

    /// Provide a session template. Subsequent `session()` calls return a
    /// `Clone` of this template; because `MockLlmSession` shares state via
    /// `Arc`, queue/usage configured on the template stays observable.
    pub fn set_session_template(&self, template: MockLlmSession) -> &Self {
        *self
            .state
            .session_template
            .lock()
            .expect("session_template mutex poisoned") = Some(template);
        self
    }

    pub fn recorded_chats(&self) -> Vec<ChatRequest> {
        self.state
            .recorded_chats
            .lock()
            .expect("recorded_chats mutex poisoned")
            .clone()
    }

    pub fn recorded_structured(&self) -> Vec<StructuredRequest> {
        self.state
            .recorded_structured
            .lock()
            .expect("recorded_structured mutex poisoned")
            .clone()
    }

    pub fn recorded_session_opts(&self) -> Vec<SessionOptions> {
        self.state
            .recorded_session_opts
            .lock()
            .expect("recorded_session_opts mutex poisoned")
            .clone()
    }
}

impl Default for MockLlmAgent {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl LlmAgent for MockLlmAgent {
    async fn chat(&self, req: ChatRequest) -> Result<ChatResponse, LlmError> {
        self.state
            .recorded_chats
            .lock()
            .expect("recorded_chats mutex poisoned")
            .push(req);
        self.state
            .chat_responses
            .lock()
            .expect("chat_responses mutex poisoned")
            .pop_front()
            .unwrap_or_else(|| Err(LlmError::Other("MockLlmAgent: chat queue exhausted".into())))
    }

    async fn structured(&self, req: StructuredRequest) -> Result<serde_json::Value, LlmError> {
        self.state
            .recorded_structured
            .lock()
            .expect("recorded_structured mutex poisoned")
            .push(req);
        self.state
            .structured_responses
            .lock()
            .expect("structured_responses mutex poisoned")
            .pop_front()
            .unwrap_or_else(|| {
                Err(LlmError::Other(
                    "MockLlmAgent: structured queue exhausted".into(),
                ))
            })
    }

    async fn session(&self, opts: SessionOptions) -> Result<Box<dyn LlmSession>, LlmError> {
        self.state
            .recorded_session_opts
            .lock()
            .expect("recorded_session_opts mutex poisoned")
            .push(opts);
        let session = self
            .state
            .session_template
            .lock()
            .expect("session_template mutex poisoned")
            .clone()
            .unwrap_or_default();
        Ok(Box::new(session))
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn role(&self) -> &AgentRoleKey {
        &self.role
    }
}

// ---------------------------------------------------------------------------
// MockLlmSession
// ---------------------------------------------------------------------------

#[derive(Default)]
struct MockSessionState {
    responses: Mutex<VecDeque<Result<ChatResponse, LlmError>>>,
    recorded_sends: Mutex<Vec<UserMessage>>,
    cumulative_usage: Mutex<Usage>,
}

/// Mock `LlmSession`. Cloneable; clones share state via `Arc`.
///
/// `usage()` returns the running sum of `Usage` values from each successful
/// canned response, optionally seeded via `with_initial_usage`.
#[derive(Clone, Default)]
pub struct MockLlmSession {
    state: Arc<MockSessionState>,
}

impl MockLlmSession {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn enqueue_send_ok(&self, resp: ChatResponse) -> &Self {
        self.state
            .responses
            .lock()
            .expect("session responses mutex poisoned")
            .push_back(Ok(resp));
        self
    }

    pub fn enqueue_send_err(&self, err: LlmError) -> &Self {
        self.state
            .responses
            .lock()
            .expect("session responses mutex poisoned")
            .push_back(Err(err));
        self
    }

    pub fn with_initial_usage(self, usage: Usage) -> Self {
        *self
            .state
            .cumulative_usage
            .lock()
            .expect("session usage mutex poisoned") = usage;
        self
    }

    pub fn recorded_sends(&self) -> Vec<UserMessage> {
        self.state
            .recorded_sends
            .lock()
            .expect("session recorded_sends mutex poisoned")
            .clone()
    }
}

#[async_trait]
impl LlmSession for MockLlmSession {
    async fn send(&mut self, msg: UserMessage) -> Result<ChatResponse, LlmError> {
        self.state
            .recorded_sends
            .lock()
            .expect("session recorded_sends mutex poisoned")
            .push(msg);
        let result = self
            .state
            .responses
            .lock()
            .expect("session responses mutex poisoned")
            .pop_front()
            .unwrap_or_else(|| {
                Err(LlmError::Other(
                    "MockLlmSession: send queue exhausted".into(),
                ))
            });
        if let Ok(ref resp) = result {
            let mut cum = self
                .state
                .cumulative_usage
                .lock()
                .expect("session usage mutex poisoned");
            cum.tokens_in = cum.tokens_in.saturating_add(resp.usage.tokens_in);
            cum.tokens_out = cum.tokens_out.saturating_add(resp.usage.tokens_out);
            cum.cost_usd_estimate += resp.usage.cost_usd_estimate;
            cum.cost_source = resp.usage.cost_source;
        }
        result
    }

    fn usage(&self) -> Usage {
        *self
            .state
            .cumulative_usage
            .lock()
            .expect("session usage mutex poisoned")
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ph0b0s_core::llm::{ChatMessage, CostSource, FinishReason};

    #[tokio::test]
    async fn chat_records_request_and_returns_canned_response() {
        let agent = MockLlmAgent::new();
        agent.enqueue_chat_text("hello world");

        let req = ChatRequest::new().system("you are a test").user("hi");
        let resp = agent.chat(req.clone()).await.unwrap();

        assert_eq!(resp.content, "hello world");
        let recorded = agent.recorded_chats();
        assert_eq!(recorded.len(), 1);
        assert_eq!(recorded[0], req);
    }

    #[tokio::test]
    async fn chat_queue_exhausted_returns_clear_error() {
        let agent = MockLlmAgent::new();
        let err = agent.chat(ChatRequest::new()).await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("chat queue exhausted"), "got: {msg}");
    }

    #[tokio::test]
    async fn chat_err_propagates() {
        let agent = MockLlmAgent::new();
        agent.enqueue_chat_err(LlmError::RateLimited {
            retry_after_secs: Some(7),
        });
        let err = agent.chat(ChatRequest::new()).await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("rate limited"), "got: {msg}");
    }

    #[tokio::test]
    async fn structured_records_request_and_returns_canned_value() {
        let agent = MockLlmAgent::new();
        agent.enqueue_structured_ok(serde_json::json!({"issues": []}));

        let req = StructuredRequest {
            messages: vec![ChatMessage::User {
                content: "scan this".into(),
            }],
            schema: serde_json::json!({"type": "object"}),
            schema_name: "Issues".into(),
            tools: Vec::new(),
            hints: Default::default(),
        };
        let value = agent.structured(req.clone()).await.unwrap();
        assert_eq!(value, serde_json::json!({"issues": []}));
        let recorded = agent.recorded_structured();
        assert_eq!(recorded, vec![req]);
    }

    #[tokio::test]
    async fn structured_queue_exhausted_returns_clear_error() {
        let agent = MockLlmAgent::new();
        let req = StructuredRequest {
            messages: Vec::new(),
            schema: serde_json::json!({}),
            schema_name: "X".into(),
            tools: Vec::new(),
            hints: Default::default(),
        };
        let err = agent.structured(req).await.unwrap_err();
        assert!(err.to_string().contains("structured queue exhausted"));
    }

    #[tokio::test]
    async fn cloned_agent_shares_queue_and_recorded_calls() {
        let agent = MockLlmAgent::new();
        let inspector = agent.clone();

        agent
            .enqueue_chat_text("from clone")
            .enqueue_chat_text("again");
        let _ = inspector.chat(ChatRequest::new()).await.unwrap();
        let _ = agent.chat(ChatRequest::new()).await.unwrap();

        // both clones see all recorded calls
        assert_eq!(agent.recorded_chats().len(), 2);
        assert_eq!(inspector.recorded_chats().len(), 2);
    }

    #[tokio::test]
    async fn enqueue_chain_is_fluent() {
        let agent = MockLlmAgent::new();
        agent
            .enqueue_chat_text("a")
            .enqueue_chat_text("b")
            .enqueue_chat_text("c");

        for expected in ["a", "b", "c"] {
            let r = agent.chat(ChatRequest::new()).await.unwrap();
            assert_eq!(r.content, expected);
        }
    }

    #[tokio::test]
    async fn session_records_opts_and_returns_mock() {
        let agent = MockLlmAgent::new();
        let opts = SessionOptions {
            system_prompt: Some("be brief".into()),
            tools: Vec::new(),
            hints: Default::default(),
        };
        let mut sess = agent.session(opts.clone()).await.unwrap();

        // freshly-created session has an empty queue
        let err = sess.send(UserMessage::new("hi")).await.unwrap_err();
        assert!(err.to_string().contains("send queue exhausted"));

        let recorded = agent.recorded_session_opts();
        assert_eq!(recorded, vec![opts]);
    }

    #[tokio::test]
    async fn session_template_is_cloned_with_shared_state() {
        let template = MockLlmSession::new();
        template.enqueue_send_ok(ChatResponse {
            content: "from template".into(),
            usage: Usage {
                tokens_in: 5,
                tokens_out: 7,
                cost_usd_estimate: 0.0,
                cost_source: CostSource::Native,
            },
            finish_reason: FinishReason::Stop,
            tool_calls: Vec::new(),
        });
        let inspector = template.clone();

        let agent = MockLlmAgent::new();
        agent.set_session_template(template);

        let mut sess = agent.session(SessionOptions::default()).await.unwrap();
        let resp = sess.send(UserMessage::new("hello")).await.unwrap();
        assert_eq!(resp.content, "from template");

        // template inspector observes the recorded send and cumulative usage
        assert_eq!(inspector.recorded_sends().len(), 1);
        assert_eq!(inspector.usage().tokens_in, 5);
        assert_eq!(inspector.usage().tokens_out, 7);
    }

    #[tokio::test]
    async fn mock_session_send_accumulates_usage() {
        let mut sess = MockLlmSession::new();
        sess.enqueue_send_ok(ChatResponse {
            content: "one".into(),
            usage: Usage {
                tokens_in: 10,
                tokens_out: 20,
                cost_usd_estimate: 0.001,
                cost_source: CostSource::Native,
            },
            ..Default::default()
        });
        sess.enqueue_send_ok(ChatResponse {
            content: "two".into(),
            usage: Usage {
                tokens_in: 5,
                tokens_out: 8,
                cost_usd_estimate: 0.0005,
                cost_source: CostSource::Estimate,
            },
            ..Default::default()
        });

        let _ = sess.send(UserMessage::new("a")).await.unwrap();
        let _ = sess.send(UserMessage::new("b")).await.unwrap();

        let u = sess.usage();
        assert_eq!(u.tokens_in, 15);
        assert_eq!(u.tokens_out, 28);
        assert!((u.cost_usd_estimate - 0.0015).abs() < 1e-9);
        // most-recent cost_source wins
        assert_eq!(u.cost_source, CostSource::Estimate);
    }

    #[tokio::test]
    async fn mock_session_initial_usage_seeds_cumulative() {
        let mut sess = MockLlmSession::new().with_initial_usage(Usage {
            tokens_in: 100,
            tokens_out: 200,
            cost_usd_estimate: 0.5,
            cost_source: CostSource::Unknown,
        });
        sess.enqueue_send_ok(ChatResponse {
            usage: Usage {
                tokens_in: 1,
                tokens_out: 2,
                cost_usd_estimate: 0.01,
                cost_source: CostSource::Native,
            },
            ..Default::default()
        });
        let _ = sess.send(UserMessage::new("x")).await.unwrap();
        let u = sess.usage();
        assert_eq!(u.tokens_in, 101);
        assert_eq!(u.tokens_out, 202);
    }

    #[tokio::test]
    async fn mock_session_send_err_propagates_without_changing_usage() {
        let mut sess = MockLlmSession::new();
        sess.enqueue_send_err(LlmError::Auth("nope".into()));
        let _ = sess.send(UserMessage::new("x")).await.unwrap_err();
        assert_eq!(sess.usage(), Usage::default());
        assert_eq!(sess.recorded_sends().len(), 1);
    }

    #[test]
    fn agent_role_and_model_setters() {
        let agent = MockLlmAgent::new()
            .with_role("triager")
            .with_model("test-model-001");
        assert_eq!(agent.model_id(), "test-model-001");
        assert_eq!(agent.role().0, "triager");
    }

    #[test]
    fn agent_default_uses_role_test_and_model_mock() {
        let a = MockLlmAgent::default();
        assert_eq!(a.model_id(), "mock");
        assert_eq!(a.role().0, "test");
    }
}
