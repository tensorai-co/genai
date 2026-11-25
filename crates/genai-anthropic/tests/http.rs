//! The provider against a mock Messages API: headers, retries, errors, and a whole stream.

use std::time::Duration;

use genai_anthropic::Anthropic;
use genai_core::http::{Http, HttpConfig, RetryPolicy};
use genai_core::{ApiErrorKind, Error, Message, Provider, Request, StopReason};
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const TEXT_STREAM: &str = include_str!("fixtures/text.sse");

fn provider(server: &MockServer) -> Anthropic {
    let mut config = HttpConfig::default();
    config.retry = RetryPolicy::new(2, Duration::from_millis(5), Duration::from_millis(50));
    let http = Http::new(&config).unwrap_or_else(|e| panic!("{e}"));
    Anthropic::builder("sk-ant-test").base_url(server.uri()).http(http).build().unwrap_or_else(|e| panic!("{e}"))
}

fn sse(body: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).insert_header("content-type", "text/event-stream").set_body_string(body)
}

fn request() -> Request {
    Request::new("anthropic/claude-haiku-5-5").message(Message::user("Hi")).max_output_tokens(64)
}

#[tokio::test]
async fn sends_auth_version_and_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages"))
        .and(header("authorization", "Bearer sk-ant-test"))
        .and(header("anthropic-version", "2026-10-21"))
        .and(header("accept", "text/event-stream"))
        .and(body_partial_json(serde_json::json!({ "model": "claude-haiku-5-5", "max_tokens": 64, "stream": true })))
        .respond_with(sse(TEXT_STREAM))
        .expect(1)
        .mount(&server)
        .await;
    let response = provider(&server).generate(request()).await;
    match response {
        Ok(r) => {
            assert_eq!(r.text(), "Hello!");
            assert_eq!(r.stop_reason, StopReason::Complete);
        }
        Err(e) => panic!("{e}"),
    }
}

#[tokio::test]
async fn retries_overload_then_succeeds() {
    let server = MockServer::start().await;
    let overloaded = ResponseTemplate::new(529).set_body_json(serde_json::json!({ "type": "error", "error": { "type": "overloaded_error", "message": "Overloaded" } }));
    Mock::given(path("/v1/messages")).respond_with(overloaded).up_to_n_times(1).expect(1).mount(&server).await;
    Mock::given(path("/v1/messages")).respond_with(sse(TEXT_STREAM)).expect(1).mount(&server).await;
    assert_eq!(provider(&server).generate(request()).await.map(|r| r.text()).ok().as_deref(), Some("Hello!"));
}

#[tokio::test]
async fn honors_retry_after_on_rate_limits() {
    let server = MockServer::start().await;
    let limited = ResponseTemplate::new(429).insert_header("retry-after", "0").set_body_json(serde_json::json!({ "type": "error", "error": { "type": "rate_limit_error", "message": "slow down" } }));
    Mock::given(path("/v1/messages")).respond_with(limited).up_to_n_times(1).expect(1).mount(&server).await;
    Mock::given(path("/v1/messages")).respond_with(sse(TEXT_STREAM)).expect(1).mount(&server).await;
    assert!(provider(&server).generate(request()).await.is_ok());
}

#[tokio::test]
async fn does_not_retry_a_spend_cap() {
    let server = MockServer::start().await;
    let capped = ResponseTemplate::new(429)
        .insert_header("request-id", "req_123")
        .set_body_json(serde_json::json!({ "type": "error", "error": { "type": "rate_limit_error", "message": "spend cap reached", "details": { "error_code": "enforced_spend_limit_reached" } } }));
    Mock::given(path("/v1/messages")).respond_with(capped).expect(1).mount(&server).await;
    match provider(&server).generate(request()).await {
        Err(Error::Api(e)) => {
            assert_eq!(e.kind, ApiErrorKind::RateLimited);
            assert_eq!(e.status, Some(429));
            assert_eq!(e.code.as_deref(), Some("enforced_spend_limit_reached"));
            assert_eq!(e.request_id.as_deref(), Some("req_123"));
        }
        other => panic!("expected a rate limit error, got {other:?}"),
    }
}

#[tokio::test]
async fn reports_bad_requests_without_retrying() {
    let server = MockServer::start().await;
    let bad = ResponseTemplate::new(400)
        .set_body_json(serde_json::json!({ "type": "error", "error": { "type": "invalid_request_error", "message": "prompt is too long: 2000001 tokens > 1000000 maximum" }, "request_id": "req_9" }));
    Mock::given(path("/v1/messages")).respond_with(bad).expect(1).mount(&server).await;
    match provider(&server).generate(request()).await {
        Err(Error::Api(e)) => {
            assert_eq!(e.kind, ApiErrorKind::ContextTooLong);
            assert_eq!(e.request_id.as_deref(), Some("req_9"));
        }
        other => panic!("expected an error, got {other:?}"),
    }
}

#[tokio::test]
async fn stream_cut_short_is_an_error() {
    let server = MockServer::start().await;
    let cut = TEXT_STREAM.split("event: message_delta").next().unwrap_or_default().to_owned();
    Mock::given(path("/v1/messages")).respond_with(sse(&cut)).mount(&server).await;
    assert!(matches!(provider(&server).generate(request()).await, Err(Error::Stream { .. })));
}

#[tokio::test]
async fn counts_tokens_with_the_trimmed_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/messages/count_tokens"))
        .and(body_partial_json(serde_json::json!({ "model": "claude-haiku-5-5" })))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({ "input_tokens": 12 })))
        .expect(1)
        .mount(&server)
        .await;
    let request = request();
    assert_eq!(provider(&server).count_tokens(&request).await.ok(), Some(12));
}

#[tokio::test]
async fn lists_models_across_pages() {
    let server = MockServer::start().await;
    let model = |id: &str| serde_json::json!({ "id": id, "display_name": id, "type": "model", "lifecycle": "active", "max_input_tokens": 1000, "max_tokens": 100, "capabilities": {} });
    Mock::given(path("/v1/models"))
        .and(wiremock::matchers::query_param("after_id", "a"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({ "data": [model("b")], "has_more": false, "last_id": "b" })))
        .mount(&server)
        .await;
    Mock::given(path("/v1/models")).respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({ "data": [model("a")], "has_more": true, "last_id": "a" }))).mount(&server).await;
    let ids: Option<Vec<String>> = provider(&server).models().await.ok().map(|m| m.into_iter().map(|m| m.id.name().to_owned()).collect());
    assert_eq!(ids, Some(vec!["a".to_owned(), "b".to_owned()]));
}
