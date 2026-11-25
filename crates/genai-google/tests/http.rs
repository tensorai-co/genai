//! The provider against a mock Interactions API: headers, retries, errors, and a whole stream.

use std::time::Duration;

use genai_core::http::{Http, HttpConfig, RetryPolicy};
use genai_core::{ApiErrorKind, Error, Message, Provider, Request, StopReason};
use genai_google::Google;
use wiremock::matchers::{body_partial_json, header, method, path, query_param};
use wiremock::{Mock, MockServer, ResponseTemplate};

const TEXT_STREAM: &str = include_str!("fixtures/text.sse");

fn provider(server: &MockServer) -> Google {
    let mut config = HttpConfig::default();
    config.retry = RetryPolicy::new(2, Duration::from_millis(5), Duration::from_millis(50));
    let http = Http::new(&config).unwrap_or_else(|e| panic!("{e}"));
    Google::builder("test-key").base_url(server.uri()).http(http).build().unwrap_or_else(|e| panic!("{e}"))
}

fn sse(body: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).insert_header("content-type", "text/event-stream").set_body_string(body)
}

fn request() -> Request {
    Request::new("google/gemini-3.5-flash-lite").message(Message::user("Hi")).max_output_tokens(64)
}

#[tokio::test]
async fn sends_key_revision_and_a_stateless_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1beta/interactions"))
        .and(header("x-goog-api-key", "test-key"))
        .and(header("api-revision", "2026-05-20"))
        .and(body_partial_json(serde_json::json!({ "model": "gemini-3.5-flash-lite", "stream": true, "store": false, "generation_config": { "max_output_tokens": 64 } })))
        .respond_with(sse(TEXT_STREAM))
        .expect(1)
        .mount(&server)
        .await;
    match provider(&server).generate(request()).await {
        Ok(r) => {
            assert_eq!(r.text(), "Hello there, friend!");
            assert_eq!(r.stop_reason, StopReason::Complete);
        }
        Err(e) => panic!("{e}"),
    }
}

#[tokio::test]
async fn retries_unavailable_then_succeeds() {
    let server = MockServer::start().await;
    let busy = ResponseTemplate::new(503).set_body_json(serde_json::json!({ "error": { "code": "service_unavailable", "message": "Overloaded" } }));
    Mock::given(path("/v1beta/interactions")).respond_with(busy).up_to_n_times(1).expect(1).mount(&server).await;
    Mock::given(path("/v1beta/interactions")).respond_with(sse(TEXT_STREAM)).expect(1).mount(&server).await;
    assert!(provider(&server).generate(request()).await.is_ok());
}

#[tokio::test]
async fn does_not_retry_a_daily_quota() {
    let server = MockServer::start().await;
    let quota = ResponseTemplate::new(429).set_body_json(serde_json::json!({ "error": { "code": "quota_exceeded", "message": "Daily quota exceeded." } }));
    Mock::given(path("/v1beta/interactions")).respond_with(quota).expect(1).mount(&server).await;
    match provider(&server).generate(request()).await {
        Err(Error::Api(e)) => assert_eq!((e.kind, e.code.as_deref(), e.status), (ApiErrorKind::RateLimited, Some("quota_exceeded"), Some(429))),
        other => panic!("expected a quota error, got {other:?}"),
    }
}

#[tokio::test]
async fn reports_unknown_models() {
    let server = MockServer::start().await;
    let missing = ResponseTemplate::new(404).set_body_json(serde_json::json!({ "error": { "code": "not_found", "message": "Model 'gemini-nope' not found." } }));
    Mock::given(path("/v1beta/interactions")).respond_with(missing).expect(1).mount(&server).await;
    let result = provider(&server).generate(request().model("google/gemini-nope")).await;
    assert!(matches!(result, Err(Error::Api(e)) if e.kind == ApiErrorKind::NotFound));
}

#[tokio::test]
async fn stream_without_completion_is_an_error() {
    let server = MockServer::start().await;
    let cut = TEXT_STREAM.split("event: interaction.completed").next().unwrap_or_default().to_owned();
    Mock::given(path("/v1beta/interactions")).respond_with(sse(&cut)).mount(&server).await;
    assert!(matches!(provider(&server).generate(request()).await, Err(Error::Stream { .. })));
}

#[tokio::test]
async fn lists_chat_models_across_pages() {
    let server = MockServer::start().await;
    let model = |name: &str, methods: &[&str]| serde_json::json!({ "name": name, "displayName": name, "inputTokenLimit": 1000, "outputTokenLimit": 100, "supportedGenerationMethods": methods });
    Mock::given(path("/v1beta/models"))
        .and(query_param("pageToken", "next"))
        .respond_with(
            ResponseTemplate::new(200).set_body_json(serde_json::json!({ "models": [model("models/gemini-4-neon", &["generateContent"]), model("models/gemini-embedding-2", &["embedContent"])] })),
        )
        .mount(&server)
        .await;
    Mock::given(path("/v1beta/models"))
        .respond_with(ResponseTemplate::new(200).set_body_json(serde_json::json!({ "models": [model("models/gemini-4-hydrogen", &["generateContent"])], "nextPageToken": "next" })))
        .mount(&server)
        .await;
    let ids: Option<Vec<String>> = provider(&server).models().await.ok().map(|m| m.into_iter().map(|m| m.id.name().to_owned()).collect());
    assert_eq!(ids, Some(vec!["gemini-4-hydrogen".to_owned(), "gemini-4-neon".to_owned()]));
}
