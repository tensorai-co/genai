//! The provider against a mock Responses API: headers, retries, errors, and a whole stream.

use std::time::Duration;

use genai_core::http::{Http, HttpConfig, RetryPolicy};
use genai_core::{ApiErrorKind, Error, Message, Provider, Request, StopReason};
use genai_openai::OpenAi;
use wiremock::matchers::{body_partial_json, header, method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

const TEXT_STREAM: &str = include_str!("fixtures/text.sse");

fn provider(server: &MockServer) -> OpenAi {
    let mut config = HttpConfig::default();
    config.retry = RetryPolicy::new(2, Duration::from_millis(5), Duration::from_millis(50));
    let http = Http::new(&config).unwrap_or_else(|e| panic!("{e}"));
    OpenAi::builder("sk-test").base_url(format!("{}/v1", server.uri())).project("proj_1").http(http).build().unwrap_or_else(|e| panic!("{e}"))
}

fn sse(body: &str) -> ResponseTemplate {
    ResponseTemplate::new(200).insert_header("content-type", "text/event-stream").set_body_string(body)
}

fn request() -> Request {
    Request::new("openai/gpt-6-luna").message(Message::user("Hi")).max_output_tokens(64)
}

#[tokio::test]
async fn sends_auth_and_a_stateless_body() {
    let server = MockServer::start().await;
    Mock::given(method("POST"))
        .and(path("/v1/responses"))
        .and(header("authorization", "Bearer sk-test"))
        .and(header("openai-project", "proj_1"))
        .and(body_partial_json(serde_json::json!({ "model": "gpt-6-luna", "stream": true, "store": false, "include": ["reasoning.encrypted_content"], "max_output_tokens": 64 })))
        .respond_with(sse(TEXT_STREAM))
        .expect(1)
        .mount(&server)
        .await;
    match provider(&server).generate(request()).await {
        Ok(r) => {
            assert_eq!(r.text(), "Hi there, friend.");
            assert_eq!(r.stop_reason, StopReason::Complete);
        }
        Err(e) => panic!("{e}"),
    }
}

#[tokio::test]
async fn retries_rate_limits_and_overload() {
    let server = MockServer::start().await;
    let limited = ResponseTemplate::new(429)
        .insert_header("retry-after", "0")
        .set_body_json(serde_json::json!({ "error": { "message": "slow down", "type": "rate_limit_error", "code": "rate_limit_exceeded" } }));
    let busy = ResponseTemplate::new(503).set_body_json(serde_json::json!({ "error": { "message": "busy", "type": "service_unavailable_error", "code": "server_is_overloaded" } }));
    Mock::given(path("/v1/responses")).respond_with(limited).up_to_n_times(1).expect(1).mount(&server).await;
    Mock::given(path("/v1/responses")).respond_with(busy).up_to_n_times(1).expect(1).mount(&server).await;
    Mock::given(path("/v1/responses")).respond_with(sse(TEXT_STREAM)).expect(1).mount(&server).await;
    assert!(provider(&server).generate(request()).await.is_ok());
}

#[tokio::test]
async fn does_not_retry_spent_credit() {
    let server = MockServer::start().await;
    let quota = ResponseTemplate::new(429)
        .insert_header("x-request-id", "req_abc")
        .set_body_json(serde_json::json!({ "error": { "message": "You exceeded your current quota.", "type": "insufficient_quota", "code": "insufficient_quota" } }));
    Mock::given(path("/v1/responses")).respond_with(quota).expect(1).mount(&server).await;
    match provider(&server).generate(request()).await {
        Err(Error::Api(e)) => assert_eq!((e.kind, e.code.as_deref(), e.request_id.as_deref()), (ApiErrorKind::Billing, Some("insufficient_quota"), Some("req_abc"))),
        other => panic!("expected a billing error, got {other:?}"),
    }
}

#[tokio::test]
async fn reports_unknown_models() {
    let server = MockServer::start().await;
    let missing = ResponseTemplate::new(404)
        .set_body_json(serde_json::json!({ "error": { "message": "The model `gpt-nope` does not exist.", "type": "invalid_request_error", "param": null, "code": "model_not_found" } }));
    Mock::given(path("/v1/responses")).respond_with(missing).expect(1).mount(&server).await;
    assert!(matches!(provider(&server).generate(request().model("openai/gpt-nope")).await, Err(Error::Api(e)) if e.kind == ApiErrorKind::NotFound));
}

#[tokio::test]
async fn stream_without_completion_is_an_error() {
    let server = MockServer::start().await;
    let cut = TEXT_STREAM.split("event: response.completed").next().unwrap_or_default().to_owned();
    Mock::given(path("/v1/responses")).respond_with(sse(&cut)).mount(&server).await;
    assert!(matches!(provider(&server).generate(request()).await, Err(Error::Stream { .. })));
}

#[tokio::test]
async fn lists_chat_models() {
    let server = MockServer::start().await;
    let model = |id: &str| serde_json::json!({ "id": id, "object": "model", "owned_by": "system", "shutdown_date": null });
    let list = serde_json::json!({ "object": "list", "data": [model("gpt-6.2-sol"), model("gpt-image-2"), model("gpt-6-luna"), model("text-embedding-3-small")] });
    Mock::given(path("/v1/models")).respond_with(ResponseTemplate::new(200).set_body_json(list)).mount(&server).await;
    let ids: Option<Vec<String>> = provider(&server).models().await.ok().map(|m| m.into_iter().map(|m| m.id.name().to_owned()).collect());
    assert_eq!(ids, Some(vec!["gpt-6-luna".to_owned(), "gpt-6.2-sol".to_owned()]));
}
