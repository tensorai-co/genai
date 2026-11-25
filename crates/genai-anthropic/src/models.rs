//! What each Claude model can do: a built-in catalog, refreshed from the live models list when the app asks for it.

use genai_core::{Capabilities, Effort, Modality, ModelId, ModelInfo, ModelStatus, ProviderId};
use serde_json::Value;

/// Which `thinking.type` values a model accepts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Thinking {
    pub(crate) adaptive: bool,
    pub(crate) enabled: bool,
    pub(crate) disabled: bool,
    pub(crate) between_tools: bool,
}

/// One model, as the request converter needs it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Spec {
    pub(crate) id: String,
    pub(crate) display_name: String,
    pub(crate) line: Option<String>,
    pub(crate) context_window: u32,
    pub(crate) max_output_tokens: u32,
    /// The `output_config.effort` values it accepts, lowest first.
    pub(crate) efforts: Vec<Effort>,
    pub(crate) thinking: Thinking,
    pub(crate) forced_tool_choice: bool,
    pub(crate) structured_output: bool,
    pub(crate) images: bool,
    pub(crate) pdfs: bool,
    pub(crate) status: ModelStatus,
}

impl Spec {
    pub(crate) fn info(&self) -> ModelInfo {
        let mut input = vec![Modality::Text];
        if self.images {
            input.push(Modality::Image);
        }
        if self.pdfs {
            input.push(Modality::Document);
        }
        let can_be_off = self.thinking.disabled || self.thinking.between_tools;
        let mut efforts = self.efforts.clone();
        if can_be_off {
            efforts.insert(0, Effort::None);
        }
        let mut info = ModelInfo::new(ModelId::new(ProviderId::Anthropic, self.id.clone()), self.display_name.clone());
        info.family = self.line.clone();
        info.context_window = Some(self.context_window);
        info.max_output_tokens = Some(self.max_output_tokens);
        info.status = self.status.clone();
        let mut caps = Capabilities::default();
        caps.input = input;
        caps.efforts = efforts;
        caps.default_effort = default_effort(&self.id, &self.efforts);
        caps.reasoning = self.thinking.adaptive || self.thinking.enabled;
        caps.reasoning_can_be_off = can_be_off;
        caps.tools = true;
        caps.forced_tool_choice = self.forced_tool_choice;
        caps.structured_output = self.structured_output;
        info.capabilities = caps;
        info
    }

    /// Reads one entry of `GET /v1/models`.
    pub(crate) fn from_live(model: &Value) -> Option<Spec> {
        let id = model.get("id")?.as_str()?.to_owned();
        let caps = model.get("capabilities").unwrap_or(&Value::Null);
        let supported = |path: &[&str]| path.iter().try_fold(caps, |v, key| v.get(key)).and_then(|v| v.get("supported")).and_then(Value::as_bool).unwrap_or(false);
        let efforts = [Effort::Low, Effort::Medium, Effort::High, Effort::XHigh, Effort::Max].into_iter().filter(|e| supported(&["effort", e.as_str()])).collect();
        let thinking = Thinking {
            adaptive: supported(&["thinking", "types", "adaptive"]),
            enabled: supported(&["thinking", "types", "enabled"]),
            disabled: supported(&["thinking", "types", "disabled"]),
            between_tools: supported(&["thinking", "types", "between_tools"]),
        };
        let retires_at = model.get("retires_at").and_then(Value::as_str).map(str::to_owned);
        let status = match model.get("lifecycle").and_then(Value::as_str) {
            Some("deprecated") => ModelStatus::Deprecated { retires_at },
            _ => ModelStatus::Active,
        };
        Some(Spec {
            display_name: model.get("display_name").and_then(Value::as_str).unwrap_or(&id).to_owned(),
            line: model.get("line").and_then(Value::as_str).map(str::to_owned),
            context_window: model.get("max_input_tokens").and_then(Value::as_u64).map_or(200_000, |n| n as u32),
            max_output_tokens: model.get("max_tokens").and_then(Value::as_u64).map_or(64_000, |n| n as u32),
            efforts,
            thinking,
            forced_tool_choice: !NO_FORCED_TOOL_CHOICE.contains(&id.as_str()),
            structured_output: supported(&["structured_outputs"]),
            images: supported(&["image_input"]),
            pdfs: supported(&["pdf_input"]),
            status,
            id,
        })
    }
}

/// Models that reject `tool_choice` `any` and `tool` with a 400. The models list doesn't report this.
const NO_FORCED_TOOL_CHOICE: &[&str] = &["claude-fable-5-2", "claude-mythos-5-2", "claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-mythos-5-1"];

/// The effort a model uses when none is given.
fn default_effort(id: &str, efforts: &[Effort]) -> Option<Effort> {
    if efforts.is_empty() {
        None
    } else if id.starts_with("claude-opus-5-5") || id.starts_with("claude-haiku-5-5") {
        Some(Effort::Medium)
    } else {
        Some(Effort::High)
    }
}

/// id, display name, line, context window, max output, efforts, thinking modes, retirement date.
type Row = (&'static str, &'static str, &'static str, u32, u32, &'static [Effort], Thinking, Option<&'static str>);

const ALL_EFFORTS: &[Effort] = &[Effort::Low, Effort::Medium, Effort::High, Effort::XHigh, Effort::Max];

/// The built-in catalog, as of API version 2026-10-21 (from `GET /v1/models` on 2026-11-24).
pub(crate) fn catalog() -> Vec<Spec> {
    let adaptive = Thinking { adaptive: true, ..Thinking::default() };
    let adaptive_off = Thinking { adaptive: true, disabled: true, ..Thinking::default() };
    let all = Thinking { adaptive: true, enabled: true, disabled: true, between_tools: false };
    let budget = Thinking { enabled: true, disabled: true, ..Thinking::default() };
    let rows: [Row; 15] = [
        ("claude-fable-5-2", "Claude Fable 5.2", "fable", 1_000_000, 128_000, ALL_EFFORTS, adaptive, None),
        ("claude-opus-5-5", "Claude Opus 5.5", "opus", 1_000_000, 128_000, ALL_EFFORTS, adaptive, None),
        ("claude-sonnet-5-5", "Claude Sonnet 5.5", "sonnet", 1_000_000, 128_000, ALL_EFFORTS, Thinking { adaptive: true, between_tools: true, ..Thinking::default() }, None),
        ("claude-haiku-5-5", "Claude Haiku 5.5", "haiku", 1_000_000, 128_000, ALL_EFFORTS, adaptive_off, None),
        ("claude-fable-5-1", "Claude Fable 5.1", "fable", 1_000_000, 128_000, ALL_EFFORTS, adaptive, None),
        ("claude-fable-5", "Claude Fable 5", "fable", 1_000_000, 128_000, ALL_EFFORTS, adaptive, None),
        ("claude-opus-5", "Claude Opus 5", "opus", 1_000_000, 128_000, ALL_EFFORTS, adaptive_off, None),
        ("claude-sonnet-5", "Claude Sonnet 5", "sonnet", 1_000_000, 128_000, ALL_EFFORTS, adaptive_off, None),
        ("claude-opus-4-8", "Claude Opus 4.8", "opus", 1_000_000, 128_000, ALL_EFFORTS, adaptive_off, None),
        ("claude-opus-4-7", "Claude Opus 4.7", "opus", 1_000_000, 128_000, ALL_EFFORTS, adaptive_off, None),
        ("claude-opus-4-6", "Claude Opus 4.6", "opus", 1_000_000, 128_000, &[Effort::Low, Effort::Medium, Effort::High, Effort::Max], all, None),
        ("claude-sonnet-4-6", "Claude Sonnet 4.6", "sonnet", 1_000_000, 128_000, &[Effort::Low, Effort::Medium, Effort::High, Effort::Max], all, None),
        ("claude-opus-4-5-20251101", "Claude Opus 4.5", "opus", 200_000, 64_000, &[Effort::Low, Effort::Medium, Effort::High], budget, Some("2027-01-22T20:00:00Z")),
        ("claude-haiku-4-5-20251001", "Claude Haiku 4.5", "haiku", 200_000, 64_000, &[], budget, None),
        ("claude-sonnet-4-5-20250929", "Claude Sonnet 4.5", "sonnet", 200_000, 64_000, &[], budget, Some("2026-11-30T17:00:00Z")),
    ];
    rows.into_iter()
        .map(|(id, name, line, context, output, efforts, thinking, retires)| Spec {
            id: id.into(),
            display_name: name.into(),
            line: Some(line.into()),
            context_window: context,
            max_output_tokens: output,
            efforts: efforts.to_vec(),
            thinking,
            forced_tool_choice: !NO_FORCED_TOOL_CHOICE.contains(&id),
            structured_output: true,
            images: true,
            pdfs: true,
            status: retires.map_or(ModelStatus::Active, |r| ModelStatus::Deprecated { retires_at: Some(r.into()) }),
        })
        .collect()
}

/// Finds `name` in `specs`: exactly, or as the alias of a dated id (`claude-haiku-4-5` for `claude-haiku-4-5-20251001`),
/// or as a dated snapshot of an alias.
pub(crate) fn find<'a>(specs: impl IntoIterator<Item = &'a Spec> + Clone, name: &str) -> Option<&'a Spec> {
    let dated = |longer: &str, shorter: &str| longer.strip_prefix(shorter).and_then(|rest| rest.strip_prefix('-')).is_some_and(|date| date.len() == 8 && date.bytes().all(|b| b.is_ascii_digit()));
    specs.clone().into_iter().find(|s| s.id == name).or_else(|| specs.into_iter().find(|s| dated(&s.id, name) || dated(name, &s.id)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_aliases_of_dated_ids() {
        let specs = catalog();
        assert_eq!(find(&specs, "claude-haiku-4-5").map(|s| s.id.as_str()), Some("claude-haiku-4-5-20251001"));
        assert_eq!(find(&specs, "claude-opus-5-5").map(|s| s.id.as_str()), Some("claude-opus-5-5"));
        assert_eq!(find(&specs, "claude-opus-5-5-20261101").map(|s| s.id.as_str()), Some("claude-opus-5-5"));
        assert!(find(&specs, "claude-opus-5").is_some_and(|s| s.id == "claude-opus-5"));
        assert!(find(&specs, "claude-unknown").is_none());
    }

    #[test]
    fn reads_live_capabilities() {
        let live = serde_json::json!({
            "id": "claude-haiku-5-5", "display_name": "Claude Haiku 5.5", "line": "haiku", "lifecycle": "active",
            "max_input_tokens": 1000000, "max_tokens": 128000,
            "capabilities": {
                "effort": {"supported": true, "low": {"supported": true}, "medium": {"supported": true}, "high": {"supported": true}, "xhigh": {"supported": false}, "max": {"supported": true}},
                "thinking": {"supported": true, "types": {"adaptive": {"supported": true}, "disabled": {"supported": true}, "enabled": {"supported": false}, "between_tools": {"supported": false}}},
                "structured_outputs": {"supported": true}, "image_input": {"supported": true}, "pdf_input": {"supported": false}
            }
        });
        let spec = Spec::from_live(&live);
        assert_eq!(spec.as_ref().map(|s| s.efforts.clone()), Some(vec![Effort::Low, Effort::Medium, Effort::High, Effort::Max]));
        assert_eq!(spec.as_ref().map(|s| s.thinking.disabled), Some(true));
        let info = spec.map(|s| s.info());
        assert_eq!(info.as_ref().map(|i| i.capabilities.efforts.first().copied()), Some(Some(Effort::None)));
        assert_eq!(info.as_ref().map(|i| i.capabilities.input.contains(&Modality::Document)), Some(false));
        assert_eq!(info.map(|i| i.capabilities.default_effort), Some(Some(Effort::Medium)));
    }
}
