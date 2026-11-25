//! What each GPT model can do: a built-in catalog (OpenAI's models list reports only ids), merged with the live list.

use genai_core::{Capabilities, Effort, Modality, ModelId, ModelInfo, ModelStatus, ProviderId};
use serde_json::Value;

/// One model, as the request converter needs it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Spec {
    pub(crate) id: String,
    pub(crate) display_name: String,
    pub(crate) family: Option<String>,
    pub(crate) context_window: u32,
    pub(crate) max_output_tokens: u32,
    /// The `reasoning.effort` values it accepts, lowest first. Empty when unknown: settings then pass through unchanged.
    pub(crate) efforts: Vec<Effort>,
    pub(crate) default_effort: Option<Effort>,
    /// Whether it still takes `temperature` and `top_p`. GPT-5 and later reasoning models don't.
    pub(crate) sampling: bool,
    pub(crate) status: ModelStatus,
}

impl Spec {
    pub(crate) fn info(&self) -> ModelInfo {
        let mut info = ModelInfo::new(ModelId::new(ProviderId::OpenAi, self.id.clone()), self.display_name.clone());
        info.family = self.family.clone();
        info.context_window = Some(self.context_window);
        info.max_output_tokens = Some(self.max_output_tokens);
        info.status = self.status.clone();
        let mut caps = Capabilities::default();
        caps.input = vec![Modality::Text, Modality::Image, Modality::Document];
        caps.efforts = self.efforts.clone();
        caps.default_effort = self.default_effort;
        caps.reasoning = !self.efforts.is_empty();
        caps.reasoning_can_be_off = self.efforts.contains(&Effort::None);
        caps.tools = true;
        caps.forced_tool_choice = true;
        caps.structured_output = true;
        info.capabilities = caps;
        info
    }

    /// Reads one entry of `GET /v1/models`, keeping only models that hold a text conversation.
    pub(crate) fn from_live(model: &Value, catalog: &[Spec]) -> Option<Spec> {
        let id = model.get("id")?.as_str()?.to_owned();
        if !is_chat_model(&id) {
            return None;
        }
        let shutdown = model.get("shutdown_date").and_then(Value::as_str).map(str::to_owned);
        let mut spec = find(catalog, &id).cloned().unwrap_or_else(|| Spec {
            id: id.clone(),
            display_name: id.clone(),
            family: None,
            context_window: 400_000,
            max_output_tokens: 128_000,
            efforts: Vec::new(),
            default_effort: None,
            sampling: true,
            status: ModelStatus::Active,
        });
        spec.id = id;
        if shutdown.is_some() {
            spec.status = ModelStatus::Deprecated { retires_at: shutdown };
        }
        Some(spec)
    }
}

/// Whether a models-list id is a model for text conversations, rather than audio, images, embeddings or moderation.
fn is_chat_model(id: &str) -> bool {
    const NOT_CHAT: &[&str] = &["image", "tts", "transcribe", "audio", "realtime", "live", "embedding", "moderation", "whisper", "dall-e", "search-api", "chatgpt-", "translate"];
    let family = id.starts_with("gpt-") || (id.starts_with('o') && id[1..].starts_with(|c: char| c.is_ascii_digit()));
    family && !NOT_CHAT.iter().any(|word| id.contains(word))
}

/// id, display name, family, context window, efforts, default effort.
type Row = (&'static str, &'static str, &'static str, u32, &'static [Effort], Option<Effort>);

/// The built-in catalog, from OpenAI's model pages (2026-11-24). Every model here gives 128K output tokens and rejects
/// `temperature` and `top_p`.
pub(crate) fn catalog() -> Vec<Spec> {
    use Effort::{High, Low, Max, Medium, None as Off, XHigh};
    let low_max: &[Effort] = &[Low, Medium, High, XHigh, Max];
    let off_max: &[Effort] = &[Off, Low, Medium, High, XHigh, Max];
    let off_xhigh: &[Effort] = &[Off, Low, Medium, High, XHigh];
    let rows: [Row; 14] = [
        ("gpt-6.2-astra", "GPT-6.2 Astra", "astra", 1_050_000, low_max, Some(Medium)),
        ("gpt-6.2-sol", "GPT-6.2 Sol", "sol", 1_050_000, low_max, Some(Medium)),
        ("gpt-6.1-terra", "GPT-6.1 Terra", "terra", 1_050_000, off_max, Some(Medium)),
        ("gpt-6.1-sol", "GPT-6.1 Sol", "sol", 1_050_000, low_max, Some(Medium)),
        ("gpt-6-luna", "GPT-6 Luna", "luna", 1_050_000, off_max, Some(Medium)),
        ("gpt-6-astra", "GPT-6 Astra", "astra", 1_050_000, low_max, Some(Medium)),
        ("gpt-6-sol", "GPT-6 Sol", "sol", 1_050_000, off_max, Some(Medium)),
        ("gpt-5.6-sol", "GPT-5.6 Sol", "sol", 1_050_000, off_max, Some(Medium)),
        ("gpt-5.6-terra", "GPT-5.6 Terra", "terra", 1_050_000, off_max, Some(Medium)),
        ("gpt-5.6-luna", "GPT-5.6 Luna", "luna", 1_050_000, off_max, Some(Medium)),
        ("gpt-5.5", "GPT-5.5", "gpt-5", 1_050_000, off_xhigh, Some(Medium)),
        ("gpt-5.4", "GPT-5.4", "gpt-5", 1_050_000, off_xhigh, Some(Off)),
        ("gpt-5.4-mini", "GPT-5.4 mini", "gpt-5", 400_000, off_xhigh, Some(Off)),
        ("gpt-5.2", "GPT-5.2", "gpt-5", 400_000, off_xhigh, Some(Off)),
    ];
    rows.into_iter()
        .map(|(id, name, family, context, efforts, default_effort)| Spec {
            id: id.into(),
            display_name: name.into(),
            family: Some(family.into()),
            context_window: context,
            max_output_tokens: 128_000,
            efforts: efforts.to_vec(),
            default_effort,
            sampling: false,
            status: ModelStatus::Active,
        })
        .collect()
}

/// Finds `name` in `specs`: exactly, or as a dated snapshot (`gpt-5.4-2026-03-05` for `gpt-5.4`).
pub(crate) fn find<'a>(specs: impl IntoIterator<Item = &'a Spec> + Clone, name: &str) -> Option<&'a Spec> {
    let snapshot_of = |snapshot: &str, alias: &str| {
        snapshot
            .strip_prefix(alias)
            .and_then(|rest| rest.strip_prefix('-'))
            .is_some_and(|date| date.len() == 10 && date.bytes().enumerate().all(|(i, b)| if i == 4 || i == 7 { b == b'-' } else { b.is_ascii_digit() }))
    };
    specs.clone().into_iter().find(|s| s.id == name).or_else(|| specs.into_iter().find(|s| snapshot_of(name, &s.id)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_snapshots_and_filters_the_live_list() {
        let catalog = catalog();
        assert_eq!(find(&catalog, "gpt-5.4-2026-03-05").map(|s| s.id.as_str()), Some("gpt-5.4"));
        assert_eq!(find(&catalog, "gpt-5.4-mini-2026-03-17").map(|s| s.id.as_str()), Some("gpt-5.4-mini"));
        assert!(find(&catalog, "gpt-5.4-pro").is_none());

        let live = |id: &str| serde_json::json!({ "id": id, "object": "model", "shutdown_date": null });
        let astra = Spec::from_live(&live("gpt-6.2-astra"), &catalog);
        assert_eq!(astra.map(|s| s.efforts.contains(&Effort::None)), Some(false));
        let unknown = Spec::from_live(&live("gpt-4.1"), &catalog);
        assert!(unknown.is_some_and(|s| s.efforts.is_empty() && s.sampling));
        for skipped in ["gpt-image-2", "gpt-4o-mini-tts", "text-embedding-3-small", "gpt-realtime-2", "omni-moderation-latest", "whisper-1", "gpt-5-search-api"] {
            assert!(Spec::from_live(&live(skipped), &catalog).is_none(), "{skipped}");
        }
        assert!(Spec::from_live(&live("o3"), &catalog).is_some());
        let retiring = Spec::from_live(&serde_json::json!({ "id": "gpt-5.2", "shutdown_date": "2027-01-01" }), &catalog);
        assert_eq!(retiring.map(|s| s.status), Some(ModelStatus::Deprecated { retires_at: Some("2027-01-01".into()) }));
    }
}
