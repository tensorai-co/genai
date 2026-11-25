//! What each Gemini model can do: a built-in catalog (thinking levels aren't in Google's models list), merged with the
//! live list when the app asks for it.

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
    /// The `thinking_level` values it accepts, lowest first. Empty when unknown: settings then pass through unchanged.
    pub(crate) levels: Vec<Effort>,
    pub(crate) default_level: Option<Effort>,
    pub(crate) thinking: bool,
    pub(crate) status: ModelStatus,
}

impl Spec {
    pub(crate) fn info(&self) -> ModelInfo {
        let mut info = ModelInfo::new(ModelId::new(ProviderId::Google, self.id.clone()), self.display_name.clone());
        info.family = self.family.clone();
        info.context_window = Some(self.context_window);
        info.max_output_tokens = Some(self.max_output_tokens);
        info.status = self.status.clone();
        let mut caps = Capabilities::default();
        caps.input = vec![Modality::Text, Modality::Image, Modality::Document, Modality::Audio, Modality::Video];
        caps.efforts = self.levels.clone();
        caps.default_effort = self.default_level;
        caps.reasoning = self.thinking;
        caps.reasoning_can_be_off = false;
        caps.tools = true;
        caps.forced_tool_choice = true;
        caps.structured_output = true;
        info.capabilities = caps;
        info
    }

    /// Reads one entry of `GET /v1beta/models`, keeping only models that hold a text conversation. The catalog fills in
    /// what the list doesn't report.
    pub(crate) fn from_live(model: &Value, catalog: &[Spec]) -> Option<Spec> {
        let id = model.get("name")?.as_str()?.trim_start_matches("models/").to_owned();
        let methods: Vec<&str> = model.get("supportedGenerationMethods").and_then(Value::as_array).into_iter().flatten().filter_map(Value::as_str).collect();
        if !is_chat_model(&id, &methods) {
            return None;
        }
        let known = find(catalog, &id);
        let number = |key: &str| model.get(key).and_then(Value::as_u64).map(|n| n as u32);
        Some(Spec {
            display_name: model.get("displayName").and_then(Value::as_str).map(str::to_owned).or_else(|| known.map(|k| k.display_name.clone())).unwrap_or_else(|| id.clone()),
            family: known.and_then(|k| k.family.clone()).or_else(|| family(&id)),
            context_window: number("inputTokenLimit").or(known.map(|k| k.context_window)).unwrap_or(1_048_576),
            max_output_tokens: number("outputTokenLimit").or(known.map(|k| k.max_output_tokens)).unwrap_or(65_536),
            levels: known.map(|k| k.levels.clone()).unwrap_or_default(),
            default_level: known.and_then(|k| k.default_level),
            thinking: model.get("thinking").and_then(Value::as_bool).unwrap_or(false),
            status: known.map_or_else(|| if id.contains("preview") { ModelStatus::Preview } else { ModelStatus::Active }, |k| k.status.clone()),
            id,
        })
    }
}

/// Whether a models-list entry is a model for text conversations, rather than speech, images, music, embeddings, live
/// audio, or an agent.
fn is_chat_model(id: &str, methods: &[&str]) -> bool {
    const NOT_CHAT: &[&str] =
        &["tts", "image", "embedding", "live", "transcribe", "omni", "robotics", "computer-use", "nano-banana", "lyria", "aqa", "decision", "deep-research", "antigravity", "genie"];
    methods.contains(&"generateContent") && (id.starts_with("gemini-") || id.starts_with("gemma-")) && !NOT_CHAT.iter().any(|word| id.contains(word))
}

/// `gemini-4` for `gemini-4-hydrogen`, `gemini-3.8` for `gemini-3.8-flash`.
fn family(id: &str) -> Option<String> {
    let mut parts = id.splitn(3, '-');
    let (a, b) = (parts.next()?, parts.next()?);
    b.starts_with(|c: char| c.is_ascii_digit()).then(|| format!("{a}-{b}"))
}

/// id, display name, max output, thinking levels, default level, status.
type Row = (&'static str, &'static str, u32, &'static [Effort], Option<Effort>, ModelStatus);

/// The built-in catalog: thinking levels from Google's thinking guide, limits from the models list (both 2026-11-24).
pub(crate) fn catalog() -> Vec<Spec> {
    use Effort::{High, Low, Medium, Minimal, XHigh};
    let lmh: &[Effort] = &[Low, Medium, High];
    let mlmh: &[Effort] = &[Minimal, Low, Medium, High];
    let lmhx: &[Effort] = &[Low, Medium, High, XHigh];
    let deprecated = |date: Option<&str>| ModelStatus::Deprecated { retires_at: date.map(str::to_owned) };
    let rows: [Row; 14] = [
        ("gemini-4-carbon", "Gemini 4 Carbon", 32_768, lmh, Some(Medium), ModelStatus::Active),
        ("gemini-4-argon", "Gemini 4 Argon", 32_768, lmh, Some(Medium), ModelStatus::Active),
        ("gemini-4-neon", "Gemini 4 Neon", 65_536, lmhx, Some(High), ModelStatus::Active),
        ("gemini-4-hydrogen", "Gemini 4 Hydrogen", 65_536, lmhx, Some(High), ModelStatus::Active),
        ("gemini-3.8-flash", "Gemini 3.8 Flash", 65_536, lmh, Some(Medium), ModelStatus::Active),
        ("gemini-3.7-flash", "Gemini 3.7 Flash", 65_536, lmh, Some(Medium), deprecated(None)),
        ("gemini-3.6-flash", "Gemini 3.6 Flash", 65_536, mlmh, Some(Medium), ModelStatus::Active),
        ("gemini-3.5-flash-lite", "Gemini 3.5 Flash-Lite", 65_536, mlmh, Some(Minimal), ModelStatus::Active),
        ("gemini-3.1-flash-lite", "Gemini 3.1 Flash-Lite", 65_536, mlmh, Some(Minimal), ModelStatus::Active),
        ("gemini-3.1-pro-preview", "Gemini 3.1 Pro Preview", 65_536, lmh, Some(High), ModelStatus::Preview),
        ("gemini-3-flash-preview", "Gemini 3 Flash Preview", 65_536, mlmh, Some(High), ModelStatus::Preview),
        ("gemini-2.5-pro", "Gemini 2.5 Pro", 65_536, lmh, None, deprecated(Some("2026-12-01T00:00:00Z"))),
        ("gemini-2.5-flash", "Gemini 2.5 Flash", 65_536, lmh, None, deprecated(Some("2026-12-01T00:00:00Z"))),
        ("gemini-2.5-flash-lite", "Gemini 2.5 Flash-Lite", 65_536, lmh, None, ModelStatus::Active),
    ];
    rows.into_iter()
        .map(|(id, name, output, levels, default_level, status)| Spec {
            id: id.into(),
            display_name: name.into(),
            family: family(id),
            context_window: 1_048_576,
            max_output_tokens: output,
            levels: levels.to_vec(),
            default_level,
            thinking: true,
            status,
        })
        .collect()
}

/// Finds `name` in `specs`, accepting the `models/` prefix.
pub(crate) fn find<'a>(specs: impl IntoIterator<Item = &'a Spec>, name: &str) -> Option<&'a Spec> {
    let name = name.trim_start_matches("models/");
    specs.into_iter().find(|s| s.id == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_only_chat_models_and_merges_the_catalog() {
        let catalog = catalog();
        let live = |name: &str, methods: &[&str]| serde_json::json!({ "name": name, "displayName": "X", "inputTokenLimit": 100, "outputTokenLimit": 10, "thinking": true, "supportedGenerationMethods": methods });
        let hydrogen = Spec::from_live(&live("models/gemini-4-hydrogen", &["generateContent", "countTokens"]), &catalog);
        assert_eq!(hydrogen.as_ref().map(|s| s.levels.clone()), Some(vec![Effort::Low, Effort::Medium, Effort::High, Effort::XHigh]));
        assert_eq!(hydrogen.as_ref().map(|s| (s.context_window, s.family.clone())), Some((100, Some("gemini-4".into()))));
        assert!(Spec::from_live(&live("models/gemini-3.5-flash", &["generateContent"]), &catalog).is_some_and(|s| s.levels.is_empty()));
        assert!(Spec::from_live(&live("models/gemini-3.8-flash-tts", &["generateContent"]), &catalog).is_none());
        assert!(Spec::from_live(&live("models/gemini-embedding-2", &["embedContent"]), &catalog).is_none());
        assert!(Spec::from_live(&live("models/gemini-3.8-live", &["bidiGenerateContent"]), &catalog).is_none());
    }
}
