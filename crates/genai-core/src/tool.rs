//! Tools the model can call.

use serde::{Deserialize, Serialize};

/// A function the model can call. The app runs it and sends back a [`ToolResult`](crate::ToolResult).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct Tool {
    /// The name the model calls it by: letters, digits, `_` and `-`, at most 64 characters to be safe across providers.
    pub name: String,
    /// What it does and when to use it. Optional, but the model relies on it.
    pub description: Option<String>,
    /// A JSON Schema for its arguments, with `"type": "object"` at the top.
    pub parameters: serde_json::Value,
    /// Whether the provider must keep the arguments valid against the schema. Providers that support this need a
    /// stricter schema; genai adjusts it.
    pub strict: bool,
}

impl Tool {
    /// A function tool named `name` that takes no arguments; add them with [`parameters`](Tool::parameters).
    pub fn function(name: impl Into<String>) -> Self {
        Tool { name: name.into(), description: None, parameters: serde_json::json!({ "type": "object", "properties": {} }), strict: false }
    }

    /// Sets the description.
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Sets the JSON Schema for the arguments.
    pub fn parameters(mut self, schema: serde_json::Value) -> Self {
        self.parameters = schema;
        self
    }

    /// Derives the argument schema from a Rust type.
    #[cfg(feature = "schemars")]
    pub fn parameters_from<T: schemars::JsonSchema>(self) -> Self {
        let schema = serde_json::to_value(schemars::schema_for!(T)).unwrap_or_default();
        self.parameters(schema)
    }

    /// Asks the provider to keep the arguments valid against the schema.
    pub fn strict(mut self, strict: bool) -> Self {
        self.strict = strict;
        self
    }
}
