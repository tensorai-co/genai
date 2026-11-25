//! Streaming: the events every provider's stream is turned into, and folding them back into a [`Response`].

use std::collections::BTreeMap;
use std::pin::Pin;

use futures_core::Stream;
use futures_util::StreamExt;

use crate::error::{Error, Result};
use crate::message::{Message, Part, Role};
use crate::model::ModelId;
use crate::provider::ProviderId;
use crate::response::{Response, StopReason, Usage, Warning};

/// A response, as it streams in.
pub type EventStream = Pin<Box<dyn Stream<Item = Result<Event>> + Send>>;

/// One step of a streamed response. Every provider's stream comes down to the same sequence: [`Start`](Event::Start),
/// then for each part a [`PartStart`](Event::PartStart), deltas and a [`PartStop`](Event::PartStop), then
/// [`Finish`](Event::Finish). `index` is the part's position in the final message.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Event {
    /// The response started.
    Start {
        /// The provider's id for the response.
        id: String,
        /// The model answering.
        model: ModelId,
    },
    /// A part started. Text and reasoning parts start empty and fill through deltas.
    PartStart {
        /// Where it goes in the message.
        index: usize,
        /// The part so far.
        part: Part,
    },
    /// More text for the text part at `index`.
    TextDelta {
        /// Which part.
        index: usize,
        /// The new text.
        text: String,
    },
    /// More reasoning to show for the reasoning part at `index`.
    ReasoningDelta {
        /// Which part.
        index: usize,
        /// The new text.
        text: String,
    },
    /// More of the tool call's arguments at `index`, as a piece of a JSON string.
    ToolCallDelta {
        /// Which part.
        index: usize,
        /// The new piece of JSON.
        arguments: String,
    },
    /// A part finished; this is the whole part, with tool arguments parsed and provider data attached.
    PartStop {
        /// Where it goes in the message.
        index: usize,
        /// The finished part.
        part: Part,
    },
    /// Token usage so far.
    Usage(Usage),
    /// The response finished.
    Finish {
        /// Why the model stopped.
        stop_reason: StopReason,
        /// Tokens used in all.
        usage: Usage,
        /// Settings that were approximated.
        warnings: Vec<Warning>,
    },
}

/// Folds [`Event`]s into a [`Response`]. Feed it every event with [`push`](Accumulator::push), then call
/// [`finish`](Accumulator::finish); or hand it the whole stream with [`collect`](Accumulator::collect).
#[derive(Debug, Default)]
pub struct Accumulator {
    id: Option<String>,
    model: Option<ModelId>,
    parts: BTreeMap<usize, Part>,
    usage: Usage,
    finish: Option<(StopReason, Vec<Warning>)>,
}

impl Accumulator {
    /// An empty accumulator.
    pub fn new() -> Self {
        Accumulator::default()
    }

    /// Takes in one event.
    pub fn push(&mut self, event: &Event) {
        match event {
            Event::Start { id, model } => {
                self.id = Some(id.clone());
                self.model = Some(model.clone());
            }
            Event::PartStop { index, part } => {
                self.parts.insert(*index, part.clone());
            }
            Event::Usage(usage) => self.usage = usage.clone(),
            Event::Finish { stop_reason, usage, warnings } => {
                self.usage = usage.clone();
                self.finish = Some((stop_reason.clone(), warnings.clone()));
            }
            Event::PartStart { .. } | Event::TextDelta { .. } | Event::ReasoningDelta { .. } | Event::ToolCallDelta { .. } => {}
        }
    }

    /// The response, or an error if the stream ended before [`Event::Finish`].
    pub fn finish(self) -> Result<Response> {
        let model = self.model.unwrap_or_else(|| ModelId::new(ProviderId::Other(String::new()), ""));
        let Some((stop_reason, warnings)) = self.finish else {
            return Err(Error::Stream { provider: model.provider().clone(), message: "the stream ended before the response finished".into() });
        };
        Ok(Response { id: self.id.unwrap_or_default(), model, message: Message::new(Role::Assistant, self.parts.into_values().collect()), stop_reason, usage: self.usage, warnings })
    }

    /// Reads the whole stream into a response.
    pub async fn collect(mut stream: EventStream) -> Result<Response> {
        let mut acc = Accumulator::new();
        while let Some(event) = stream.next().await {
            acc.push(&event?);
        }
        acc.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn folds_events_in_index_order() {
        let model = ModelId::from("anthropic/claude-haiku-5-5");
        let events = vec![
            Ok(Event::Start { id: "msg_1".into(), model: model.clone() }),
            Ok(Event::PartStop { index: 1, part: Part::text("world") }),
            Ok(Event::PartStop { index: 0, part: Part::text("hello ") }),
            Ok(Event::Finish { stop_reason: StopReason::Complete, usage: Usage { input_tokens: 3, output_tokens: 2, ..Usage::default() }, warnings: vec![] }),
        ];
        let response = Accumulator::collect(Box::pin(futures_util::stream::iter(events))).await;
        let response = response.ok();
        assert_eq!(response.as_ref().map(Response::text).as_deref(), Some("hello world"));
        assert_eq!(response.map(|r| r.usage.output_tokens), Some(2));
    }

    #[tokio::test]
    async fn unfinished_stream_is_an_error() {
        let events = vec![Ok(Event::Start { id: "msg_1".into(), model: ModelId::from("anthropic/x") })];
        assert!(matches!(Accumulator::collect(Box::pin(futures_util::stream::iter(events))).await, Err(Error::Stream { .. })));
    }
}
