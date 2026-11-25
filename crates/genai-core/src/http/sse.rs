use std::collections::VecDeque;

use futures_core::Stream;
use futures_util::StreamExt;

use crate::error::{Error, Result};
use crate::provider::ProviderId;

/// One server-sent event.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SseEvent {
    /// The `event:` name, if the server sent one.
    pub event: Option<String>,
    /// The `data:` lines, joined with newlines.
    pub data: String,
}

/// An incremental server-sent events parser, following the HTML spec: lines end in `\n`, `\r\n` or `\r`; a blank line
/// ends an event; `:` starts a comment; `id:` and `retry:` are ignored. Bytes may arrive split anywhere, even inside a
/// UTF-8 character.
#[derive(Debug, Default)]
pub struct SseParser {
    buffer: Vec<u8>,
    event: Option<String>,
    data: Vec<String>,
    has_data: bool,
}

impl SseParser {
    /// A new parser.
    pub fn new() -> Self {
        SseParser::default()
    }

    /// Takes more bytes, and returns the events they complete.
    pub fn push(&mut self, bytes: &[u8]) -> Vec<SseEvent> {
        self.buffer.extend_from_slice(bytes);
        let mut events = Vec::new();
        let mut start = 0;
        let mut i = 0;
        while i < self.buffer.len() {
            match self.buffer[i] {
                b'\n' => {
                    let line = self.buffer[start..i].to_vec();
                    self.line(&line, &mut events);
                    i += 1;
                    start = i;
                }
                b'\r' => {
                    // A `\r` at the very end may be the first half of `\r\n`: wait for the next bytes.
                    if i + 1 == self.buffer.len() {
                        break;
                    }
                    let line = self.buffer[start..i].to_vec();
                    self.line(&line, &mut events);
                    i += if self.buffer[i + 1] == b'\n' { 2 } else { 1 };
                    start = i;
                }
                _ => i += 1,
            }
        }
        self.buffer.drain(..start);
        events
    }

    /// Ends the input. Returns the last event if the server didn't end it with a blank line.
    pub fn finish(&mut self) -> Option<SseEvent> {
        if !self.buffer.is_empty() {
            let line = std::mem::take(&mut self.buffer);
            let mut events = Vec::new();
            self.line(&line, &mut events);
        }
        self.dispatch()
    }

    fn line(&mut self, line: &[u8], events: &mut Vec<SseEvent>) {
        if line.is_empty() {
            events.extend(self.dispatch());
            return;
        }
        if line[0] == b':' {
            return;
        }
        let line = String::from_utf8_lossy(line);
        let (field, value) = match line.split_once(':') {
            Some((field, value)) => (field, value.strip_prefix(' ').unwrap_or(value)),
            None => (line.as_ref(), ""),
        };
        match field {
            "event" => self.event = Some(value.to_owned()),
            "data" => {
                self.data.push(value.to_owned());
                self.has_data = true;
            }
            _ => {}
        }
    }

    fn dispatch(&mut self) -> Option<SseEvent> {
        let event = self.event.take();
        if !self.has_data {
            return None;
        }
        self.has_data = false;
        Some(SseEvent { event, data: std::mem::take(&mut self.data).join("\n") })
    }
}

/// The events in a streaming HTTP response body.
pub fn sse_events(provider: ProviderId, response: reqwest::Response) -> impl Stream<Item = Result<SseEvent>> + Send + 'static {
    let bytes = response.bytes_stream();
    let state = (Box::pin(bytes), SseParser::new(), VecDeque::new(), false, provider);
    futures_util::stream::unfold(state, |(mut bytes, mut parser, mut pending, mut done, provider)| async move {
        loop {
            if let Some(event) = pending.pop_front() {
                return Some((Ok(event), (bytes, parser, pending, done, provider)));
            }
            if done {
                return None;
            }
            match bytes.next().await {
                Some(Ok(chunk)) => pending.extend(parser.push(&chunk)),
                Some(Err(e)) => {
                    done = true;
                    let error = Error::Network { provider: provider.clone(), source: Box::new(e) };
                    return Some((Err(error), (bytes, parser, pending, done, provider)));
                }
                None => {
                    done = true;
                    pending.extend(parser.finish());
                }
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all(chunks: &[&[u8]]) -> Vec<SseEvent> {
        let mut parser = SseParser::new();
        let mut events: Vec<SseEvent> = chunks.iter().flat_map(|c| parser.push(c)).collect();
        events.extend(parser.finish());
        events
    }

    #[test]
    fn parses_named_events_split_anywhere() {
        let body = b"event: message_start\ndata: {\"a\":1}\n\n: comment\nevent: ping\r\ndata: {}\r\n\r\ndata: line1\ndata: line2\n\n";
        let whole = all(&[body]);
        assert_eq!(whole.len(), 3);
        assert_eq!(whole[0], SseEvent { event: Some("message_start".into()), data: "{\"a\":1}".into() });
        assert_eq!(whole[1].event.as_deref(), Some("ping"));
        assert_eq!(whole[2], SseEvent { event: None, data: "line1\nline2".into() });

        // Byte by byte gives the same events, including a split `\r\n`.
        let bytes: Vec<&[u8]> = body.chunks(1).collect();
        assert_eq!(all(&bytes), whole);
    }

    #[test]
    fn keeps_utf8_split_across_chunks() {
        let body = "data: héllo ✓\n\n".as_bytes();
        let (a, b) = body.split_at(8);
        assert_eq!(all(&[a, b])[0].data, "héllo ✓");
    }

    #[test]
    fn dispatches_an_unterminated_last_event() {
        assert_eq!(all(&[b"event: x\ndata: tail"]), vec![SseEvent { event: Some("x".into()), data: "tail".into() }]);
    }

    #[test]
    fn ignores_events_without_data() {
        assert!(all(&[b"event: x\n\n"]).is_empty());
    }
}
