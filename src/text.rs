use serde::{Deserialize, Serialize};
use std::path::Path;

/// One transcribed word with timestamps in seconds. Field names match the
/// faster-whisper JSON this tool also accepts (`w`, `s`, `e`, `p`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Word {
    pub w: String,
    pub s: f64,
    pub e: f64,
    #[serde(default)]
    pub p: f32,
}

const CONTRACTIONS: &[&str] = &[
    "it's", "that's", "let's", "what's", "there's", "here's", "he's", "she's", "who's", "where's",
];

/// Lowercase, strip punctuation, and fold possessive `'s` so "guy's" matches "guy"
/// (but leave contractions like "let's" alone).
pub fn norm(w: &str) -> String {
    let s: String = w
        .to_lowercase()
        .replace('\u{2019}', "'")
        .chars()
        .filter(|c| c.is_ascii_alphanumeric() || *c == '\'')
        .collect();
    if CONTRACTIONS.contains(&s.as_str()) {
        return s;
    }
    s.strip_suffix("'s").map(String::from).unwrap_or(s)
}

pub fn tokens(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(norm)
        .filter(|t| !t.is_empty())
        .collect()
}

pub fn words_path(dir: &Path, id: &str) -> std::path::PathBuf {
    dir.join(format!("{id}.json"))
}

pub fn load_words(dir: &Path, id: &str) -> anyhow::Result<Option<Vec<Word>>> {
    let p = words_path(dir, id);
    if !p.exists() {
        return Ok(None);
    }
    Ok(Some(serde_json::from_str(&std::fs::read_to_string(p)?)?))
}

/// Remove whisper repetition loops ("thank you thank you thank you ..." over a quiet stretch):
/// runs where a 2–4 word pattern repeats 6+ times back to back, or a single word 8+ times.
/// Real speech rarely does that; a genuine "yeah, yeah, yeah, yeah, yeah" survives.
pub fn drop_loops(words: Vec<Word>) -> Vec<Word> {
    let n: Vec<String> = words.iter().map(|w| norm(&w.w)).collect();
    let mut keep = vec![true; words.len()];
    let mut i = 0;
    while i < n.len() {
        let mut skipped = false;
        for len in 1..=4 {
            let min_reps = if len == 1 { 8 } else { 6 };
            let mut reps = 1;
            while i + (reps + 1) * len <= n.len()
                && n[i + reps * len..i + (reps + 1) * len] == n[i..i + len]
            {
                reps += 1;
            }
            if reps >= min_reps {
                keep[i..i + reps * len].iter_mut().for_each(|k| *k = false);
                i += reps * len;
                skipped = true;
                break;
            }
        }
        if !skipped {
            i += 1;
        }
    }
    words
        .into_iter()
        .zip(keep)
        .filter_map(|(w, k)| k.then_some(w))
        .collect()
}
