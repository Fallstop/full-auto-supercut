use crate::asr::Asr;
use crate::audio;
use crate::library::Recording;
use crate::project::{Event, Log, info};
use crate::text::{Word, crosses_sentence, norm, tokens, tokens_exact};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hit {
    pub id: String,
    /// How many back-to-back occurrences this clip covers ("this guy, this guy" = 2).
    pub n: usize,
    pub s: f64,
    pub e: f64,
    pub ctx: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub heard: Option<String>,
    #[serde(default)]
    pub drop: bool,
}

/// Clip window for `words[i..j]`: padded by `pre`/`post`, but never bleeding far into neighbours.
pub fn window(words: &[Word], i: usize, j: usize, pre: f64, post: f64) -> (f64, f64) {
    let mut s = words[i].s - pre;
    let mut e = words[j - 1].e + post;
    if i > 0 {
        s = s.max((words[i].s - 0.03).min(words[i - 1].e + 0.02));
    }
    if j < words.len() {
        e = e.min((words[j - 1].e + 0.05).max(words[j].s - 0.02));
    }
    (s.max(0.0), e)
}

/// Every occurrence of `phrase`, padded by `pre`/`post` seconds but never bleeding far into the
/// neighbouring words. Overlapping occurrences are merged into one clip.
pub fn find(phrase: &str, id: &str, words: &[Word], pre: f64, post: f64) -> Vec<Hit> {
    let toks = tokens(phrase);
    let n: Vec<String> = words.iter().map(|w| norm(&w.w)).collect();
    let mut hits: Vec<Hit> = Vec::new();
    if toks.is_empty() || n.len() < toks.len() {
        return hits;
    }
    for i in 0..=n.len() - toks.len() {
        if n[i..i + toks.len()] != toks[..] {
            continue;
        }
        let j = i + toks.len();
        let raw: Vec<&str> = words[i..j].iter().map(|w| w.w.as_str()).collect();
        if crosses_sentence(&raw) {
            continue;
        }
        let (s, e) = window(words, i, j, pre, post);
        if let Some(last) = hits.last_mut().filter(|h| s < h.e) {
            last.e = e;
            last.n += 1;
            continue;
        }
        let ctx = words[i.saturating_sub(6)..(j + 6).min(words.len())]
            .iter()
            .map(|w| w.w.trim())
            .collect::<Vec<_>>()
            .join(" ");
        hits.push(Hit {
            id: id.into(),
            n: 1,
            s,
            e,
            ctx,
            heard: None,
            drop: false,
        });
    }
    hits
}

/// Where the phrase sits in a clip transcript: (words before its first occurrence, words after
/// its last occurrence), or None if it isn't there.
fn locate(heard: &[String], phrase: &[String]) -> Option<(usize, usize)> {
    let at: Vec<usize> = (0..=heard.len().saturating_sub(phrase.len()))
        .filter(|&i| heard.get(i..i + phrase.len()) == Some(phrase))
        .collect();
    Some((*at.first()?, heard.len() - at.last()? - phrase.len()))
}

const MAX_TRIES: usize = 8;

/// Re-transcribe each clip window on its own and steer the window until whisper hears exactly the
/// phrase: stray words after it pull the end in, stray words before it push the start later, and a
/// phrase that isn't (fully) heard widens the window. A window with one stray word is only used if
/// no clean window turns up (and never when `strict`); if nothing passes, the hit is dropped.
pub fn verify_one(
    asr: &Asr,
    video: &Path,
    want: &[String],
    h: &mut Hit,
    strict: bool,
) -> Result<bool> {
    // Long internal pauses ("right? ... so") drag, so cap the clip relative to the phrase length.
    let max_dur = h.n as f64 * (0.5 + 0.55 * want.len() as f64) + 0.3 * (h.n - 1) as f64;
    let (s0, e0) = (h.s, h.e);
    let (mut s, mut e) = (s0, e0);
    let mut last = String::new();
    let mut best: Option<(f64, f64, String)> = None;
    let mut seen = std::collections::HashSet::new();
    for _ in 0..MAX_TRIES {
        s = s.clamp((s0 - 0.4).max(0.0), s0 + 0.35);
        e = e.clamp(e0 - 0.5, e0 + 0.45);
        if e - s < 0.15 || e - s > max_dur || !seen.insert(((s * 100.0) as i64, (e * 100.0) as i64))
        {
            break;
        }
        let mut pcm = vec![0.0f32; 8000];
        pcm.extend(audio::decode(video, Some((s, e - s)))?);
        pcm.extend(std::iter::repeat_n(0.0, 8000));
        last = asr.text(&pcm)?;
        // Strict (sentence) mode matches exact words; supercuts fold "guy's" into "guy".
        let heard = if strict {
            tokens_exact(&last)
        } else {
            tokens(&last)
        };
        let Some((before, after)) = locate(&heard, want) else {
            (s, e) = (s - 0.1, e + 0.12);
            continue;
        };
        let extra = heard.iter().filter(|t| !want.contains(t)).count();
        let fits = heard.len() <= (h.n + 1) * want.len() + 1;
        if extra == 0 && fits {
            best = Some((s, e, last.clone()));
            break;
        }
        if !strict && extra <= 1 && fits && best.is_none() {
            best = Some((s, e, last.clone()));
        }
        if before == 0 && after == 0 {
            break; // stray words are in the middle; trimming won't help
        }
        s += 0.08 * before.min(3) as f64;
        e -= 0.1 * after.min(3) as f64;
    }
    let ok = best.is_some();
    if let Some((s, e, heard)) = best {
        (h.s, h.e, last) = (s, e, heard);
    }
    h.heard = Some(last.trim().to_string());
    h.drop = !ok;
    Ok(ok)
}

pub fn verify(
    asr: &Asr,
    recs: &[Recording],
    phrase: &str,
    hits: &mut [Hit],
    log: Log,
) -> Result<usize> {
    let want = tokens(phrase);
    let mut kept = 0;
    let total = hits.len();
    for (i, h) in hits.iter_mut().enumerate() {
        let rec = recs
            .iter()
            .find(|r| r.id == h.id)
            .expect("hit for unknown recording");
        let ok = verify_one(asr, &rec.video, &want, h, false)?;
        kept += ok as usize;
        info(
            log,
            format!(
                "{} {} {} {:>8.2}s  {:?}",
                if ok { "ok  " } else { "drop" },
                rec.label,
                h.id,
                h.s,
                h.heard.as_deref().unwrap_or("")
            ),
        );
        log(Event::Progress(i + 1, total));
    }
    Ok(kept)
}
