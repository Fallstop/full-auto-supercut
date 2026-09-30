use crate::asr::Asr;
use crate::audio;
use crate::library::Recording;
use crate::text::{Word, norm, tokens};
use anyhow::Result;
use serde::{Deserialize, Serialize};

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
        let mut s = words[i].s - pre;
        let mut e = words[j - 1].e + post;
        if i > 0 {
            s = s.max((words[i].s - 0.03).min(words[i - 1].e + 0.02));
        }
        if j < words.len() {
            e = e.min((words[j - 1].e + 0.05).max(words[j].s - 0.02));
        }
        let s = s.max(0.0);
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

/// Padding adjustments (start, end) tried in order until the clip sounds right.
const TRIES: &[(f64, f64)] = &[
    (0.0, 0.0),
    (-0.1, 0.1),
    (0.06, 0.0),
    (0.0, -0.06),
    (-0.2, 0.25),
    (0.1, 0.05),
    (0.16, 0.0),
    (-0.3, 0.35),
];

/// If a clip transcript contains the phrase and (almost) nothing else, how many stray words it has.
fn stray_words(heard: &[String], phrase: &[String], n: usize, dur: f64) -> Option<usize> {
    let found = heard.windows(phrase.len()).any(|w| w == phrase);
    let extra = heard.iter().filter(|t| !phrase.contains(t)).count();
    let fits = heard.len() <= (n + 1) * phrase.len() + 1 && dur <= 2.5 + 1.5 * n as f64;
    (found && extra <= 1 && fits).then_some(extra)
}

/// Re-transcribe each clip window on its own and nudge the padding until whisper hears exactly the
/// phrase. A window with one stray word ("it's some sort of") is only used if no window is clean;
/// if nothing passes, the hit is dropped. Catches mistimed words and mis-recognitions.
pub fn verify(
    asr: &Asr,
    recs: &[Recording],
    phrase: &str,
    hits: &mut [Hit],
    log: bool,
) -> Result<usize> {
    let want = tokens(phrase);
    let mut kept = 0;
    for h in hits.iter_mut() {
        let rec = recs
            .iter()
            .find(|r| r.id == h.id)
            .expect("hit for unknown recording");
        let mut last = String::new();
        let mut best: Option<(f64, f64, String)> = None;
        for &(ds, de) in TRIES {
            let (s, e) = ((h.s + ds).max(0.0), h.e + de);
            if e - s < 0.1 {
                continue;
            }
            let mut pcm = vec![0.0f32; 8000];
            pcm.extend(audio::decode(&rec.video, Some((s, e - s)))?);
            pcm.extend(std::iter::repeat_n(0.0, 8000));
            last = asr.text(&pcm)?;
            match stray_words(&tokens(&last), &want, h.n, e - s) {
                Some(0) => {
                    best = Some((s, e, last.clone()));
                    break;
                }
                Some(_) if best.is_none() => best = Some((s, e, last.clone())),
                _ => {}
            }
        }
        let ok = best.is_some();
        if let Some((s, e, heard)) = best {
            (h.s, h.e, last) = (s, e, heard);
        }
        h.heard = Some(last.trim().to_string());
        h.drop = !ok;
        kept += ok as usize;
        if log {
            eprintln!(
                "{} {} {:>8.2}s  {:?}",
                if ok { "ok  " } else { "drop" },
                h.id,
                h.s,
                h.heard.as_deref().unwrap_or("")
            );
        }
    }
    Ok(kept)
}
