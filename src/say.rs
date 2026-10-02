//! "Say anything": build an arbitrary sentence out of words spoken somewhere in the recordings.
//!
//! The sentence is covered with the fewest, longest runs of words that were actually spoken in a
//! row (a real "you know what" sounds far better than three spliced words). For each run, every
//! take in the recordings is scored for how cleanly it can be cut out, the best is verified by
//! re-transcribing it, and the takes are levelled and joined.

use crate::asr::Asr;
use crate::audio;
use crate::hits::{self, Hit};
use crate::library::Recording;
use crate::project::{Event, Log, Project, info};
use crate::render::{self, Clip};
use crate::text::{Word, crosses_sentence, norm_exact};
use anyhow::{Result, bail};
use std::collections::HashMap;
use std::path::PathBuf;

/// Longest run of consecutive words looked up as one chunk.
const MAX_CHUNK: usize = 8;
/// Takes kept per chunk (ranked best first).
const MAX_TAKES: usize = 24;
/// Breathing room between pieces so spliced words don't run together.
const PAUSE: f64 = 0.08;
/// Takes tried (in rank order) before settling for an unverified one.
const MAX_ATTEMPTS: usize = 6;

pub struct Index {
    pub docs: Vec<(Recording, Vec<Word>)>,
    norm: Vec<Vec<String>>,
    postings: HashMap<String, Vec<(u32, u32)>>,
}

#[derive(Clone, Debug)]
pub struct Take {
    pub doc: usize,
    /// Index of the chunk's first word in that recording's transcript.
    pub start: usize,
    pub score: f64,
}

#[derive(Clone, Debug)]
pub struct Chunk {
    /// Normalised words in this chunk.
    pub toks: Vec<String>,
    /// The words as the user typed them (for captions).
    pub typed: String,
    /// Range of sentence-word indices covered.
    pub span: (usize, usize),
    /// Total number of takes found (may exceed `takes.len()`).
    pub found: usize,
    pub takes: Vec<Take>,
    /// Which take is selected.
    pub choice: usize,
}

impl Chunk {
    pub fn missing(&self) -> bool {
        self.takes.is_empty()
    }
}

#[derive(Clone, Debug, Default)]
pub struct Plan {
    pub chunks: Vec<Chunk>,
}

impl Plan {
    pub fn missing_words(&self) -> Vec<&str> {
        self.chunks
            .iter()
            .filter(|c| c.missing())
            .map(|c| c.typed.as_str())
            .collect()
    }
}

impl Index {
    pub fn build(docs: Vec<(Recording, Vec<Word>)>) -> Self {
        let norm: Vec<Vec<String>> = docs
            .iter()
            .map(|(_, w)| w.iter().map(|x| norm_exact(&x.w)).collect())
            .collect();
        let mut postings: HashMap<String, Vec<(u32, u32)>> = HashMap::new();
        for (d, toks) in norm.iter().enumerate() {
            for (i, t) in toks.iter().enumerate() {
                postings
                    .entry(t.clone())
                    .or_default()
                    .push((d as u32, i as u32));
            }
        }
        Self {
            docs,
            norm,
            postings,
        }
    }

    pub fn words(&self) -> usize {
        self.norm.iter().map(Vec::len).sum()
    }

    /// Words that *were* said and are spelled like `word` (for "did you mean").
    pub fn suggest(&self, word: &str, k: usize) -> Vec<String> {
        let w: Vec<char> = word.chars().collect();
        let max = if w.len() >= 5 { 2 } else { 1 };
        let mut near: Vec<(usize, usize, &String)> = self
            .postings
            .iter()
            .filter(|(v, _)| v.chars().count().abs_diff(w.len()) <= max)
            .filter_map(|(v, p)| {
                let d = levenshtein(&w, &v.chars().collect::<Vec<_>>());
                (d <= max).then_some((d, usize::MAX - p.len(), v))
            })
            .collect();
        near.sort();
        near.into_iter()
            .take(k)
            .map(|(_, _, v)| v.clone())
            .collect()
    }

    /// How many times `phrase` was spoken as one run.
    pub fn count(&self, phrase: &str) -> usize {
        let toks = crate::text::tokens_exact(phrase);
        if toks.is_empty() {
            0
        } else {
            self.occurrences(&toks, usize::MAX).len()
        }
    }

    /// Where `toks` was spoken as one run (not across a sentence break), up to `limit` places.
    fn occurrences(&self, toks: &[String], limit: usize) -> Vec<(usize, usize)> {
        // Anchor on the rarest word so common ones ("the") don't make lookups slow.
        let Some((k, anchor)) = toks
            .iter()
            .enumerate()
            .filter_map(|(k, t)| self.postings.get(t).map(|p| (k, p)))
            .min_by_key(|(_, p)| p.len())
        else {
            return Vec::new();
        };
        if toks.iter().any(|t| !self.postings.contains_key(t)) {
            return Vec::new();
        }
        let mut out = Vec::new();
        for &(d, i) in anchor {
            let (d, i) = (d as usize, i as usize);
            let Some(start) = i.checked_sub(k) else {
                continue;
            };
            let doc = &self.norm[d];
            if start + toks.len() > doc.len() || doc[start..start + toks.len()] != *toks {
                continue;
            }
            let raw: Vec<&str> = self.docs[d].1[start..start + toks.len()]
                .iter()
                .map(|w| w.w.as_str())
                .collect();
            if toks.len() > 1 && crosses_sentence(&raw) {
                continue;
            }
            out.push((d, start));
            if out.len() >= limit {
                break;
            }
        }
        out
    }

    /// How cleanly a take can be cut out: confident recognition, a pause on either side (so the
    /// cut doesn't clip a neighbouring word), and a plausible speaking rate.
    fn score(&self, doc: usize, start: usize, len: usize) -> f64 {
        let w = &self.docs[doc].1;
        let run = &w[start..start + len];
        let p = run.iter().map(|x| x.p as f64).sum::<f64>() / len as f64;
        let gap_before = if start == 0 {
            0.3
        } else {
            (run[0].s - w[start - 1].e).clamp(0.0, 0.3)
        };
        let gap_after = w
            .get(start + len)
            .map_or(0.3, |n| (n.s - run[len - 1].e).clamp(0.0, 0.3));
        let per_word = (run[len - 1].e - run[0].s) / len as f64;
        let rate_penalty = if per_word < 0.12 {
            0.12 - per_word
        } else if per_word > 0.8 {
            per_word - 0.8
        } else {
            0.0
        };
        p + 1.5 * (gap_before + gap_after) - 3.0 * rate_penalty
    }

    pub fn plan(&self, sentence: &str) -> Plan {
        self.plan_capped(sentence, MAX_CHUNK)
    }

    /// Like [`Index::plan`], but with pieces at most `max_chunk` words long.
    pub fn plan_capped(&self, sentence: &str, max_chunk: usize) -> Plan {
        let typed: Vec<&str> = sentence
            .split_whitespace()
            .filter(|w| !norm_exact(w).is_empty())
            .collect();
        let toks: Vec<String> = typed.iter().map(|w| norm_exact(w)).collect();
        let m = toks.len();
        // best[i] = (chunks used to cover toks[..i], -log takes, previous cut point)
        let mut best: Vec<Option<(usize, f64, usize)>> = vec![None; m + 1];
        best[0] = Some((0, 0.0, 0));
        for i in 0..m {
            let Some((chunks, cost, _)) = best[i] else {
                continue;
            };
            for len in 1..=max_chunk.max(1).min(m - i) {
                let n = self.occurrences(&toks[i..i + len], 200).len();
                if n == 0 && len > 1 {
                    break; // a longer run can't exist if this one doesn't
                }
                // Fewest chunks wins; among equals prefer chunks with more takes to choose from.
                let cand = (chunks + 1, cost - ((n.max(1)) as f64).ln(), i);
                let better = match best[i + len] {
                    None => true,
                    Some((c, k, _)) => (cand.0, cand.1) < (c, k),
                };
                if better {
                    best[i + len] = Some(cand);
                }
            }
        }
        let mut cuts = vec![m];
        let mut i = m;
        while i > 0 {
            i = best[i]
                .expect("every prefix is reachable via single words")
                .2;
            cuts.push(i);
        }
        cuts.reverse();

        let mut chunks: Vec<Chunk> = Vec::new();
        for w in cuts.windows(2) {
            let (a, b) = (w[0], w[1]);
            let occ = self.occurrences(&toks[a..b], 5000);
            let mut takes: Vec<Take> = occ
                .iter()
                .map(|&(doc, start)| Take {
                    doc,
                    start,
                    score: self.score(doc, start, b - a),
                })
                .collect();
            takes.sort_by(|x, y| y.score.total_cmp(&x.score));
            takes.truncate(MAX_TAKES);
            // Keep the voice consistent: prefer a near-best take from the previous chunk's recording.
            let prev_doc = chunks
                .last()
                .and_then(|c| c.takes.get(c.choice))
                .map(|t| t.doc);
            let top = takes.first().map_or(0.0, |t| t.score);
            let choice = prev_doc
                .and_then(|pd| {
                    takes
                        .iter()
                        .position(|t| t.doc == pd && t.score > top - 0.2)
                })
                .unwrap_or(0);
            chunks.push(Chunk {
                toks: toks[a..b].to_vec(),
                typed: typed[a..b].join(" "),
                span: (a, b),
                found: occ.len(),
                takes,
                choice,
            });
        }
        Plan { chunks }
    }

    pub fn take_window(&self, chunk: &Chunk, take: &Take) -> (f64, f64) {
        hits::window(
            &self.docs[take.doc].1,
            take.start,
            take.start + chunk.toks.len(),
            0.06,
            0.1,
        )
    }

    pub fn take_context(&self, chunk: &Chunk, take: &Take) -> String {
        let w = &self.docs[take.doc].1;
        let (a, b) = (take.start, take.start + chunk.toks.len());
        let side = |r: &[Word]| r.iter().map(|x| x.w.trim()).collect::<Vec<_>>().join(" ");
        format!(
            "…{} [{}] {}…",
            side(&w[a.saturating_sub(4)..a]),
            side(&w[a..b]),
            side(&w[b..(b + 4).min(w.len())])
        )
    }
}

fn levenshtein(a: &[char], b: &[char]) -> usize {
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    for (i, ca) in a.iter().enumerate() {
        let mut cur = vec![i + 1; b.len() + 1];
        for (j, cb) in b.iter().enumerate() {
            cur[j + 1] = (prev[j] + (ca != cb) as usize)
                .min(prev[j + 1] + 1)
                .min(cur[j] + 1);
        }
        prev = cur;
    }
    prev[b.len()]
}

pub struct SayOpts {
    pub out: PathBuf,
    pub captions: bool,
    pub verify: bool,
}

/// Loudness of the louder half of 20 ms frames, in dBFS (ignores the padding's silence).
fn speech_dbfs(pcm: &[f32]) -> f64 {
    let mut db: Vec<f64> = pcm
        .chunks(320)
        .map(|f| {
            10.0 * (f.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / f.len() as f64)
                .max(1e-12)
                .log10()
        })
        .collect();
    db.sort_by(|a, b| b.total_cmp(a));
    let top = &db[..db.len().div_ceil(2)];
    top.iter().sum::<f64>() / top.len().max(1) as f64
}

/// Find a clean, levelled clip for one piece. The chosen take is tried first, then the others in
/// rank order. If no take of a multi-word piece verifies, the piece is re-planned as smaller
/// pieces (which usually have many more takes to choose from) before settling for a best guess.
fn cut_chunk(
    index: &Index,
    asr: Option<&Asr>,
    chunk: &Chunk,
    captions: bool,
    log: Log,
    depth: usize,
) -> Result<Vec<Clip>> {
    let order: Vec<&Take> = std::iter::once(&chunk.takes[chunk.choice])
        .chain(
            chunk
                .takes
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != chunk.choice)
                .map(|(_, t)| t),
        )
        .take(if asr.is_some() { MAX_ATTEMPTS } else { 1 })
        .collect();
    let mut picked: Option<(&Take, f64, f64)> = None;
    for take in &order {
        let (s, e) = index.take_window(chunk, take);
        let Some(asr) = asr else {
            picked = Some((take, s, e));
            break;
        };
        let rec = &index.docs[take.doc].0;
        let mut h = Hit {
            id: rec.id.clone(),
            n: 1,
            s,
            e,
            ctx: String::new(),
            heard: None,
            drop: false,
        };
        if hits::verify_one(asr, &rec.video, &chunk.toks, &mut h, true)? {
            // Verification converges on the tightest clean window, which can shave the tail off
            // the last word. Keep a little more on each side if it still sounds clean.
            let (s, e) = (h.s, h.e);
            let mut wide = Hit {
                s: s - 0.03,
                e: e + 0.08,
                ..h.clone()
            };
            let keep =
                hits::verify_one(asr, &rec.video, &chunk.toks, &mut wide, true)? && wide.e > e;
            picked = Some(if keep {
                (take, wide.s, wide.e)
            } else {
                (take, s, e)
            });
            break;
        }
        info(
            log,
            format!(
                "  \u{201c}{}\u{201d}: {} take sounded like {:?}, trying another",
                chunk.typed,
                rec.label,
                h.heard.unwrap_or_default()
            ),
        );
    }
    if picked.is_none() && chunk.toks.len() > 1 && depth < 3 {
        let sub = index.plan_capped(&chunk.typed, chunk.toks.len() - 1);
        if sub.missing_words().is_empty() {
            info(
                log,
                format!(
                    "  \u{201c}{}\u{201d}: no clean take, splitting it into {} pieces",
                    chunk.typed,
                    sub.chunks.len()
                ),
            );
            let mut clips = Vec::new();
            for c in &sub.chunks {
                clips.extend(cut_chunk(
                    index,
                    Some(asr.unwrap()),
                    c,
                    captions,
                    log,
                    depth + 1,
                )?);
            }
            return Ok(clips);
        }
    }
    let (take, s, e) = picked.unwrap_or_else(|| {
        let t = order[0];
        let (s, e) = index.take_window(chunk, t);
        info(
            log,
            format!(
                "  \u{201c}{}\u{201d}: no take came out clean, using the best guess",
                chunk.typed
            ),
        );
        (t, s, e)
    });
    let rec = &index.docs[take.doc].0;
    let gain =
        (-20.0 - speech_dbfs(&audio::decode(&rec.video, Some((s, e - s)))?)).clamp(-12.0, 18.0);
    info(
        log,
        format!(
            "\u{201c}{}\u{201d} \u{2190} {} {:.1}s",
            chunk.typed, rec.label, s
        ),
    );
    Ok(vec![Clip {
        video: rec.video.clone(),
        s,
        e,
        gain_db: gain,
        tag: None,
        label: None,
        caption: captions.then(|| chunk.typed.clone()),
        pause: PAUSE,
    }])
}

/// Verify, level and join the planned takes into one video. Returns the output path.
pub fn build(
    project: &Project,
    index: &Index,
    plan: &Plan,
    opts: &SayOpts,
    log: Log,
) -> Result<PathBuf> {
    let missing = plan.missing_words();
    if !missing.is_empty() {
        bail!("never said in these recordings: {}", missing.join(", "));
    }
    if plan.chunks.is_empty() {
        bail!("type a sentence first");
    }
    let asr = if opts.verify {
        Some(project.asr(log)?)
    } else {
        None
    };
    let mut clips = Vec::new();
    for (k, chunk) in plan.chunks.iter().enumerate() {
        clips.extend(cut_chunk(
            index,
            asr.as_deref(),
            chunk,
            opts.captions,
            log,
            0,
        )?);
        log(Event::Progress(k + 1, plan.chunks.len() + 1));
    }
    let words: Vec<String> = plan.chunks.iter().flat_map(|c| c.toks.clone()).collect();
    let slug: String = words.iter().take(8).cloned().collect::<Vec<_>>().join("_");
    std::fs::create_dir_all(&opts.out)?;
    let out = opts.out.join(format!("say_{slug}.mp4"));
    let work = project.work.join("say").join(&slug);
    render::render(&clips, &render::default_font(), None, &work, &out, &|_| {})?;
    log(Event::Progress(
        plan.chunks.len() + 1,
        plan.chunks.len() + 1,
    ));
    info(log, format!("wrote {}", out.display()));
    Ok(out)
}

/// Open a video (or part of one) in whatever player is installed, without blocking.
pub fn play(path: &std::path::Path, range: Option<(f64, f64)>) -> Result<()> {
    use std::process::{Command, Stdio};
    let quiet = |c: &mut Command| {
        c.stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
    };
    if which("mpv") {
        let mut c = Command::new("mpv");
        if let Some((s, e)) = range {
            c.arg(format!("--start={s:.2}"))
                .arg(format!("--end={e:.2}"));
        }
        quiet(c.arg("--force-window=yes").arg(path));
        c.spawn()?;
    } else if which("ffplay") {
        let mut c = Command::new("ffplay");
        c.args([
            "-autoexit",
            "-loglevel",
            "quiet",
            "-window_title",
            "full-auto-supercut",
        ]);
        if let Some((s, e)) = range {
            c.args(["-ss", &format!("{s:.2}"), "-t", &format!("{:.2}", e - s)]);
        }
        quiet(c.arg(path));
        c.spawn()?;
    } else {
        let mut c = Command::new("xdg-open");
        quiet(c.arg(path));
        c.spawn()?;
    }
    Ok(())
}

fn which(bin: &str) -> bool {
    std::env::var_os("PATH")
        .is_some_and(|p| std::env::split_paths(&p).any(|d| d.join(bin).is_file()))
}

/// For the CLI: plan and print, then build.
pub fn describe(index: &Index, plan: &Plan) -> String {
    plan.chunks
        .iter()
        .map(|c| match c.takes.get(c.choice) {
            None => format!("  \u{2717} \u{201c}{}\u{201d}: never said", c.typed),
            Some(t) => format!(
                "  \u{2713} \u{201c}{}\u{201d}: {} take{}, using {} {}",
                c.typed,
                c.found,
                if c.found == 1 { "" } else { "s" },
                index.docs[t.doc].0.label,
                index.take_context(c, t)
            ),
        })
        .collect::<Vec<_>>()
        .join("\n")
}
