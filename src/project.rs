//! A folder of recordings plus its cache: transcripts, the whisper model, mining and cutting.
//! Long operations report through a [`Log`] callback so the CLI can print and the TUI can draw.

use crate::asr::Asr;
use crate::library::{self, Recording};
use crate::text::{self, Word};
use crate::{audio, hits, mine, render};
use anyhow::{Context, Result, bail};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

const MODEL_URL: &str =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin";

pub enum Event {
    Info(String),
    /// Work item `done` of `total` finished.
    Progress(usize, usize),
}

pub type Log<'a> = &'a (dyn Fn(Event) + Sync);

pub fn info(log: Log, msg: impl Into<String>) {
    log(Event::Info(msg.into()));
}

pub struct Project {
    pub dir: PathBuf,
    pub work: PathBuf,
    pub model: Option<PathBuf>,
    pub recs: Vec<Recording>,
    asr: Mutex<Option<Arc<Asr>>>,
}

pub struct CutOpts {
    pub out: PathBuf,
    pub pre: f64,
    pub post: f64,
    pub verify: bool,
    pub max_clips: usize,
    pub subtitle: Option<String>,
    pub font: Option<PathBuf>,
}

impl Default for CutOpts {
    fn default() -> Self {
        Self {
            out: PathBuf::from("."),
            pre: 0.12,
            post: 0.2,
            verify: true,
            max_clips: 300,
            subtitle: None,
            font: None,
        }
    }
}

impl Project {
    pub fn open(dir: &Path, work: Option<PathBuf>, model: Option<PathBuf>) -> Result<Self> {
        let recs = library::scan(dir)?;
        if recs.is_empty() {
            bail!("no videos found in {}", dir.display());
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            work: work.unwrap_or_else(|| dir.join(".supercut")),
            model,
            recs,
            asr: Mutex::new(None),
        })
    }

    pub fn words_dir(&self) -> PathBuf {
        self.work.join("words")
    }

    pub fn folder_name(&self) -> String {
        let dir = std::fs::canonicalize(&self.dir).unwrap_or(self.dir.clone());
        dir.file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default()
    }

    /// The whisper model, loaded once (and downloaded on first use if it's the default).
    pub fn asr(&self, log: Log) -> Result<Arc<Asr>> {
        let mut slot = self.asr.lock().unwrap();
        if let Some(a) = &*slot {
            return Ok(a.clone());
        }
        let model = match &self.model {
            Some(m) => m.clone(),
            None => {
                let home = std::env::var("HOME").context("HOME not set")?;
                let p =
                    PathBuf::from(home).join(".cache/full-auto-supercut/ggml-large-v3-turbo.bin");
                if !p.exists() {
                    info(
                        log,
                        format!(
                            "downloading whisper large-v3-turbo (1.6 GB) to {}",
                            p.display()
                        ),
                    );
                    std::fs::create_dir_all(p.parent().unwrap())?;
                    let part = p.with_extension("part");
                    let status = std::process::Command::new("curl")
                        .args(["-fsSL", "-o"])
                        .arg(&part)
                        .arg(MODEL_URL)
                        .status()?;
                    if !status.success() {
                        bail!("model download failed");
                    }
                    std::fs::rename(part, &p)?;
                }
                p
            }
        };
        info(
            log,
            format!(
                "loading {}",
                model.file_name().unwrap_or_default().to_string_lossy()
            ),
        );
        let asr = Arc::new(Asr::new(&model)?);
        *slot = Some(asr.clone());
        Ok(asr)
    }

    /// Recordings that don't have a transcript yet.
    pub fn pending(&self) -> Vec<&Recording> {
        let dir = self.words_dir();
        self.recs
            .iter()
            .filter(|r| !text::words_path(&dir, &r.id).exists())
            .collect()
    }

    pub fn transcribe(&self, log: Log) -> Result<()> {
        let dir = self.words_dir();
        std::fs::create_dir_all(&dir)?;
        let todo = self.pending();
        if todo.is_empty() {
            return Ok(());
        }
        let asr = self.asr(log)?;
        for (i, r) in todo.iter().enumerate() {
            let t = Instant::now();
            let pcm = audio::decode(&r.video, None)?;
            let words = if audio::is_silent(&pcm) {
                info(
                    log,
                    format!(
                        "[{}/{}] {} ({}): no speech, skipping",
                        i + 1,
                        todo.len(),
                        r.label,
                        r.id
                    ),
                );
                Vec::new()
            } else {
                text::drop_loops(asr.words(&pcm, 0.0, false)?)
            };
            let dst = text::words_path(&dir, &r.id);
            std::fs::write(dst.with_extension("tmp"), serde_json::to_string(&words)?)?;
            std::fs::rename(dst.with_extension("tmp"), dst)?;
            info(
                log,
                format!(
                    "[{}/{}] {} ({}): {} words in {:.0}s",
                    i + 1,
                    todo.len(),
                    r.label,
                    r.id,
                    words.len(),
                    t.elapsed().as_secs_f64()
                ),
            );
            log(Event::Progress(i + 1, todo.len()));
        }
        Ok(())
    }

    /// Every transcribed recording with its words.
    pub fn docs(&self) -> Result<Vec<(Recording, Vec<Word>)>> {
        let mut out = Vec::new();
        for r in &self.recs {
            if let Some(w) = text::load_words(&self.words_dir(), &r.id)? {
                out.push((r.clone(), w));
            }
        }
        Ok(out)
    }

    pub fn candidates(&self) -> Result<Vec<mine::Candidate>> {
        // Near-empty transcripts (dead mics, a few stray words) would distort per-recording rates.
        let texts: Vec<String> = self
            .docs()?
            .iter()
            .filter(|(_, w)| w.len() >= 500)
            .map(|(_, w)| w.iter().map(|x| x.w.as_str()).collect())
            .collect();
        Ok(mine::mine(&mine::docs_from_text(&texts), 4, 15))
    }

    /// Supercut every occurrence of `phrase`; returns the styled and clean output paths.
    pub fn cut(&self, phrase: &str, opts: &CutOpts, log: Log) -> Result<Vec<PathBuf>> {
        let mut hits: Vec<hits::Hit> = self
            .docs()?
            .iter()
            .flat_map(|(r, w)| hits::find(phrase, &r.id, w, opts.pre, opts.post))
            .collect();
        let found: usize = hits.iter().map(|h| h.n).sum();
        info(
            log,
            format!(
                "\u{201c}{phrase}\u{201d}: {found} occurrences in {} clips",
                hits.len()
            ),
        );
        if hits.is_empty() {
            bail!("\u{201c}{phrase}\u{201d} isn't in any transcript");
        }
        if hits.len() > opts.max_clips {
            let step = hits.len() as f64 / opts.max_clips as f64;
            hits = (0..opts.max_clips)
                .map(|i| hits[(i as f64 * step) as usize].clone())
                .collect();
            info(log, format!("sampled down to {} clips", hits.len()));
        }
        let slug = text::tokens(phrase).join("_");
        let work = self.work.join(format!("cut_{slug}"));
        std::fs::create_dir_all(&work)?;
        if opts.verify {
            let asr = self.asr(log)?;
            let kept = hits::verify(&asr, &self.recs, phrase, &mut hits, log)?;
            info(log, format!("verified {kept}/{} clips", hits.len()));
        }
        std::fs::write(work.join("hits.json"), serde_json::to_string_pretty(&hits)?)?;

        let font = opts.font.clone().unwrap_or_else(render::default_font);
        let subtitle = opts.subtitle.clone().unwrap_or_else(|| self.folder_name());
        let title = format!("\u{201c}{}\u{201d}", phrase.to_uppercase());
        std::fs::create_dir_all(&opts.out)?;
        let mut outs = Vec::new();
        for clean in [false, true] {
            info(
                log,
                format!("rendering {} cut", if clean { "clean" } else { "styled" }),
            );
            let out = opts.out.join(format!(
                "{slug}_supercut{}.mp4",
                if clean { "_clean" } else { "" }
            ));
            let clips = render::supercut_clips(&hits, &self.recs, clean);
            let total: usize = hits.iter().filter(|h| !h.drop).map(|h| h.n).sum();
            let n_recs = hits
                .iter()
                .filter(|h| !h.drop)
                .map(|h| &h.id)
                .collect::<std::collections::HashSet<_>>()
                .len();
            let cards = (!clean).then(|| render::Cards {
                title: vec![
                    (title.clone(), 110, "white", -90),
                    ("a supercut".into(), 40, "0xffcc33", 40),
                    (subtitle.clone(), 30, "0x999999", 110),
                ],
                end: vec![
                    (total.to_string(), 160, "0xffcc33", -120),
                    ("times.".into(), 60, "white", 60),
                    (
                        format!(
                            "across {n_recs} recording{}",
                            if n_recs == 1 { "" } else { "s" }
                        ),
                        32,
                        "0x999999",
                        140,
                    ),
                ],
            });
            let dir = work.join(if clean { "clips_clean" } else { "clips" });
            render::render(&clips, &font, cards.as_ref(), &dir, &out, log)?;
            info(log, format!("wrote {}", out.display()));
            outs.push(out);
        }
        Ok(outs)
    }
}
