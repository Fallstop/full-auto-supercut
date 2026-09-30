mod asr;
mod audio;
mod hits;
mod library;
mod mine;
mod render;
mod text;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use library::Recording;
use std::path::PathBuf;
use std::time::Instant;

const MODEL_URL: &str =
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo.bin";

/// Point it at a folder of videos; get back a supercut of whatever someone keeps saying.
#[derive(Parser)]
#[command(version, about)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Transcribe, find the catchphrase, verify every clip, and render the supercut. No questions asked.
    FullAuto {
        #[command(flatten)]
        common: Common,
        /// Use the Nth-ranked candidate instead of the top one.
        #[arg(long, default_value_t = 1)]
        pick: usize,
        #[command(flatten)]
        cut: CutOpts,
    },
    /// Transcribe every video to word-timestamped JSON (skips ones already done).
    Transcribe {
        #[command(flatten)]
        common: Common,
    },
    /// Rank catchphrase candidates.
    Mine {
        #[command(flatten)]
        common: Common,
        /// Mine the platform captions in yt-dlp .info.json/.srt sidecars instead (no GPU needed).
        #[arg(long)]
        captions: bool,
        #[arg(long, default_value_t = 25)]
        top: usize,
    },
    /// Make a supercut of a specific phrase.
    Cut {
        #[command(flatten)]
        common: Common,
        phrase: String,
        #[command(flatten)]
        cut: CutOpts,
    },
}

#[derive(Args)]
struct Common {
    /// Folder of videos (mp4/mkv/webm/mov/m4v).
    dir: PathBuf,
    /// ggml whisper model. The default (large-v3-turbo) is downloaded on first use.
    #[arg(long, env = "SUPERCUT_MODEL")]
    model: Option<PathBuf>,
    /// Where transcripts and clips are cached [default: <dir>/.supercut].
    #[arg(long)]
    work: Option<PathBuf>,
}

#[derive(Args)]
struct CutOpts {
    /// Output folder for the rendered videos.
    #[arg(short, long, default_value = ".")]
    out: PathBuf,
    /// Seconds of padding before each occurrence.
    #[arg(long, default_value_t = 0.12)]
    pre: f64,
    /// Seconds of padding after each occurrence.
    #[arg(long, default_value_t = 0.2)]
    post: f64,
    /// Skip re-transcribing each clip to check it.
    #[arg(long)]
    no_verify: bool,
    /// Evenly sample down to at most this many clips.
    #[arg(long, default_value_t = 300)]
    max_clips: usize,
    /// Subtitle on the title card [default: folder name].
    #[arg(long)]
    subtitle: Option<String>,
    /// Font for cards and overlays [default: fc-match "sans:black"].
    #[arg(long)]
    font: Option<PathBuf>,
}

impl Common {
    fn work(&self) -> PathBuf {
        self.work
            .clone()
            .unwrap_or_else(|| self.dir.join(".supercut"))
    }
    fn words_dir(&self) -> PathBuf {
        self.work().join("words")
    }
    fn recordings(&self) -> Result<Vec<Recording>> {
        let recs = library::scan(&self.dir)?;
        if recs.is_empty() {
            bail!("no videos found in {}", self.dir.display());
        }
        Ok(recs)
    }
    fn asr(&self) -> Result<asr::Asr> {
        let model = match &self.model {
            Some(m) => m.clone(),
            None => {
                let home = std::env::var("HOME").context("HOME not set")?;
                let p =
                    PathBuf::from(home).join(".cache/full-auto-supercut/ggml-large-v3-turbo.bin");
                if !p.exists() {
                    eprintln!(
                        "downloading whisper large-v3-turbo (1.6 GB) to {}",
                        p.display()
                    );
                    std::fs::create_dir_all(p.parent().unwrap())?;
                    let part = p.with_extension("part");
                    let status = std::process::Command::new("curl")
                        .args(["-fL", "--progress-bar", "-o"])
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
        asr::Asr::new(&model)
    }
}

fn transcribe(common: &Common, recs: &[Recording]) -> Result<()> {
    let dir = common.words_dir();
    std::fs::create_dir_all(&dir)?;
    let todo: Vec<&Recording> = recs
        .iter()
        .filter(|r| !text::words_path(&dir, &r.id).exists())
        .collect();
    if todo.is_empty() {
        return Ok(());
    }
    let asr = common.asr()?;
    for (i, r) in todo.iter().enumerate() {
        let t = Instant::now();
        let pcm = audio::decode(&r.video, None)?;
        let words = if audio::is_silent(&pcm) {
            eprintln!(
                "[{}/{}] {} ({}): no speech, skipping",
                i + 1,
                todo.len(),
                r.label,
                r.id
            );
            Vec::new()
        } else {
            text::drop_loops(asr.words(&pcm, 0.0, false)?)
        };
        let dst = text::words_path(&dir, &r.id);
        std::fs::write(dst.with_extension("tmp"), serde_json::to_string(&words)?)?;
        std::fs::rename(dst.with_extension("tmp"), dst)?;
        eprintln!(
            "[{}/{}] {} ({}): {} words in {:.0}s",
            i + 1,
            todo.len(),
            r.label,
            r.id,
            words.len(),
            t.elapsed().as_secs_f64()
        );
    }
    Ok(())
}

fn load_docs(common: &Common, recs: &[Recording]) -> Result<Vec<(Recording, Vec<text::Word>)>> {
    let mut out = Vec::new();
    for r in recs {
        if let Some(w) = text::load_words(&common.words_dir(), &r.id)? {
            out.push((r.clone(), w));
        }
    }
    Ok(out)
}

fn transcript_texts(common: &Common, recs: &[Recording]) -> Result<Vec<String>> {
    // Near-empty transcripts (dead mics, a few stray words) would distort per-recording rates.
    Ok(load_docs(common, recs)?
        .iter()
        .filter(|(_, w)| w.len() >= 500)
        .map(|(_, w)| w.iter().map(|x| x.w.as_str()).collect())
        .collect())
}

fn print_candidates(cands: &[mine::Candidate], top: usize) {
    eprintln!(
        "{:>3}  {:<30} {:>6} {:>9} {:>6} {:>6}",
        "#", "phrase", "count", "habitual", "lift", "score"
    );
    for (i, c) in cands.iter().take(top).enumerate() {
        let habitual = format!("{}/{}", c.habitual, c.per_recording.len());
        eprintln!(
            "{:>3}  {:<30} {:>6} {:>9} {:>5.0}x {:>6.1}",
            i + 1,
            format!("\u{201c}{}\u{201d}", c.phrase),
            c.count,
            habitual,
            c.lift,
            c.score
        );
    }
}

fn cut(common: &Common, recs: &[Recording], phrase: &str, opts: &CutOpts) -> Result<()> {
    let docs = load_docs(common, recs)?;
    let mut hits: Vec<hits::Hit> = docs
        .iter()
        .flat_map(|(r, w)| hits::find(phrase, &r.id, w, opts.pre, opts.post))
        .collect();
    let found: usize = hits.iter().map(|h| h.n).sum();
    eprintln!(
        "\u{201c}{phrase}\u{201d}: {found} occurrences in {} clips",
        hits.len()
    );
    if hits.is_empty() {
        bail!("phrase not found in any transcript");
    }
    if hits.len() > opts.max_clips {
        let step = hits.len() as f64 / opts.max_clips as f64;
        hits = (0..opts.max_clips)
            .map(|i| hits[(i as f64 * step) as usize].clone())
            .collect();
        eprintln!("sampled down to {} clips", hits.len());
    }
    let slug = text::tokens(phrase).join("_");
    let work = common.work().join(format!("cut_{slug}"));
    std::fs::create_dir_all(&work)?;
    if !opts.no_verify {
        let asr = common.asr()?;
        let kept = hits::verify(&asr, recs, phrase, &mut hits, true)?;
        eprintln!("verified {kept}/{} clips", hits.len());
    }
    std::fs::write(work.join("hits.json"), serde_json::to_string_pretty(&hits)?)?;

    let font = opts.font.clone().unwrap_or_else(render::default_font);
    let subtitle = opts.subtitle.clone().unwrap_or_else(|| {
        let dir = std::fs::canonicalize(&common.dir).unwrap_or(common.dir.clone());
        dir.file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_default()
    });
    std::fs::create_dir_all(&opts.out)?;
    for clean in [false, true] {
        let style = render::Style {
            clean,
            title: format!("\u{201c}{}\u{201d}", phrase.to_uppercase()),
            subtitle: subtitle.clone(),
            font: font.clone(),
        };
        let out = opts.out.join(format!(
            "{slug}_supercut{}.mp4",
            if clean { "_clean" } else { "" }
        ));
        render::render(
            &hits,
            recs,
            &style,
            &work.join(if clean { "clips_clean" } else { "clips" }),
            &out,
        )?;
        eprintln!("wrote {}", out.display());
    }
    Ok(())
}

fn main() -> Result<()> {
    match Cli::parse().cmd {
        Cmd::Transcribe { common } => transcribe(&common, &common.recordings()?),
        Cmd::Mine {
            common,
            captions,
            top,
        } => {
            let recs = common.recordings()?;
            let texts = if captions {
                recs.iter().filter_map(library::captions).collect()
            } else {
                transcribe(&common, &recs)?;
                transcript_texts(&common, &recs)?
            };
            print_candidates(&mine::mine(&mine::docs_from_text(&texts), 4, 15), top);
            Ok(())
        }
        Cmd::Cut {
            common,
            phrase,
            cut: opts,
        } => {
            let recs = common.recordings()?;
            transcribe(&common, &recs)?;
            cut(&common, &recs, &phrase, &opts)
        }
        Cmd::FullAuto {
            common,
            pick,
            cut: opts,
        } => {
            let recs = common.recordings()?;
            eprintln!("{} videos in {}", recs.len(), common.dir.display());
            transcribe(&common, &recs)?;
            let cands = mine::mine(
                &mine::docs_from_text(&transcript_texts(&common, &recs)?),
                4,
                15,
            );
            print_candidates(&cands, 15);
            let chosen = cands
                .get(pick.saturating_sub(1))
                .context("no catchphrase candidates found")?;
            eprintln!("\npicked \u{201c}{}\u{201d}\n", chosen.phrase);
            cut(&common, &recs, &chosen.phrase, &opts)
        }
    }
}
