mod asr;
mod audio;
mod hits;
mod library;
mod mine;
mod project;
mod render;
mod say;
mod text;
mod tui;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand};
use project::{CutOpts, Event, Project};
use std::path::PathBuf;

/// Point it at a folder of videos; get back a supercut of whatever someone keeps saying.
/// Run with just a folder to open the interactive TUI.
#[derive(Parser)]
#[command(version, about, args_conflicts_with_subcommands = true)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,
    #[command(flatten)]
    tui: Option<Common>,
}

#[derive(Subcommand)]
enum Cmd {
    /// Open the interactive TUI (the default when you just pass a folder).
    Tui {
        #[command(flatten)]
        common: Common,
    },
    /// Transcribe, find the catchphrase, verify every clip, and render the supercut. No questions asked.
    FullAuto {
        #[command(flatten)]
        common: Common,
        /// Use the Nth-ranked candidate instead of the top one.
        #[arg(long, default_value_t = 1)]
        pick: usize,
        #[command(flatten)]
        cut: CutArgs,
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
        cut: CutArgs,
    },
    /// Make the recordings say any sentence, stitched from words they actually said.
    Say {
        #[command(flatten)]
        common: Common,
        sentence: String,
        /// Output folder.
        #[arg(short, long, default_value = ".")]
        out: PathBuf,
        /// Burn the words in as captions.
        #[arg(long)]
        captions: bool,
        /// Skip re-transcribing each piece to check it.
        #[arg(long)]
        no_verify: bool,
    },
}

#[derive(Args, Clone)]
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
struct CutArgs {
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

impl From<CutArgs> for CutOpts {
    fn from(a: CutArgs) -> Self {
        Self {
            out: a.out,
            pre: a.pre,
            post: a.post,
            verify: !a.no_verify,
            max_clips: a.max_clips,
            subtitle: a.subtitle,
            font: a.font,
        }
    }
}

impl Common {
    fn open(&self) -> Result<Project> {
        Project::open(&self.dir, self.work.clone(), self.model.clone())
    }
}

/// CLI progress: print messages, ignore progress ticks.
fn print(e: Event) {
    if let Event::Info(msg) = e {
        eprintln!("{msg}");
    }
}

fn print_candidates(cands: &[mine::Candidate], top: usize) {
    eprintln!(
        "{:>3}  {:<30} {:>6} {:>9} {:>6} {:>6}",
        "#", "phrase", "count", "habitual", "lift", "score"
    );
    for (i, c) in cands.iter().take(top).enumerate() {
        let habitual = format!("{}/{}", c.habitual, c.per_recording.len());
        let phrase = format!("\u{201c}{}\u{201d}", c.phrase);
        eprintln!(
            "{:>3}  {:<30} {:>6} {:>9} {:>5.0}x {:>6.1}",
            i + 1,
            phrase,
            c.count,
            habitual,
            c.lift,
            c.score
        );
    }
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let cmd = match (cli.cmd, cli.tui) {
        (Some(cmd), _) => cmd,
        (None, Some(common)) => Cmd::Tui { common },
        (None, None) => bail!("pass a folder of videos (or --help)"),
    };
    match cmd {
        Cmd::Tui { common } => tui::run(common.open()?),
        Cmd::Transcribe { common } => common.open()?.transcribe(&print),
        Cmd::Mine {
            common,
            captions,
            top,
        } => {
            let p = common.open()?;
            let cands = if captions {
                let texts: Vec<String> = p.recs.iter().filter_map(library::captions).collect();
                mine::mine(&mine::docs_from_text(&texts), 4, 15)
            } else {
                p.transcribe(&print)?;
                p.candidates()?
            };
            print_candidates(&cands, top);
            Ok(())
        }
        Cmd::Cut {
            common,
            phrase,
            cut,
        } => {
            let p = common.open()?;
            p.transcribe(&print)?;
            p.cut(&phrase, &cut.into(), &print).map(|_| ())
        }
        Cmd::FullAuto { common, pick, cut } => {
            let p = common.open()?;
            eprintln!("{} videos in {}", p.recs.len(), p.dir.display());
            p.transcribe(&print)?;
            let cands = p.candidates()?;
            print_candidates(&cands, 15);
            let chosen = cands
                .get(pick.saturating_sub(1))
                .context("no catchphrase candidates found")?;
            eprintln!("\npicked \u{201c}{}\u{201d}\n", chosen.phrase);
            p.cut(&chosen.phrase, &cut.into(), &print).map(|_| ())
        }
        Cmd::Say {
            common,
            sentence,
            out,
            captions,
            no_verify,
        } => {
            let p = common.open()?;
            p.transcribe(&print)?;
            let index = say::Index::build(p.docs()?);
            let plan = index.plan(&sentence);
            eprintln!("{}", say::describe(&index, &plan));
            let opts = say::SayOpts {
                out,
                captions,
                verify: !no_verify,
            };
            say::build(&p, &index, &plan, &opts, &print).map(|_| ())
        }
    }
}
