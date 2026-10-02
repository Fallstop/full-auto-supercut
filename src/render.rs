//! Cuts clips with ffmpeg (in parallel), adds optional title/end cards and overlays, and joins them.

use crate::hits::Hit;
use crate::library::Recording;
use crate::project::{Event, Log};
use anyhow::{Result, bail};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicUsize, Ordering};

const VENC: &[&str] = &[
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30",
];
const AENC: &[&str] = &["-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2"];
/// Intermediate pieces keep lossless audio: AAC adds encoder delay at every segment boundary,
/// which eats the start or end of a word once there are hundreds of short pieces.
const APCM: &[&str] = &["-c:a", "pcm_s16le", "-ar", "48000", "-ac", "2"];
const FRAME: &str = "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30,setsar=1";

/// One piece of source video to cut, with optional overlays.
pub struct Clip {
    pub video: PathBuf,
    pub s: f64,
    pub e: f64,
    /// Volume adjustment in dB (sentence mode levels each word).
    pub gain_db: f64,
    /// Bottom-right tag, e.g. "#12".
    pub tag: Option<String>,
    /// Bottom-left label, e.g. "Wed 12 Aug".
    pub label: Option<String>,
    /// Big caption at the bottom centre.
    pub caption: Option<String>,
    /// Seconds of held frame and silence after the clip.
    pub pause: f64,
}

/// (text, font size, colour, vertical offset from centre)
pub type CardLine = (String, u32, &'static str, i32);

pub struct Cards {
    pub title: Vec<CardLine>,
    pub end: Vec<CardLine>,
}

fn ffmpeg(args: &[&str]) -> Result<()> {
    let out = Command::new("ffmpeg")
        .args(["-nostdin", "-y", "-loglevel", "error"])
        .args(args)
        .output()?;
    if !out.status.success() {
        bail!(
            "ffmpeg {}: {}",
            args.join(" "),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(())
}

/// Escape text for an ffmpeg drawtext `text='...'` value.
fn esc(t: &str) -> String {
    t.replace('\\', "\\\\")
        .replace('\'', "\u{2019}")
        .replace(':', "\\:")
        .replace('%', "\\%")
}

pub fn default_font() -> PathBuf {
    let out = Command::new("fc-match")
        .args(["-f", "%{file}", "sans:black"])
        .output();
    out.ok()
        .map(|o| PathBuf::from(String::from_utf8_lossy(&o.stdout).to_string()))
        .unwrap_or_default()
}

fn card(dst: &Path, font: &Path, lines: &[CardLine], dur: f64) -> Result<()> {
    let draws = lines
        .iter()
        .map(|(t, size, color, dy)| {
            format!("drawtext=fontfile='{}':text='{}':fontsize={size}:fontcolor={color}:x=(w-tw)/2:y=(h/2)+({dy})", font.display(), esc(t))
        })
        .collect::<Vec<_>>()
        .join(",");
    let color = format!("color=c=0x111111:s=1280x720:r=30:d={dur}");
    let dur = dur.to_string();
    let mut args = vec![
        "-f",
        "lavfi",
        "-i",
        &color,
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=48000:cl=stereo",
        "-t",
        &dur,
        "-vf",
        &draws,
    ];
    args.extend(VENC);
    args.extend(APCM);
    args.push(dst.to_str().unwrap());
    ffmpeg(&args)
}

/// Supercut clips from verified hits: a running `#N` counter and the recording label, unless clean.
pub fn supercut_clips(hits: &[Hit], recs: &[Recording], clean: bool) -> Vec<Clip> {
    let mut first = 1;
    hits.iter()
        .filter(|h| !h.drop)
        .map(|h| {
            let rec = recs
                .iter()
                .find(|r| r.id == h.id)
                .expect("hit for unknown recording");
            let tag = if h.n == 1 {
                format!("#{first}")
            } else {
                format!("#{first}-{}", first + h.n - 1)
            };
            first += h.n;
            Clip {
                video: rec.video.clone(),
                s: h.s,
                e: h.e,
                gain_db: 0.0,
                tag: (!clean).then_some(tag),
                label: (!clean).then(|| rec.label.clone()),
                caption: None,
                pause: 0.0,
            }
        })
        .collect()
}

pub fn render(
    clips: &[Clip],
    font: &Path,
    cards: Option<&Cards>,
    work: &Path,
    out: &Path,
    log: Log,
) -> Result<()> {
    if clips.is_empty() {
        bail!("no clips to render");
    }
    std::fs::create_dir_all(work)?;
    let font = font.display().to_string();
    let done = AtomicUsize::new(0);
    let paths: Vec<PathBuf> = clips
        .par_iter()
        .enumerate()
        .map(|(i, c)| -> Result<PathBuf> {
            let dst = work.join(format!("c{i:05}.mkv"));
            let d = c.e - c.s;
            let mut vf = FRAME.to_string();
            if let Some(tag) = &c.tag {
                vf += &format!(",drawtext=fontfile='{font}':text='{}':fontsize=56:fontcolor=white:borderw=4:bordercolor=black:x=w-tw-28:y=h-th-28", esc(tag));
            }
            if let Some(label) = &c.label {
                vf += &format!(",drawtext=fontfile='{font}':text='{}':fontsize=26:fontcolor=white@0.85:borderw=3:bordercolor=black:x=28:y=h-th-32", esc(label));
            }
            if let Some(caption) = &c.caption {
                vf += &format!(",drawtext=fontfile='{font}':text='{}':fontsize=64:fontcolor=white:borderw=5:bordercolor=black:x=(w-tw)/2:y=h-th-60", esc(caption));
            }
            let mut af = format!(
                "aresample=48000,volume={:.2}dB,afade=t=in:d=0.015,afade=t=out:st={:.3}:d=0.03",
                c.gain_db,
                (d - 0.03).max(0.0)
            );
            if c.pause > 0.0 {
                vf += &format!(",tpad=stop_mode=clone:stop_duration={:.3}", c.pause);
                af += &format!(",apad=pad_dur={:.3}", c.pause);
            }
            let (ss, t) = (format!("{:.3}", c.s), format!("{d:.3}"));
            // -t before -i limits what's read, so the pause padding still makes it to the output.
            let mut args = vec!["-ss", &ss, "-t", &t, "-i", c.video.to_str().unwrap(), "-map", "0:v:0", "-map", "0:a:0", "-vf", &vf, "-af", &af];
            args.extend(VENC);
            args.extend(APCM);
            args.push(dst.to_str().unwrap());
            ffmpeg(&args)?;
            log(Event::Progress(done.fetch_add(1, Ordering::Relaxed) + 1, clips.len()));
            Ok(dst)
        })
        .collect::<Result<_>>()?;

    let mut parts = Vec::new();
    if let Some(cards) = cards {
        let title = work.join("title.mkv");
        card(&title, Path::new(&font), &cards.title, 2.5)?;
        parts.push(title);
    }
    parts.extend(paths);
    if let Some(cards) = cards {
        let end = work.join("end.mkv");
        card(&end, Path::new(&font), &cards.end, 3.0)?;
        parts.push(end);
    }
    let list = work.join("list.txt");
    std::fs::write(
        &list,
        parts
            .iter()
            .map(|p| format!("file '{}'\n", p.display()))
            .collect::<String>(),
    )?;
    let mut args = vec![
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list.to_str().unwrap(),
        "-c:v",
        "copy",
        "-af",
        "loudnorm=I=-16:TP=-1.5:LRA=11",
    ];
    args.extend(AENC);
    args.extend(["-movflags", "+faststart", out.to_str().unwrap()]);
    ffmpeg(&args)
}
