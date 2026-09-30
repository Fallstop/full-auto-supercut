//! Cuts clips with ffmpeg (in parallel), adds optional title/end cards and overlays, and joins them.

use crate::hits::Hit;
use crate::library::Recording;
use anyhow::{Context, Result, bail};
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::process::Command;

const VENC: &[&str] = &[
    "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-r", "30",
];
const AENC: &[&str] = &["-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2"];
const FRAME: &str = "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30,setsar=1";

pub struct Style {
    /// No title card, end card, counter or date label: just the clips.
    pub clean: bool,
    pub title: String,
    pub subtitle: String,
    pub font: PathBuf,
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

fn card(dst: &Path, font: &Path, lines: &[(&str, u32, &str, i32)], dur: f64) -> Result<()> {
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
    args.extend(AENC);
    args.push(dst.to_str().unwrap());
    ffmpeg(&args)
}

pub fn render(
    hits: &[Hit],
    recs: &[Recording],
    style: &Style,
    work: &Path,
    out: &Path,
) -> Result<()> {
    std::fs::create_dir_all(work)?;
    let hits: Vec<&Hit> = hits.iter().filter(|h| !h.drop).collect();
    if hits.is_empty() {
        bail!("no clips to render");
    }
    let mut first = 1;
    let jobs: Vec<(usize, &Hit, usize)> = hits
        .iter()
        .enumerate()
        .map(|(i, h)| {
            let f = first;
            first += h.n;
            (i, *h, f)
        })
        .collect();
    let total = first - 1;
    let font = style.font.display().to_string();

    let clips: Vec<PathBuf> = jobs
        .par_iter()
        .map(|&(i, h, first)| -> Result<PathBuf> {
            let rec = recs.iter().find(|r| r.id == h.id).context("hit for unknown recording")?;
            let dst = work.join(format!("c{i:05}.mp4"));
            let d = h.e - h.s;
            let mut vf = FRAME.to_string();
            if !style.clean {
                let tag = if h.n == 1 { format!("#{first}") } else { format!("#{first}-{}", first + h.n - 1) };
                vf += &format!(
                    ",drawtext=fontfile='{font}':text='{}':fontsize=56:fontcolor=white:borderw=4:bordercolor=black:x=w-tw-28:y=h-th-28\
                     ,drawtext=fontfile='{font}':text='{}':fontsize=26:fontcolor=white@0.85:borderw=3:bordercolor=black:x=28:y=h-th-32",
                    esc(&tag), esc(&rec.label)
                );
            }
            let af = format!("aresample=48000,afade=t=in:d=0.015,afade=t=out:st={:.3}:d=0.03", (d - 0.03).max(0.0));
            let (ss, t) = (format!("{:.3}", h.s), format!("{d:.3}"));
            let mut args = vec!["-ss", &ss, "-i", rec.video.to_str().unwrap(), "-t", &t, "-map", "0:v:0", "-map", "0:a:0", "-vf", &vf, "-af", &af];
            args.extend(VENC);
            args.extend(AENC);
            args.push(dst.to_str().unwrap());
            ffmpeg(&args)?;
            Ok(dst)
        })
        .collect::<Result<_>>()?;

    let mut parts = Vec::new();
    if !style.clean {
        let title = work.join("title.mp4");
        card(
            &title,
            &style.font,
            &[
                (&style.title, 110, "white", -90),
                ("a supercut", 40, "0xffcc33", 40),
                (&style.subtitle, 30, "0x999999", 110),
            ],
            2.5,
        )?;
        parts.push(title);
    }
    parts.extend(clips);
    if !style.clean {
        let end = work.join("end.mp4");
        let n_recs = hits
            .iter()
            .map(|h| &h.id)
            .collect::<std::collections::HashSet<_>>()
            .len();
        let across = format!(
            "across {n_recs} recording{}",
            if n_recs == 1 { "" } else { "s" }
        );
        card(
            &end,
            &style.font,
            &[
                (&total.to_string(), 160, "0xffcc33", -120),
                ("times.", 60, "white", 60),
                (&across, 32, "0x999999", 140),
            ],
            3.0,
        )?;
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
