//! Discovers recordings in a folder: video files plus optional yt-dlp `.info.json` sidecars
//! (as produced by `yt-dlp --write-info-json`, e.g. for Panopto lecture downloads).

use anyhow::{Context, Result};
use regex::Regex;
use std::path::{Path, PathBuf};
use std::sync::LazyLock;

const VIDEO_EXTS: &[&str] = &["mp4", "mkv", "webm", "mov", "m4v"];

static BRACKET_ID: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[([0-9A-Za-z_-]{6,})\]").unwrap());
static DATE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\b(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\w* (\d{1,2} [A-Z][a-z]{2})").unwrap()
});

#[derive(Debug, Clone)]
pub struct Recording {
    /// Short stable id: first 8 chars of the bracketed yt-dlp id, else the file stem.
    pub id: String,
    pub video: PathBuf,
    /// Human label burned into clips, e.g. "Wed 12 Aug".
    pub label: String,
    /// Sort key: `timestamp` from info.json when available, else 0 (then file name order).
    pub timestamp: i64,
    pub info_json: Option<PathBuf>,
}

pub fn scan(dir: &Path) -> Result<Vec<Recording>> {
    let mut out = Vec::new();
    for entry in std::fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))? {
        let path = entry?.path();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();
        if !VIDEO_EXTS.contains(&ext.as_str()) {
            continue;
        }
        let stem = path.file_stem().unwrap().to_string_lossy().to_string();
        let id = BRACKET_ID
            .captures_iter(&stem)
            .last()
            .map(|c| c[1].chars().take(8).collect())
            .unwrap_or_else(|| stem.clone());
        let info_json = Some(path.with_extension("info.json")).filter(|p| p.exists());
        let (mut timestamp, mut title) = (0, stem.clone());
        if let Some(p) = &info_json
            && let Ok(v) = serde_json::from_str::<serde_json::Value>(&std::fs::read_to_string(p)?)
        {
            timestamp = v["timestamp"].as_i64().unwrap_or(0);
            title = v["title"].as_str().unwrap_or(&stem).to_string();
        }
        let label = DATE
            .captures(&title)
            .map(|c| format!("{} {}", &c[1], &c[2]))
            .unwrap_or_else(|| BRACKET_ID.replace_all(&title, "").trim().to_string());
        out.push(Recording {
            id,
            video: path,
            label,
            timestamp,
            info_json,
        });
    }
    out.sort_by(|a, b| (a.timestamp, &a.video).cmp(&(b.timestamp, &b.video)));
    Ok(out)
}

/// Panopto/yt-dlp captions for a recording: embedded `subtitles.*[0].data` in info.json,
/// or any sibling `.srt` sharing the id. Returns plain caption text.
pub fn captions(rec: &Recording) -> Option<String> {
    let mut srt = None;
    if let Some(p) = &rec.info_json {
        let v: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(p).ok()?).ok()?;
        if let Some(subs) = v["subtitles"].as_object() {
            srt = subs.values().find_map(|l| {
                l[0]["data"]
                    .as_str()
                    .filter(|d| !d.is_empty())
                    .map(String::from)
            });
        }
    }
    if srt.is_none() {
        let dir = rec.video.parent()?;
        srt = std::fs::read_dir(dir)
            .ok()?
            .flatten()
            .map(|e| e.path())
            .find_map(|p| {
                let name = p.file_name()?.to_string_lossy().to_string();
                (name.ends_with(".srt") && name.contains(&rec.id))
                    .then(|| std::fs::read_to_string(&p).ok())
                    .flatten()
            });
    }
    let text = srt?
        .lines()
        .filter(|l| {
            !l.trim().is_empty()
                && !l.contains("-->")
                && !l.trim().chars().all(|c| c.is_ascii_digit())
        })
        .collect::<Vec<_>>()
        .join(" ");
    Some(text)
}
