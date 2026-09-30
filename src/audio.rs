use anyhow::{Context, Result, bail};
use std::path::Path;
use std::process::Command;

/// Decode audio to 16 kHz mono f32 via ffmpeg, optionally only the window `[start, start+dur)`.
pub fn decode(video: &Path, window: Option<(f64, f64)>) -> Result<Vec<f32>> {
    let mut cmd = Command::new("ffmpeg");
    cmd.args(["-nostdin", "-loglevel", "error"]);
    if let Some((start, _)) = window {
        cmd.args(["-ss", &format!("{start:.3}")]);
    }
    cmd.arg("-i").arg(video);
    if let Some((_, dur)) = window {
        cmd.args(["-t", &format!("{dur:.3}")]);
    }
    cmd.args(["-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]);
    let out = cmd.output().context("running ffmpeg (is it installed?)")?;
    if !out.status.success() {
        bail!(
            "ffmpeg failed on {}: {}",
            video.display(),
            String::from_utf8_lossy(&out.stderr)
        );
    }
    Ok(out
        .stdout
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| i16::from_le_bytes(*b) as f32 / 32768.0)
        .collect())
}

/// True if the recording has no usable speech: quiet *and* flat. Loudness alone isn't enough —
/// a lecture recorded from across the room can average -55 dBFS — but speech always swings
/// between pauses and words, so the spread between loud and quiet half-seconds gives it away.
pub fn is_silent(pcm: &[f32]) -> bool {
    let mut db: Vec<f64> = pcm
        .chunks(8000)
        .map(|w| {
            10.0 * (w.iter().map(|&x| (x as f64).powi(2)).sum::<f64>() / w.len() as f64)
                .max(1e-12)
                .log10()
        })
        .collect();
    if db.is_empty() {
        return true;
    }
    db.sort_by(f64::total_cmp);
    let pct = |p: f64| db[((db.len() - 1) as f64 * p) as usize];
    let (p10, p90) = (pct(0.1), pct(0.9));
    p90 - p10 < 10.0 && p90 < -45.0
}
