//! Local speech-to-text via whisper.cpp (whisper-rs), producing word-level timestamps.

use crate::text::Word;
use anyhow::{Context, Result};
use std::path::Path;
use std::sync::Mutex;
use whisper_rs::{
    DtwMode, DtwModelPreset, FullParams, SamplingStrategy, WhisperContext,
    WhisperContextParameters, WhisperState,
};

pub struct Asr {
    ctx: WhisperContext,
    /// Reused across calls: creating a state allocates GPU buffers, which dominates the cost of
    /// transcribing the many short clips during verification.
    state: Mutex<WhisperState>,
}

/// whisper.cpp's DTW timestamps land a little after the true word onset; measured against
/// faster-whisper's word alignment the median lag is ~0.2 s.
const DTW_LAG: f64 = 0.2;

/// Nudges whisper to keep disfluencies ("um", "you know") instead of cleaning them away,
/// since those are exactly what supercuts are made of.
const PROMPT: &str = "Um, uh, you know, so, right, okay. Yeah, yeah.";

fn dtw_preset(model: &Path) -> Option<DtwModelPreset> {
    let name = model.file_name()?.to_string_lossy().to_lowercase();
    Some(match () {
        _ if name.contains("large-v3-turbo") => DtwModelPreset::LargeV3Turbo,
        _ if name.contains("large-v3") => DtwModelPreset::LargeV3,
        _ if name.contains("large-v2") => DtwModelPreset::LargeV2,
        _ if name.contains("medium.en") => DtwModelPreset::MediumEn,
        _ if name.contains("medium") => DtwModelPreset::Medium,
        _ if name.contains("small.en") => DtwModelPreset::SmallEn,
        _ if name.contains("small") => DtwModelPreset::Small,
        _ if name.contains("base.en") => DtwModelPreset::BaseEn,
        _ if name.contains("base") => DtwModelPreset::Base,
        _ if name.contains("tiny.en") => DtwModelPreset::TinyEn,
        _ if name.contains("tiny") => DtwModelPreset::Tiny,
        _ => return None,
    })
}

impl Asr {
    pub fn new(model: &Path) -> Result<Self> {
        whisper_rs::install_logging_hooks();
        let mut cp = WhisperContextParameters::default();
        if let Some(model_preset) = dtw_preset(model) {
            cp.dtw_parameters.mode = DtwMode::ModelPreset { model_preset };
        }
        let ctx = WhisperContext::new_with_params(model.to_str().context("model path")?, cp)
            .with_context(|| format!("loading whisper model {}", model.display()))?;
        let state = Mutex::new(ctx.create_state()?);
        Ok(Self { ctx, state })
    }

    /// Transcribe 16 kHz mono samples. `offset` (seconds) is added to every timestamp.
    pub fn words(&self, audio: &[f32], offset: f64, beam: bool) -> Result<Vec<Word>> {
        let mut state = self.state.lock().unwrap();
        let strategy = if beam {
            SamplingStrategy::BeamSearch {
                beam_size: 5,
                patience: -1.0,
            }
        } else {
            SamplingStrategy::Greedy { best_of: 1 }
        };
        let mut p = FullParams::new(strategy);
        p.set_language(Some("en"));
        p.set_n_threads(std::thread::available_parallelism().map_or(4, |n| n.get().min(8)) as i32);
        p.set_token_timestamps(true);
        p.set_no_context(true); // stops long lectures from getting stuck in repetition loops
        p.set_initial_prompt(PROMPT);
        p.set_print_progress(false);
        p.set_print_realtime(false);
        p.set_print_special(false);
        p.set_print_timestamps(false);
        state.full(p, audio)?;

        let eot = self.ctx.token_eot();
        let mut words: Vec<Word> = Vec::new();
        for seg in state.as_iter() {
            if seg.no_speech_probability() > 0.6 {
                continue; // whisper hallucinates "Thank you." over silence
            }
            for i in 0..seg.n_tokens() {
                let Some(tok) = seg.get_token(i) else {
                    continue;
                };
                if tok.token_id() >= eot {
                    continue; // special / timestamp tokens
                }
                let Ok(piece) = tok.to_str_lossy() else {
                    continue;
                };
                let d = tok.token_data();
                // DTW gives a much better onset than the t0/t1 heuristic when available.
                let start = if d.t_dtw >= 0 {
                    d.t_dtw as f64 / 100.0 - DTW_LAG
                } else {
                    d.t0 as f64 / 100.0
                } + offset;
                let end = (d.t1 as f64 / 100.0 + offset).max(start);
                match words.last_mut() {
                    Some(w) if !piece.starts_with(' ') && !w.w.is_empty() => {
                        w.w.push_str(&piece);
                        w.e = w.e.max(end);
                        w.p = w.p.min(tok.token_probability());
                    }
                    _ => words.push(Word {
                        w: piece.to_string(),
                        s: start,
                        e: end,
                        p: tok.token_probability(),
                    }),
                }
            }
        }
        words.retain(|w| !w.w.trim().is_empty());
        // Token end times from whisper.cpp are unreliable (often equal to the start). A word ends
        // where the next begins, capped so a pause doesn't get swallowed.
        for i in 0..words.len() {
            let next = words.get(i + 1).map_or(f64::MAX, |n| n.s);
            let w = &mut words[i];
            w.s = w.s.max(0.0);
            let spoken = (w.e - w.s).max(0.12 + 0.06 * w.w.trim().len() as f64);
            w.e = (w.s + spoken).min(next).max(w.s + 0.05);
        }
        for w in &mut words {
            w.s = (w.s * 1000.0).round() / 1000.0;
            w.e = (w.e * 1000.0).round() / 1000.0;
        }
        Ok(words)
    }

    /// Plain-text transcript of a short window, used for verification.
    pub fn text(&self, audio: &[f32]) -> Result<String> {
        Ok(self
            .words(audio, 0.0, true)?
            .iter()
            .map(|w| w.w.as_str())
            .collect::<String>())
    }
}
