# full-auto-supercut

Point it at a folder of videos and get back a supercut of whatever someone keeps saying.

```sh
full-auto-supercut full-auto ~/Videos/lectures -o out/
```

That one command:

1. **Transcribes** every video locally with Whisper (`large-v3-turbo` via whisper.cpp), with word-level timestamps.
2. **Mines catchphrases**: it finds the phrases one speaker leans on far more than everyone else ("you know", "this guy", "some sort of").
3. **Cuts every occurrence** with a little padding and merges back-to-back repeats ("this guy, this guy").
4. **Verifies every clip** by re-transcribing just that clip and steering the window until Whisper hears exactly the phrase. Stray words after the phrase pull the end in, stray words before it push the start later, and a half-heard phrase widens the window. Clips that never come out clean, or that drag (long pauses mid-phrase), are dropped.
5. **Renders two cuts** in parallel with ffmpeg:
   - `<phrase>_supercut.mp4`: title card, a running `#N` counter, a date/recording label on each clip, and an end card with the total.
   - `<phrase>_supercut_clean.mp4`: just the clips, back to back.

Both cuts are loudness-normalised.

It works on any folder of `mp4`/`mkv`/`webm`/`mov`/`m4v`. If the videos were downloaded with `yt-dlp --write-info-json` (e.g. Panopto lecture recordings), recordings are ordered by date and labelled like `Wed 12 Aug`, and the platform captions can be mined without a GPU.

## How it picks the phrase

For every 1–4 word n-gram, it finds the recordings where the phrase is used *habitually* (at least 25% of its peak per-word rate) and scores:

```
score = √(uses in habitual recordings) × ln(1 + lift) × length bonus × filler bonus
lift  = rate in habitual recordings ÷ rate everywhere else
```

A verbal tic is something one speaker overuses compared with everyone else, so "right" and "okay" (everyone says them) rank low. On top of that:

- The phrase has to be a **habit**, appearing in at least 3 recordings. One-offs like a fire-alarm announcement or a video played in class don't count.
- It has to be made of **everyday spoken words**, so topic jargon that clusters in a few recordings doesn't pass for a tic.
- Phrases containing a **filler or discourse marker** ("right", "okay", "guys", "you know", "sort of") get a bonus over grammatical glue ("so you", "you need"). This matters most when one person does all the talking, so there's nobody to compare against and lift is flat.
- It has to be **one utterance**. Phrases that run across a sentence boundary ("…, right? So, …") are ignored, because they have a pause in the middle and never clip cleanly. Repeats like "Yeah. Yeah. Yeah." are the exception.
- The most complete form of a phrase wins: "some sort of" beats "sort of".

Whisper invents text over silence, so there are three guards against it:

- Recordings with no speech are skipped. The test is that the audio is quiet *and* flat (loud and quiet half-seconds within 10 dB of each other), so a lecture recorded from across the room still gets transcribed.
- Segments Whisper itself marks as probably not speech are dropped.
- Repetition loops ("thank you thank you thank you…") are removed.

## Commands

| | |
|---|---|
| `full-auto <dir>` | everything above; `--pick N` takes the Nth-ranked phrase instead |
| `mine <dir>` | print the ranked candidates (`--captions` mines yt-dlp caption sidecars instead, no GPU) |
| `cut <dir> "<phrase>"` | supercut a phrase of your choosing |
| `transcribe <dir>` | just transcribe (cached; reruns skip finished videos) |

Useful options: `-o <out dir>`, `--pre/--post <seconds>` padding, `--max-clips N` (evenly samples), `--no-verify`, `--subtitle`, `--font`, `--model <ggml file>`, `--work <cache dir>` (default `<dir>/.supercut`).

Transcripts are cached as `<work>/words/<id>.json` (`[{w, s, e, p}]`). This is the same shape faster-whisper word output converts to, so you can drop in transcripts from elsewhere.

## Install

Needs `ffmpeg` and `fc-match` (fontconfig) on `PATH`, plus a Rust toolchain. The model (1.6 GB) is downloaded to `~/.cache/full-auto-supercut/` on first use.

```sh
# NVIDIA GPU (needs the CUDA toolkit)
cargo install --git https://github.com/Fallstop/full-auto-supercut --features cuda
# or: --features vulkan / --features metal / no features for CPU (slow)
```

On an RTX 3090, transcription runs at about 70× real time (an hour of lecture takes ~50 s). Cutting and verifying a 200-clip supercut takes a couple of minutes.

## Notes

- Word timings come from whisper.cpp's DTW token alignment, with a 0.2 s lag correction calibrated against faster-whisper. The verify pass absorbs the remaining jitter.
- Whisper tends to tidy away disfluencies. The initial prompt nudges it to keep fillers like "um" and "you know", but counts of those will still be undercounts.

## License

MIT
