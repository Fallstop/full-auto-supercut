//! Finds supercut-worthy catchphrases: phrases one speaker leans on far more than everyone else.
//!
//! For each n-gram, the recordings where it's used *habitually* (rate ≥ 25% of its peak rate) are
//! assumed to be the speaker(s) with the tic. Score = √(uses there) × ln(1 + lift) × length bonus,
//! where lift = habitual rate ÷ rate everywhere else. Two filters keep it funny:
//! - it must be a habit across several recordings, not a one-off (fire-alarm announcements,
//!   a video played in class);
//! - it must be made of everyday spoken words, so topic jargon ("cpu", "shared resource")
//!   that clusters in a few lectures doesn't masquerade as a tic;
//! - phrases with a filler or discourse marker ("right", "you know", "sort of") get a bonus over
//!   grammatical glue ("so you"), which matters most when one speaker dominates and lift is flat.

use crate::text::{ends_sentence, tokens_with_raw};
use std::collections::{HashMap, HashSet};

/// Words that make an n-gram look like a fragment when they sit at either edge of it.
const EDGE_STOP: &[&str] = &[
    "the", "a", "an", "of", "to", "and", "is", "in", "that", "it", "for", "with", "on", "at", "as",
    "be", "or", "are", "was", "we", "i", "if", "but", "this", "then", "which", "will", "have",
    "has", "can",
];
/// Unigrams too generic to be a joke on their own.
const BORING: &[&str] = &[
    "right", "okay", "like", "yeah", "because", "very", "let", "has", "why", "going", "does",
    "well", "really", "the", "a", "an", "of", "to", "and", "is", "in", "that", "it", "for", "with",
    "on", "at", "as", "be", "or", "are", "was", "we", "i", "you", "so", "this", "then", "not",
    "have", "will", "can", "what", "do", "if", "one", "they", "there", "but", "all", "just",
    "from", "by", "my", "me", "your", "our", "us", "he", "she", "his", "her", "it's", "that's",
    "i'm", "don't", "no", "yes", "how", "which", "when", "some", "would", "should", "could",
    "want", "need", "get", "go", "see", "say", "know", "think", "two", "zero", "three", "four",
    "here", "now", "up", "out", "about", "time", "these", "those", "them", "its", "into", "more",
    "any", "also", "only", "same", "other", "value",
];

/// Everyday spoken English plus the usual filler vocabulary. Every word of a catchphrase must be here.
const EVERYDAY: &str = "a about actually after again ah ahead alright all almost along already also always am amazing an
and another any anybody anyone anything anyway are aren't around as ask at away awesome back bad basically be beautiful
because been before being believe best better big bit both boy bring bunch but by call came can can't case cause certain
certainly check clear come comes coming cool could couldn't crazy cut day did didn't different do does doesn't doing
don't done down each easy eh either else end enough entire er essentially even ever every everybody everyone everything
exactly example fact fair far feel few fine first folks for forth forget found friend friends from fun funny gentlemen
get gets getting give given go goes going gone gonna good got gotta great guess guy guys had half hand happen happens
happy hard has have haven't having he hear heard hello help her here hey hi him his hmm hold honestly hope how huh i
i'd i'll i'm i've idea if important in indeed instead interesting into is isn't it it'll it's its just keep kind knew
know kinda ladies last later least leave left less let let's life like likes literally little long look looking looks
lot lots love magic make makes making man many matter may maybe me mean means meant might mind mine minute moment more
most much must my myself need needs never new next nice no nobody nope not nothing now obviously of off oh ok okay on
once one only oops or other others otherwise our out over own people perfect perhaps person place play please point
possible pretty probably problem put quick quickly quite rather real really remember right said same saw say saying
says see seem seems seen send sense she should show simple simply since so some somebody someone something sometimes
somewhere sort sorry start stay still stuff such super supposed sure take talk talking tell than thank thanks that
that'll that's the their them then there there's these they they're thing things think this those though thought three
through time to today together told too totally tried true try trying turn two uh uh-huh um understand unless until up
upon us use used very wait want wanted wants was wasn't way we we'll we're we've well went were what what's whatever
when where whether which while who whole why will wish with without won't wonder work works world would wouldn't wow
ya yeah yep yes yet you you'd you'll you're you've your yourself";

/// Filler words and discourse markers: a phrase containing one of these gets a bonus.
const FILLERS: &[&str] = &[
    "right",
    "okay",
    "ok",
    "alright",
    "yeah",
    "yep",
    "like",
    "basically",
    "actually",
    "literally",
    "guys",
    "guy",
    "folks",
    "know",
    "sort",
    "kind",
    "kinda",
    "stuff",
    "thing",
    "whatever",
    "gonna",
    "um",
    "uh",
    "er",
    "cool",
    "awesome",
    "perfect",
    "beautiful",
    "magic",
    "crazy",
    "obviously",
    "essentially",
    "honestly",
    "anyway",
    "wow",
    "oops",
    "hmm",
    "huh",
    "bit",
    "bunch",
    "mean",
    "totally",
    "super",
    "pretty",
    "eh",
];

#[derive(Debug, Clone, serde::Serialize)]
pub struct Candidate {
    pub phrase: String,
    pub count: usize,
    /// Recordings where the phrase is used habitually (rate ≥ 25% of its peak rate).
    pub habitual: usize,
    /// How many times more often it's said in those recordings than in the rest.
    pub lift: f64,
    pub score: f64,
    pub per_recording: Vec<usize>,
}

/// A recording's tokens, and whether each one ends a sentence.
pub struct Doc {
    pub toks: Vec<String>,
    pub ends: Vec<bool>,
}

impl Doc {
    fn len(&self) -> usize {
        self.toks.len()
    }
}

pub fn mine(docs: &[Doc], max_n: usize, min_count: usize) -> Vec<Candidate> {
    let totals: Vec<f64> = docs.iter().map(|d| d.len().max(1) as f64).collect();
    let k = docs.len();
    let mut counts: HashMap<String, Vec<usize>> = HashMap::new();
    for n in 1..=max_n {
        for (di, d) in docs.iter().enumerate() {
            for (i, g) in d.toks.windows(n).enumerate() {
                let repeat = g.iter().all(|t| *t == g[0]);
                if !repeat && d.ends[i..i + n - 1].iter().any(|&e| e) {
                    continue; // spans a sentence boundary
                }
                if n == 1 && (BORING.contains(&g[0].as_str()) || g[0].len() < 3) {
                    continue;
                }
                if n > 1
                    && (EDGE_STOP.contains(&g[0].as_str())
                        && EDGE_STOP.contains(&g[n - 1].as_str()))
                {
                    continue;
                }
                // "sort of" / "kind of" / "bunch of" are complete phrases despite ending in "of".
                let of_phrase = n > 1
                    && g[n - 1] == "of"
                    && ["sort", "kind", "bunch", "lot"].contains(&g[n - 2].as_str());
                if n > 1 && EDGE_STOP.contains(&g[n - 1].as_str()) && !of_phrase {
                    continue;
                }
                counts.entry(g.join(" ")).or_insert_with(|| vec![0; k])[di] += 1;
            }
        }
    }

    let everyday: HashSet<&str> = EVERYDAY.split_whitespace().collect();
    let total_words: f64 = totals.iter().sum();
    let min_habitual = if k >= 6 { 3 } else { k.min(2) };
    let mut cands: Vec<Candidate> = counts
        .into_iter()
        .filter_map(|(phrase, per)| {
            let count: usize = per.iter().sum();
            let n = phrase.split(' ').count();
            if count < min_count || !phrase.split(' ').all(|t| everyday.contains(t)) {
                return None;
            }
            let rates: Vec<f64> = per
                .iter()
                .zip(&totals)
                .map(|(&c, &t)| c as f64 / t)
                .collect();
            let peak = rates.iter().cloned().fold(0.0, f64::max);
            let habit: Vec<usize> = (0..k)
                .filter(|&i| rates[i] > 0.0 && rates[i] >= 0.25 * peak)
                .collect();
            if habit.len() < min_habitual {
                return None;
            }
            let (hc, hw) = habit
                .iter()
                .fold((0.0, 0.0), |(c, w), &i| (c + per[i] as f64, w + totals[i]));
            let (rc, rw) = (count as f64 - hc, total_words - hw);
            // Rest-of-corpus rate with a +1 prior so "never said elsewhere" doesn't divide by zero.
            let lift = if rw > 0.0 {
                ((hc / hw) / ((rc + 1.0) / rw)).min(200.0)
            } else {
                1.0
            };
            let length_bonus = [0.0, 0.6, 1.0, 1.1, 1.1][n.min(4)];
            // Fillers and discourse markers are what make a supercut funny; pure grammatical glue
            // ("so you", "you need") isn't, even when one speaker says it a lot.
            let filler = if phrase.split(' ').any(|t| FILLERS.contains(&t)) {
                1.6
            } else {
                0.6
            };
            let score = hc.sqrt() * (1.0 + lift).ln() * length_bonus * filler;
            Some(Candidate {
                phrase,
                count,
                habitual: habit.len(),
                lift,
                score,
                per_recording: per,
            })
        })
        .collect();
    cands.sort_by(|a, b| b.score.total_cmp(&a.score));

    // Keep the most complete form of a phrase: drop an n-gram when a longer n-gram containing it
    // accounts for most of its uses ("sort of" → "some sort of"), and drop a longer n-gram that is
    // just a rarer extension of a stronger shorter one ("this guy is" → "this guy").
    let all: Vec<Candidate> = cands.clone();
    let contains = |long: &str, short: &str| {
        long != short && format!(" {long} ").contains(&format!(" {short} "))
    };
    let mut out = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for c in cands {
        let absorbed = all
            .iter()
            .any(|o| contains(&o.phrase, &c.phrase) && o.count as f64 >= 0.7 * c.count as f64);
        let extension = all
            .iter()
            .any(|o| contains(&c.phrase, &o.phrase) && o.score > c.score * 1.3);
        if absorbed || extension || !seen.insert(c.phrase.clone()) {
            continue;
        }
        out.push(c);
    }
    out
}

pub fn docs_from_text(texts: &[String]) -> Vec<Doc> {
    texts
        .iter()
        .map(|t| {
            let tw = tokens_with_raw(t);
            Doc {
                ends: tw.iter().map(|(_, raw)| ends_sentence(raw)).collect(),
                toks: tw.into_iter().map(|(t, _)| t).collect(),
            }
        })
        .collect()
}
