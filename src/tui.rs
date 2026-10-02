//! Interactive terminal UI: browse catchphrases, build sentences, watch jobs run.

use crate::mine::Candidate;
use crate::project::{CutOpts, Event, Project};
use crate::say::{self, Index, Plan, SayOpts};
use crate::text;
use anyhow::Result;
use ratatui::crossterm::event::{
    self, Event as TermEvent, KeyCode, KeyEvent, KeyEventKind, KeyModifiers,
};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{
    Block, BorderType, Cell, LineGauge, Paragraph, Row, Table, TableState, Tabs, Wrap,
};
use ratatui::{DefaultTerminal, Frame};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, Sender, channel};
use std::time::{Duration, Instant};

const ACCENT: Color = Color::Rgb(255, 204, 51);
const CHUNK_COLORS: [Color; 2] = [Color::Rgb(120, 220, 140), Color::Rgb(110, 200, 240)];
const MISSING: Color = Color::Rgb(255, 95, 95);
const DIM: Color = Color::Rgb(130, 130, 130);

#[derive(Clone, Copy, PartialEq)]
enum Tab {
    Phrases,
    Say,
    Recordings,
}

const TABS: [(Tab, &str); 3] = [
    (Tab::Phrases, "Catchphrases"),
    (Tab::Say, "Say anything"),
    (Tab::Recordings, "Recordings"),
];

enum Msg {
    Event(Event),
    Loaded(Arc<Index>, Vec<Candidate>),
    Done {
        title: String,
        result: Result<Vec<PathBuf>, String>,
        play: bool,
        reload: bool,
    },
}

struct Job {
    title: String,
    done: usize,
    total: usize,
    started: Instant,
}

struct App {
    project: Arc<Project>,
    index: Option<Arc<Index>>,
    cands: Vec<Candidate>,
    tab: Tab,
    // Catchphrases
    phrase_table: TableState,
    custom: String,
    // Say anything
    input: String,
    cursor: usize,
    plan: Plan,
    chunk_sel: usize,
    captions: bool,
    verify: bool,
    autoplay: bool,
    // Recordings
    rec_table: TableState,
    rec_words: Vec<Option<usize>>,
    // Jobs
    job: Option<Job>,
    log: VecDeque<String>,
    last_output: Option<PathBuf>,
    out_dir: PathBuf,
    tx: Sender<Msg>,
    rx: Receiver<Msg>,
    quit: bool,
}

/// whisper.cpp / CUDA write straight to stderr, which would scribble over the UI.
/// Point fd 2 at a log file while the TUI runs; returns the original to restore.
#[cfg(unix)]
fn redirect_stderr(path: &std::path::Path) -> Option<i32> {
    use std::os::fd::AsRawFd;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .ok()?;
    unsafe {
        let saved = libc::dup(2);
        libc::dup2(file.as_raw_fd(), 2);
        Some(saved)
    }
}

#[cfg(unix)]
fn restore_stderr(saved: Option<i32>) {
    if let Some(fd) = saved {
        unsafe {
            libc::dup2(fd, 2);
            libc::close(fd);
        }
    }
}

#[cfg(not(unix))]
fn redirect_stderr(_: &std::path::Path) -> Option<i32> {
    None
}

#[cfg(not(unix))]
fn restore_stderr(_: Option<i32>) {}

pub fn run(project: Project) -> Result<()> {
    std::fs::create_dir_all(&project.work)?;
    let saved = redirect_stderr(&project.work.join("tui.log"));
    let mut terminal = ratatui::init();
    let result = App::new(project).run(&mut terminal);
    ratatui::restore();
    restore_stderr(saved);
    result
}

impl App {
    fn new(project: Project) -> Self {
        let (tx, rx) = channel();
        let mut app = Self {
            project: Arc::new(project),
            index: None,
            cands: Vec::new(),
            tab: Tab::Say,
            phrase_table: TableState::default().with_selected(Some(0)),
            custom: String::new(),
            input: String::new(),
            cursor: 0,
            plan: Plan::default(),
            chunk_sel: 0,
            captions: true,
            verify: true,
            autoplay: true,
            rec_table: TableState::default().with_selected(Some(0)),
            rec_words: Vec::new(),
            job: None,
            log: VecDeque::new(),
            last_output: None,
            out_dir: PathBuf::from("."),
            tx,
            rx,
            quit: false,
        };
        let pending = app.project.pending().len();
        if pending > 0 {
            app.push_log(format!(
                "{pending} recording(s) not transcribed yet \u{2014} press Ctrl+T to transcribe"
            ));
        }
        app.reload();
        app
    }

    fn push_log(&mut self, line: String) {
        self.log.push_back(line);
        while self.log.len() > 300 {
            self.log.pop_front();
        }
    }

    /// (Re)build the word index and catchphrase ranking in the background.
    fn reload(&mut self) {
        self.rec_words = self
            .project
            .recs
            .iter()
            .map(|r| {
                text::load_words(&self.project.words_dir(), &r.id)
                    .ok()
                    .flatten()
                    .map(|w| w.len())
            })
            .collect();
        let (project, tx) = (self.project.clone(), self.tx.clone());
        std::thread::spawn(move || {
            let index = project.docs().map(Index::build);
            let cands = project.candidates().unwrap_or_default();
            match index {
                Ok(index) => {
                    let _ = tx.send(Msg::Loaded(Arc::new(index), cands));
                }
                Err(e) => {
                    let _ = tx.send(Msg::Event(Event::Info(format!(
                        "couldn't load transcripts: {e:#}"
                    ))));
                }
            }
        });
    }

    /// Run `work` on a background thread, streaming its log into the job panel.
    fn spawn<F>(&mut self, title: String, play: bool, reload: bool, work: F)
    where
        F: FnOnce(&(dyn Fn(Event) + Sync)) -> Result<Vec<PathBuf>> + Send + 'static,
    {
        if let Some(job) = &self.job {
            let busy = format!("busy with {} \u{2014} wait for it to finish", job.title);
            self.push_log(busy);
            return;
        }
        self.push_log(format!("\u{25b6} {title}"));
        self.job = Some(Job {
            title: title.clone(),
            done: 0,
            total: 0,
            started: Instant::now(),
        });
        let tx = self.tx.clone();
        std::thread::spawn(move || {
            let sink = tx.clone();
            let log = move |e: Event| {
                let _ = sink.send(Msg::Event(e));
            };
            let result = work(&log).map_err(|e| format!("{e:#}"));
            let _ = tx.send(Msg::Done {
                title,
                result,
                play,
                reload,
            });
        });
    }

    fn run(mut self, terminal: &mut DefaultTerminal) -> Result<()> {
        while !self.quit {
            while let Ok(msg) = self.rx.try_recv() {
                self.on_msg(msg);
            }
            terminal.draw(|f| self.draw(f))?;
            if event::poll(Duration::from_millis(60))?
                && let TermEvent::Key(key) = event::read()?
                && key.kind == KeyEventKind::Press
            {
                self.on_key(key);
            }
        }
        Ok(())
    }

    fn on_msg(&mut self, msg: Msg) {
        match msg {
            Msg::Event(Event::Info(line)) => self.push_log(line),
            Msg::Event(Event::Progress(done, total)) => {
                if let Some(job) = &mut self.job {
                    (job.done, job.total) = (done, total);
                }
            }
            Msg::Loaded(index, cands) => {
                self.push_log(format!(
                    "indexed {} words from {} recordings",
                    index.words(),
                    index.docs.len()
                ));
                self.index = Some(index);
                self.cands = cands;
                self.replan();
            }
            Msg::Done {
                title,
                result,
                play,
                reload,
            } => {
                let secs = self
                    .job
                    .take()
                    .map_or(0.0, |j| j.started.elapsed().as_secs_f64());
                match result {
                    Ok(outs) => {
                        self.push_log(format!("\u{2713} {title} ({secs:.0}s)"));
                        if let Some(first) = outs.first() {
                            self.last_output = Some(first.clone());
                            if play && self.autoplay {
                                let _ = say::play(first, None);
                            }
                        }
                    }
                    Err(e) => self.push_log(format!("\u{2717} {title}: {e}")),
                }
                if reload {
                    self.reload();
                }
            }
        }
    }

    fn replan(&mut self) {
        if let Some(index) = &self.index {
            self.plan = index.plan(&self.input);
            self.chunk_sel = self.chunk_sel.min(self.plan.chunks.len().saturating_sub(1));
        }
    }

    // ---------------------------------------------------------------- keys

    fn on_key(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        match key.code {
            KeyCode::Char('c' | 'q') if ctrl => self.quit = true,
            KeyCode::F(n @ 1..=3) => self.tab = TABS[n as usize - 1].0,
            KeyCode::Right if ctrl => self.cycle_tab(1),
            KeyCode::Left if ctrl => self.cycle_tab(TABS.len() - 1),
            KeyCode::Char('t') if ctrl => self.transcribe(),
            KeyCode::Char('o') if ctrl => {
                if let Some(p) = &self.last_output {
                    let _ = say::play(p, None);
                }
            }
            KeyCode::Char('a') if ctrl => {
                self.autoplay = !self.autoplay;
                self.push_log(format!(
                    "autoplay {}",
                    if self.autoplay { "on" } else { "off" }
                ));
            }
            KeyCode::Char('v') if ctrl => {
                self.verify = !self.verify;
                self.push_log(format!(
                    "verification {}",
                    if self.verify { "on" } else { "off" }
                ));
            }
            _ => match self.tab {
                Tab::Phrases => self.on_key_phrases(key),
                Tab::Say => self.on_key_say(key),
                Tab::Recordings => self.on_key_recordings(key),
            },
        }
    }

    fn cycle_tab(&mut self, by: usize) {
        let i = TABS.iter().position(|(t, _)| *t == self.tab).unwrap();
        self.tab = TABS[(i + by) % TABS.len()].0;
    }

    fn transcribe(&mut self) {
        let n = self.project.pending().len();
        if n == 0 {
            self.push_log("everything is already transcribed".into());
            return;
        }
        let project = self.project.clone();
        self.spawn(
            format!("transcribing {n} recording(s)"),
            false,
            true,
            move |log| project.transcribe(log).map(|_| Vec::new()),
        );
    }

    fn on_key_phrases(&mut self, key: KeyEvent) {
        let n = self.cands.len();
        match key.code {
            KeyCode::Up => select(&mut self.phrase_table, n, -1),
            KeyCode::Down => select(&mut self.phrase_table, n, 1),
            KeyCode::PageUp => select(&mut self.phrase_table, n, -10),
            KeyCode::PageDown => select(&mut self.phrase_table, n, 10),
            KeyCode::Esc => self.custom.clear(),
            KeyCode::Backspace => {
                self.custom.pop();
            }
            KeyCode::Char(c) if !key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.custom.push(c)
            }
            KeyCode::Enter => {
                let phrase = if self.custom.trim().is_empty() {
                    match self.phrase_table.selected().and_then(|i| self.cands.get(i)) {
                        Some(c) => c.phrase.clone(),
                        None => return,
                    }
                } else {
                    self.custom.trim().to_string()
                };
                let project = self.project.clone();
                let opts = CutOpts {
                    out: self.out_dir.clone(),
                    verify: self.verify,
                    ..CutOpts::default()
                };
                self.spawn(
                    format!("supercut of \u{201c}{phrase}\u{201d}"),
                    true,
                    false,
                    move |log| project.cut(&phrase, &opts, log),
                );
            }
            _ => {}
        }
    }

    fn on_key_say(&mut self, key: KeyEvent) {
        let ctrl = key.modifiers.contains(KeyModifiers::CONTROL);
        let mut edited = false;
        match key.code {
            KeyCode::Char('k') if ctrl => {
                self.captions = !self.captions;
                self.push_log(format!(
                    "captions {}",
                    if self.captions { "on" } else { "off" }
                ));
            }
            KeyCode::Char('p') if ctrl => self.preview_take(),
            KeyCode::Char('u') if ctrl => {
                self.input.clear();
                self.cursor = 0;
                edited = true;
            }
            KeyCode::Char(c) if !ctrl => {
                self.input.insert(self.cursor, c);
                self.cursor += c.len_utf8();
                edited = true;
            }
            KeyCode::Backspace => {
                if let Some((i, _)) = self.input[..self.cursor].char_indices().last() {
                    self.input.remove(i);
                    self.cursor = i;
                    edited = true;
                }
            }
            KeyCode::Delete => {
                if self.cursor < self.input.len() {
                    self.input.remove(self.cursor);
                    edited = true;
                }
            }
            KeyCode::Left => {
                if let Some((i, _)) = self.input[..self.cursor].char_indices().last() {
                    self.cursor = i;
                }
            }
            KeyCode::Right => {
                if let Some(c) = self.input[self.cursor..].chars().next() {
                    self.cursor += c.len_utf8();
                }
            }
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.input.len(),
            KeyCode::Up => self.chunk_sel = self.chunk_sel.saturating_sub(1),
            KeyCode::Down => {
                self.chunk_sel = (self.chunk_sel + 1).min(self.plan.chunks.len().saturating_sub(1))
            }
            KeyCode::Tab => self.cycle_take(1),
            KeyCode::BackTab => self.cycle_take(-1),
            KeyCode::Enter => self.build_sentence(),
            _ => {}
        }
        if edited {
            self.replan();
        }
    }

    fn cycle_take(&mut self, by: isize) {
        if let Some(c) = self.plan.chunks.get_mut(self.chunk_sel)
            && !c.takes.is_empty()
        {
            c.choice = (c.choice as isize + by).rem_euclid(c.takes.len() as isize) as usize;
        }
    }

    fn preview_take(&mut self) {
        let (Some(index), Some(c)) = (&self.index, self.plan.chunks.get(self.chunk_sel)) else {
            return;
        };
        if let Some(t) = c.takes.get(c.choice) {
            let w = index.take_window(c, t);
            let _ = say::play(&index.docs[t.doc].0.video, Some(w));
        }
    }

    fn build_sentence(&mut self) {
        let Some(index) = self.index.clone() else {
            self.push_log("still indexing transcripts\u{2026}".into());
            return;
        };
        let missing = self.plan.missing_words();
        if !missing.is_empty() {
            let msg = format!(
                "can't build it: never said {}",
                missing
                    .iter()
                    .map(|w| format!("\u{201c}{w}\u{201d}"))
                    .collect::<Vec<_>>()
                    .join(", ")
            );
            self.push_log(msg);
            return;
        }
        if self.plan.chunks.is_empty() {
            return;
        }
        let (project, plan) = (self.project.clone(), self.plan.clone());
        let opts = SayOpts {
            out: self.out_dir.clone(),
            captions: self.captions,
            verify: self.verify,
        };
        let title = format!("\u{201c}{}\u{201d}", self.input.trim());
        self.spawn(title, true, false, move |log| {
            say::build(&project, &index, &plan, &opts, log).map(|p| vec![p])
        });
    }

    fn on_key_recordings(&mut self, key: KeyEvent) {
        let n = self.project.recs.len();
        match key.code {
            KeyCode::Up => select(&mut self.rec_table, n, -1),
            KeyCode::Down => select(&mut self.rec_table, n, 1),
            KeyCode::PageUp => select(&mut self.rec_table, n, -10),
            KeyCode::PageDown => select(&mut self.rec_table, n, 10),
            KeyCode::Enter => {
                if let Some(r) = self
                    .rec_table
                    .selected()
                    .and_then(|i| self.project.recs.get(i))
                {
                    let _ = say::play(&r.video, None);
                }
            }
            _ => {}
        }
    }

    // ---------------------------------------------------------------- drawing

    fn draw(&mut self, f: &mut Frame) {
        let [header, tabs, body, jobs, footer] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(2),
            Constraint::Min(8),
            Constraint::Length(9),
            Constraint::Length(1),
        ])
        .areas(f.area());

        let words = self
            .index
            .as_ref()
            .map_or("indexing\u{2026}".to_string(), |i| {
                format!("{} words", thousands(i.words()))
            });
        f.render_widget(
            Line::from(vec![
                " full-auto-supercut ".black().on_yellow().bold(),
                format!("  {}", self.project.folder_name()).bold(),
                format!(
                    "  \u{00b7}  {} recordings \u{00b7} {words}",
                    self.project.recs.len()
                )
                .fg(DIM),
            ]),
            header,
        );

        let selected = TABS.iter().position(|(t, _)| *t == self.tab).unwrap();
        f.render_widget(
            Tabs::new(
                TABS.iter()
                    .enumerate()
                    .map(|(i, (_, name))| format!(" F{} {name} ", i + 1)),
            )
            .select(selected)
            .highlight_style(Style::new().fg(Color::Black).bg(ACCENT).bold())
            .divider(" ")
            .padding("", ""),
            tabs.inner(ratatui::layout::Margin::new(1, 0)),
        );

        match self.tab {
            Tab::Phrases => self.draw_phrases(f, body),
            Tab::Say => self.draw_say(f, body),
            Tab::Recordings => self.draw_recordings(f, body),
        }
        self.draw_jobs(f, jobs);

        let keys: &[(&str, &str)] = match self.tab {
            Tab::Phrases => &[
                ("\u{2191}\u{2193}", "select"),
                ("type", "custom phrase"),
                ("Enter", "make supercut"),
                ("Ctrl+V", "verify"),
                ("Ctrl+O", "play last"),
            ],
            Tab::Say => &[
                ("type", "sentence"),
                ("\u{2191}\u{2193}", "piece"),
                ("Tab", "other take"),
                ("Ctrl+P", "preview"),
                ("Enter", "build"),
                ("Ctrl+K", "captions"),
                ("Ctrl+O", "play last"),
            ],
            Tab::Recordings => &[
                ("\u{2191}\u{2193}", "select"),
                ("Enter", "play"),
                ("Ctrl+T", "transcribe pending"),
            ],
        };
        let mut spans = vec![];
        for (k, what) in keys.iter().chain(&[("F1-F3", "tabs"), ("Ctrl+Q", "quit")]) {
            spans.push(format!(" {k} ").black().on_gray());
            spans.push(format!(" {what}  ").fg(DIM));
        }
        f.render_widget(Line::from(spans), footer);
    }

    fn draw_phrases(&mut self, f: &mut Frame, area: Rect) {
        let [custom, table] =
            Layout::vertical([Constraint::Length(3), Constraint::Min(3)]).areas(area);
        let count = self
            .index
            .as_ref()
            .filter(|_| !self.custom.trim().is_empty())
            .map(|i| i.count(&self.custom));
        let hint = match count {
            None if self.custom.is_empty() => {
                "type to supercut any phrase, or pick one below".fg(DIM)
            }
            None => "".into(),
            Some(0) => "  never said".fg(MISSING),
            Some(n) => {
                format!("  {n} occurrence{}", if n == 1 { "" } else { "s" }).fg(CHUNK_COLORS[0])
            }
        };
        f.render_widget(
            Paragraph::new(Line::from(vec![self.custom.clone().bold(), hint]))
                .block(panel("Custom phrase")),
            custom,
        );
        if !self.custom.is_empty() {
            f.set_cursor_position((
                custom.x + 1 + self.custom.chars().count() as u16,
                custom.y + 1,
            ));
        }

        let rows = self.cands.iter().enumerate().map(|(i, c)| {
            Row::new(vec![
                Cell::from(format!("{:>2}", i + 1)).fg(DIM),
                Cell::from(format!("\u{201c}{}\u{201d}", c.phrase)).bold(),
                Cell::from(format!("{:>5}", c.count)),
                Cell::from(format!("{:>2}/{:<2}", c.habitual, c.per_recording.len())),
                Cell::from(format!("{:>4.0}\u{00d7}", c.lift)),
                Cell::from(format!("{:>5.1}", c.score)).fg(ACCENT),
                Cell::from(sparkline(&c.per_recording)).fg(CHUNK_COLORS[1]),
            ])
        });
        let header = Row::new([
            "#",
            "phrase",
            "count",
            "habit",
            "lift",
            "score",
            "per recording",
        ])
        .fg(DIM);
        let title = if self.cands.is_empty() && self.index.is_none() {
            "Catchphrases (ranking\u{2026})"
        } else {
            "Catchphrases"
        };
        f.render_stateful_widget(
            Table::new(
                rows,
                [
                    Constraint::Length(3),
                    Constraint::Min(22),
                    Constraint::Length(6),
                    Constraint::Length(6),
                    Constraint::Length(6),
                    Constraint::Length(6),
                    Constraint::Length(self.project.recs.len() as u16 + 2),
                ],
            )
            .header(header)
            .row_highlight_style(Style::new().bg(Color::Rgb(50, 50, 60)))
            .highlight_symbol("\u{25b6} ")
            .block(panel(title)),
            table,
            &mut self.phrase_table,
        );
    }

    fn draw_say(&mut self, f: &mut Frame, area: Rect) {
        let [input_area, pieces_area, opts_area] = Layout::vertical([
            Constraint::Length(5),
            Constraint::Min(4),
            Constraint::Length(1),
        ])
        .areas(area);

        // Sentence, coloured by piece: alternating colours per piece, red for words never said.
        let spans_at = word_spans(&self.input);
        let mut owner = vec![None; spans_at.len()];
        for (k, c) in self.plan.chunks.iter().enumerate() {
            for o in owner.iter_mut().take(c.span.1).skip(c.span.0) {
                *o = Some(k);
            }
        }
        let mut line = Vec::new();
        let mut at = 0;
        for (w, &(a, b)) in spans_at.iter().enumerate() {
            line.push(Span::raw(self.input[at..a].to_string()));
            let style = match owner
                .get(w)
                .copied()
                .flatten()
                .map(|k| (k, &self.plan.chunks[k]))
            {
                Some((_, c)) if c.missing() => Style::new().fg(MISSING).bold().underlined(),
                Some((k, _)) => {
                    let s = Style::new().fg(CHUNK_COLORS[k % 2]).bold();
                    if k == self.chunk_sel {
                        s.underlined()
                    } else {
                        s
                    }
                }
                None => Style::new(),
            };
            line.push(Span::styled(self.input[a..b].to_string(), style));
            at = b;
        }
        line.push(Span::raw(self.input[at..].to_string()));
        let missing = self.plan.missing_words();
        let status = if self.input.trim().is_empty() {
            Line::from(
                "Type anything. Words light up when someone in the recordings said them.".fg(DIM),
            )
        } else if !missing.is_empty() {
            let mut spans = vec![format!("never said: {}", missing.join(", ")).fg(MISSING)];
            let tips: Vec<String> = match &self.index {
                Some(index) => missing
                    .iter()
                    .flat_map(|w| index.suggest(&text::norm_exact(w), 3))
                    .collect(),
                None => Vec::new(),
            };
            if !tips.is_empty() {
                spans.push(format!("   did you mean: {}", tips.join(", ")).fg(DIM));
            }
            Line::from(spans)
        } else {
            let n = self.plan.chunks.len();
            let words = self.plan.chunks.iter().map(|c| c.toks.len()).sum::<usize>();
            Line::from(
                format!(
                    "\u{2713} all {words} words found \u{00b7} {n} piece{} \u{00b7} Enter to build",
                    if n == 1 { "" } else { "s" }
                )
                .fg(CHUNK_COLORS[0]),
            )
        };
        f.render_widget(
            Paragraph::new(Text::from(vec![
                Line::from(line).bold(),
                Line::raw(""),
                status,
            ]))
            .wrap(Wrap { trim: false })
            .block(panel("Sentence")),
            input_area,
        );
        let cursor_col = self.input[..self.cursor].chars().count() as u16;
        let inner_w = input_area.width.saturating_sub(2).max(1);
        f.set_cursor_position((
            input_area.x + 1 + cursor_col % inner_w,
            input_area.y + 1 + cursor_col / inner_w,
        ));

        // Pieces and the take chosen for each.
        let rows: Vec<Row> = self
            .plan
            .chunks
            .iter()
            .enumerate()
            .map(|(k, c)| {
                let color = if c.missing() {
                    MISSING
                } else {
                    CHUNK_COLORS[k % 2]
                };
                match (c.takes.get(c.choice), &self.index) {
                    (Some(t), Some(index)) => {
                        let (s, _) = index.take_window(c, t);
                        Row::new(vec![
                            Cell::from(format!("\u{201c}{}\u{201d}", c.typed))
                                .fg(color)
                                .bold(),
                            Cell::from(format!(
                                "{} take{}",
                                c.found,
                                if c.found == 1 { "" } else { "s" }
                            )),
                            Cell::from(format!("{}/{}", c.choice + 1, c.takes.len())).fg(DIM),
                            Cell::from(format!("{} {}", index.docs[t.doc].0.label, clock(s))),
                            Cell::from(index.take_context(c, t)).fg(DIM),
                        ])
                    }
                    _ => Row::new(vec![
                        Cell::from(format!("\u{201c}{}\u{201d}", c.typed))
                            .fg(color)
                            .bold(),
                        Cell::from("never said").fg(MISSING),
                        Cell::from(""),
                        Cell::from(""),
                        Cell::from(match &self.index {
                            Some(index) => {
                                let tips = index.suggest(&c.toks.join(" "), 4);
                                if tips.is_empty() {
                                    String::new()
                                } else {
                                    format!("try: {}", tips.join(", "))
                                }
                            }
                            None => String::new(),
                        })
                        .fg(DIM),
                    ]),
                }
            })
            .collect();
        let mut state =
            TableState::default().with_selected((!rows.is_empty()).then_some(self.chunk_sel));
        f.render_stateful_widget(
            Table::new(
                rows,
                [
                    Constraint::Min(16),
                    Constraint::Length(11),
                    Constraint::Length(6),
                    Constraint::Length(20),
                    Constraint::Fill(2),
                ],
            )
            .header(Row::new(["piece", "found", "take", "from", "context"]).fg(DIM))
            .row_highlight_style(Style::new().bg(Color::Rgb(50, 50, 60)))
            .highlight_symbol("\u{25b6} ")
            .block(panel("Pieces")),
            pieces_area,
            &mut state,
        );

        let flag = |on: bool, name: &str| {
            if on {
                format!(" {name} on ").black().on_green()
            } else {
                format!(" {name} off ").fg(DIM).on_black()
            }
        };
        f.render_widget(
            Line::from(vec![
                " ".into(),
                flag(self.captions, "captions"),
                " ".into(),
                flag(self.verify, "verify"),
                " ".into(),
                flag(self.autoplay, "autoplay"),
                format!(
                    "   saving to {}",
                    self.out_dir
                        .canonicalize()
                        .unwrap_or(self.out_dir.clone())
                        .display()
                )
                .fg(DIM),
            ]),
            opts_area,
        );
    }

    fn draw_recordings(&mut self, f: &mut Frame, area: Rect) {
        let rows = self.project.recs.iter().zip(&self.rec_words).map(|(r, w)| {
            let status = match w {
                None => "not transcribed".fg(ACCENT),
                Some(0) => "no speech".fg(DIM),
                Some(n) => format!("{} words", thousands(*n)).fg(CHUNK_COLORS[0]),
            };
            Row::new(vec![
                Cell::from(r.label.clone()).bold(),
                Cell::from(r.id.clone()).fg(DIM),
                Cell::from(status),
                Cell::from(
                    r.video
                        .file_name()
                        .unwrap_or_default()
                        .to_string_lossy()
                        .to_string(),
                )
                .fg(DIM),
            ])
        });
        f.render_stateful_widget(
            Table::new(
                rows,
                [
                    Constraint::Length(14),
                    Constraint::Length(10),
                    Constraint::Length(16),
                    Constraint::Fill(1),
                ],
            )
            .header(Row::new(["recording", "id", "transcript", "file"]).fg(DIM))
            .row_highlight_style(Style::new().bg(Color::Rgb(50, 50, 60)))
            .highlight_symbol("\u{25b6} ")
            .block(panel("Recordings")),
            area,
            &mut self.rec_table,
        );
    }

    fn draw_jobs(&self, f: &mut Frame, area: Rect) {
        let block = panel(match &self.job {
            Some(j) => format!("Working: {}", j.title),
            None => "Log".into(),
        });
        let inner = block.inner(area);
        f.render_widget(block, area);
        let [gauge, log] = if self.job.is_some() {
            Layout::vertical([Constraint::Length(1), Constraint::Min(1)]).areas(inner)
        } else {
            [Rect::default(), inner]
        };
        if let Some(j) = &self.job {
            let ratio = if j.total > 0 {
                j.done as f64 / j.total as f64
            } else {
                0.0
            };
            let spinner = [
                "\u{280b}", "\u{2819}", "\u{2839}", "\u{2838}", "\u{283c}", "\u{2834}", "\u{2826}",
                "\u{2827}", "\u{2807}", "\u{280f}",
            ][(j.started.elapsed().as_millis() / 100) as usize % 10];
            let label = if j.total > 0 {
                format!("{spinner} {}/{} ", j.done, j.total)
            } else {
                format!("{spinner} ")
            };
            f.render_widget(
                LineGauge::default()
                    .ratio(ratio.min(1.0))
                    .label(label)
                    .filled_style(Style::new().fg(ACCENT))
                    .unfilled_style(Style::new().fg(Color::Rgb(60, 60, 60))),
                gauge,
            );
        }
        let n = log.height as usize;
        let lines: Vec<Line> = self
            .log
            .iter()
            .skip(self.log.len().saturating_sub(n))
            .map(|l| {
                let style =
                    if l.starts_with('\u{2717}') || l.starts_with("can't") || l.starts_with("drop")
                    {
                        Style::new().fg(MISSING)
                    } else if l.starts_with('\u{2713}') {
                        Style::new().fg(CHUNK_COLORS[0])
                    } else {
                        Style::new().fg(Color::Gray)
                    };
                Line::styled(l.clone(), style)
            })
            .collect();
        f.render_widget(Paragraph::new(lines), log);
    }
}

fn panel<'a>(title: impl Into<Line<'a>>) -> Block<'a> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(Color::Rgb(80, 80, 90)))
        .title(title.into().bold().fg(ACCENT))
}

fn select(state: &mut TableState, n: usize, by: isize) {
    if n == 0 {
        return;
    }
    let i = state.selected().unwrap_or(0) as isize + by;
    state.select(Some(i.clamp(0, n as isize - 1) as usize));
}

/// Byte ranges of the words in `s` that the planner sees (same filter as `Index::plan`).
fn word_spans(s: &str) -> Vec<(usize, usize)> {
    let mut out = Vec::new();
    let mut start = None;
    for (i, c) in s.char_indices().chain(std::iter::once((s.len(), ' '))) {
        match (c.is_whitespace(), start) {
            (false, None) => start = Some(i),
            (true, Some(a)) => {
                if !text::norm_exact(&s[a..i]).is_empty() {
                    out.push((a, i));
                }
                start = None;
            }
            _ => {}
        }
    }
    out
}

fn sparkline(counts: &[usize]) -> String {
    const BARS: [char; 8] = [
        '\u{2581}', '\u{2582}', '\u{2583}', '\u{2584}', '\u{2585}', '\u{2586}', '\u{2587}',
        '\u{2588}',
    ];
    let max = counts.iter().copied().max().unwrap_or(0).max(1) as f64;
    counts
        .iter()
        .map(|&c| {
            if c == 0 {
                ' '
            } else {
                BARS[((c as f64 / max) * 7.0).round() as usize]
            }
        })
        .collect()
}

fn clock(s: f64) -> String {
    let s = s as u64;
    if s >= 3600 {
        format!("{}:{:02}:{:02}", s / 3600, s / 60 % 60, s % 60)
    } else {
        format!("{}:{:02}", s / 60, s % 60)
    }
}

fn thousands(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::new();
    for (i, c) in s.chars().enumerate() {
        if i > 0 && (s.len() - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(c);
    }
    out
}
