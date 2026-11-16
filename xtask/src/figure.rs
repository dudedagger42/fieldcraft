//! `cargo xtask figure`: draw the validation figure from a results data file.
//!
//! The figure has two panels. Panel A is the signed error of each validation test against its
//! reference, one column group per test, a shaded band at the test's bar, filled markers inside
//! the bar and hollow ones outside, coarse and fine grids told apart by marker shape. Panel B
//! compares the interpolated-wall rules on the rotated and offset channels: one marker per wall
//! offset, a thin line spanning the offsets, three series side by side.
//!
//! Every number comes from the data file (`docs/figure/results.json`, schema in
//! `docs/figure/README.md`); the script holds none. The output is an SVG written by hand with
//! std only, so there is no plotting dependency and the same input gives the same bytes. A data
//! item not yet reproduced at the proposed commit marks the figure as a draft; `--final` refuses
//! to draw a draft.

use serde_json::Value;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

/// Most cases, points, groups or series one figure accepts (the data file is input).
const MAX_ITEMS: usize = 256;
/// Colour-blind-safe palette (Okabe and Ito): blue, vermillion, bluish green.
const PALETTE: [&str; 3] = ["#0072B2", "#D55E00", "#009E73"];
const INK: &str = "#222222";
const GRID: &str = "#d9d9d9";
const BAND: &str = "#9ec5e6";

const WIDTH: f64 = 1000.0;
const LEFT: f64 = 70.0;
const RIGHT: f64 = 970.0;

struct Point {
    grid: String,
    value: f64,
}

struct Case {
    label: String,
    /// Scores printed under the label (`|` starts a new line); optional.
    note: Option<String>,
    bar: Option<f64>,
    reproduced: bool,
    points: Vec<Point>,
}

struct Series {
    name: String,
    values: Vec<f64>,
}

struct Group {
    label: String,
    reproduced: bool,
    series: Vec<Series>,
}

struct Data {
    develop: String,
    ours: String,
    backend: String,
    axis_a: f64,
    cases: Vec<Case>,
    axis_b: f64,
    bar_b: f64,
    groups: Vec<Group>,
}

fn get<'a>(v: &'a Value, key: &str) -> Result<&'a Value, String> {
    v.get(key).ok_or_else(|| format!("the data file has no `{key}`"))
}

fn text(v: &Value, key: &str) -> Result<String, String> {
    get(v, key)?.as_str().map(str::to_string).ok_or_else(|| format!("`{key}` must be a string"))
}

fn finite(v: &Value, what: &str) -> Result<f64, String> {
    match v.as_f64() {
        Some(x) if x.is_finite() => Ok(x),
        _ => Err(format!("{what} must be a finite number")),
    }
}

fn items<'a>(v: &'a Value, key: &str) -> Result<&'a Vec<Value>, String> {
    let a = get(v, key)?.as_array().ok_or_else(|| format!("`{key}` must be a list"))?;
    if a.len() > MAX_ITEMS {
        return Err(format!("`{key}` has {} entries; at most {MAX_ITEMS} are drawn", a.len()));
    }
    Ok(a)
}

fn flag(v: &Value, key: &str) -> Result<bool, String> {
    get(v, key)?.as_bool().ok_or_else(|| format!("`{key}` must be true or false"))
}

fn positive(v: &Value, key: &str) -> Result<f64, String> {
    let x = finite(get(v, key)?, key)?;
    if x > 0.0 { Ok(x) } else { Err(format!("`{key}` must be positive")) }
}

fn parse(v: &Value) -> Result<Data, String> {
    let a = get(v, "panel_a")?;
    let mut cases = Vec::new();
    for c in items(a, "cases")? {
        let label = text(c, "label")?;
        let bar = match c.get("bar") {
            None | Some(Value::Null) => None,
            Some(b) => Some(finite(b, "a case's `bar`")?.abs()),
        };
        let mut points = Vec::new();
        for p in items(c, "points")? {
            points.push(Point { grid: text(p, "grid")?, value: finite(get(p, "value")?, "a point's `value`")? });
        }
        let note = c.get("note").and_then(Value::as_str).map(str::to_string);
        cases.push(Case { reproduced: flag(c, "reproduced")?, label, note, bar, points });
    }
    let b = get(v, "panel_b")?;
    let mut groups = Vec::new();
    for g in items(b, "groups")? {
        let mut series = Vec::new();
        for s in items(g, "series")? {
            let mut values = Vec::new();
            for x in items(s, "values")? {
                values.push(finite(x, "a series value")?);
            }
            series.push(Series { name: text(s, "name")?, values });
        }
        if series.len() > PALETTE.len() {
            return Err(format!("a group has {} series; at most {} are drawn", series.len(), PALETTE.len()));
        }
        groups.push(Group { label: text(g, "label")?, reproduced: flag(g, "reproduced")?, series });
    }
    Ok(Data {
        develop: text(v, "develop_commit")?,
        ours: text(v, "our_commit")?,
        backend: text(v, "backend")?,
        axis_a: positive(a, "axis")?,
        cases,
        axis_b: positive(b, "axis")?,
        bar_b: positive(b, "bar")?,
        groups,
    })
}

fn esc(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;").replace('"', "&quot;")
}

/// The plot rectangle of a panel and its vertical scale: values in [-ymax, ymax] map to
/// [bottom, top]; anything beyond is pinned to the edge by the caller.
struct Frame {
    top: f64,
    bottom: f64,
    ymax: f64,
}

impl Frame {
    fn y(&self, v: f64) -> f64 {
        let c = v.clamp(-self.ymax, self.ymax);
        self.top + (self.ymax - c) / (2.0 * self.ymax) * (self.bottom - self.top)
    }
}

fn text_at(out: &mut String, x: f64, y: f64, size: f64, anchor: &str, weight: &str, s: &str) {
    let _ = writeln!(
        out,
        r#"<text x="{x:.1}" y="{y:.1}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" fill="{INK}">{}</text>"#,
        esc(s)
    );
}

fn axes(out: &mut String, f: &Frame, title: &str) {
    let step = if f.ymax > 8.0 { 2.0 } else { 1.0 };
    let mut t = -(f.ymax / step).floor() * step;
    let mut guard = 0;
    while t <= f.ymax + 1e-9 && guard < 64 {
        let y = f.y(t);
        let (colour, w) = if t.abs() < 1e-9 { (INK, 1.2) } else { (GRID, 0.8) };
        let _ = writeln!(out, r#"<line x1="{LEFT}" y1="{y:.1}" x2="{RIGHT}" y2="{y:.1}" stroke="{colour}" stroke-width="{w}"/>"#);
        let tick = if t.abs() < 1e-9 { "0".to_string() } else { format!("{t:+.0}") };
        text_at(out, LEFT - 8.0, y + 4.0, 12.0, "end", "normal", &tick);
        t += step;
        guard += 1;
    }
    let cy = (f.top + f.bottom) / 2.0;
    let _ = writeln!(
        out,
        r#"<text x="22" y="{cy:.1}" font-size="12" text-anchor="middle" fill="{INK}" transform="rotate(-90 22 {cy:.1})">error against the reference (percent)</text>"#
    );
    text_at(out, LEFT, f.top - 34.0, 15.0, "start", "bold", title);
    let _ = writeln!(
        out,
        r#"<rect x="{LEFT}" y="{:.1}" width="{:.1}" height="{:.1}" fill="none" stroke="{INK}" stroke-width="1"/>"#,
        f.top,
        RIGHT - LEFT,
        f.bottom - f.top
    );
}

/// One marker centred on (x, y): shape 0 circle, 1 square, 2 diamond, 3 triangle.
fn marker(out: &mut String, shape: usize, x: f64, y: f64, colour: &str, filled: bool) {
    let fill = if filled { colour } else { "#ffffff" };
    let style = format!(r#"fill="{fill}" stroke="{colour}" stroke-width="2""#);
    let r = 5.5;
    let _ = match shape {
        0 => writeln!(out, r#"<circle cx="{x:.1}" cy="{y:.1}" r="{r}" {style}/>"#),
        1 => writeln!(out, r#"<rect x="{:.1}" y="{:.1}" width="{:.1}" height="{:.1}" {style}/>"#, x - r, y - r, 2.0 * r, 2.0 * r),
        2 => writeln!(
            out,
            r#"<polygon points="{x:.1},{:.1} {:.1},{y:.1} {x:.1},{:.1} {:.1},{y:.1}" {style}/>"#,
            y - r - 1.0,
            x + r + 1.0,
            y + r + 1.0,
            x - r - 1.0
        ),
        _ => {
            writeln!(out, r#"<polygon points="{x:.1},{:.1} {:.1},{:.1} {:.1},{:.1}" {style}/>"#, y - r - 1.0, x + r + 1.0, y + r, x - r - 1.0, y + r)
        }
    };
}

/// A marker pinned to the panel edge with an arrow and its printed value, for a value off the scale.
fn off_scale(out: &mut String, f: &Frame, x: f64, v: f64, colour: &str) {
    let up = v > 0.0;
    let y = if up { f.top + 12.0 } else { f.bottom - 12.0 };
    let tip = if up { y - 9.0 } else { y + 9.0 };
    let _ = writeln!(
        out,
        r#"<polygon points="{x:.1},{tip:.1} {:.1},{y:.1} {:.1},{y:.1}" fill="{colour}" stroke="{colour}" stroke-width="1"/>"#,
        x - 6.0,
        x + 6.0
    );
    let ty = if up { y + 14.0 } else { y - 6.0 };
    text_at(out, x, ty, 12.0, "middle", "bold", &format!("{v:+.1} %"));
}

fn labels(out: &mut String, x: f64, y: f64, label: &str) {
    for (i, line) in label.split('|').enumerate() {
        text_at(out, x, y + 14.0 * i as f64, 11.0, "middle", "normal", line);
    }
}

fn panel_a(out: &mut String, d: &Data, f: &Frame) {
    axes(out, f, "A. Error against the reference, each validation test");
    let n = d.cases.len().max(1) as f64;
    let slot = (RIGHT - LEFT) / n;
    for (i, c) in d.cases.iter().enumerate() {
        let x0 = LEFT + slot * i as f64;
        if let Some(bar) = c.bar {
            let (yt, yb) = (f.y(bar), f.y(-bar));
            let _ = writeln!(
                out,
                r#"<rect x="{:.1}" y="{yt:.1}" width="{:.1}" height="{:.1}" fill="{BAND}" fill-opacity="0.45"/>"#,
                x0 + 4.0,
                slot - 8.0,
                yb - yt
            );
        }
        let np = c.points.len() as f64 + 1.0;
        for (j, p) in c.points.iter().enumerate() {
            let x = x0 + slot * (j as f64 + 1.0) / np;
            let inside = c.bar.is_some_and(|b| p.value.abs() <= b);
            let colour = if inside { PALETTE[0] } else { PALETTE[1] };
            if p.value.abs() > f.ymax {
                off_scale(out, f, x, p.value, colour);
            } else {
                let shape = match p.grid.as_str() {
                    "coarse" => 0,
                    "fine" => 1,
                    _ => 2,
                };
                marker(out, shape, x, f.y(p.value), colour, inside);
            }
        }
        let full = match &c.note {
            Some(n) => format!("{}|{n}", c.label),
            None => c.label.clone(),
        };
        labels(out, x0 + slot / 2.0, f.bottom + 18.0, &full);
        if i > 0 {
            let _ = writeln!(
                out,
                r#"<line x1="{x0:.1}" y1="{:.1}" x2="{x0:.1}" y2="{:.1}" stroke="{GRID}" stroke-width="0.8" stroke-dasharray="3 3"/>"#,
                f.top, f.bottom
            );
        }
    }
}

fn panel_b(out: &mut String, d: &Data, f: &Frame) {
    axes(out, f, "B. Interpolated walls: develop against ours");
    let (yt, yb) = (f.y(d.bar_b), f.y(-d.bar_b));
    let _ = writeln!(out, r#"<rect x="{LEFT}" y="{yt:.1}" width="{:.1}" height="{:.1}" fill="{BAND}" fill-opacity="0.45"/>"#, RIGHT - LEFT, yb - yt);
    let n = d.groups.len().max(1) as f64;
    let slot = (RIGHT - LEFT) / n;
    for (i, g) in d.groups.iter().enumerate() {
        let x0 = LEFT + slot * i as f64;
        let ns = g.series.len() as f64 + 1.0;
        for (j, s) in g.series.iter().enumerate() {
            let x = x0 + slot * (j as f64 + 1.0) / ns;
            let colour = PALETTE.get(j).copied().unwrap_or(INK);
            let lo = s.values.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = s.values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            if lo.is_finite() && hi.is_finite() {
                let _ =
                    writeln!(out, r#"<line x1="{x:.1}" y1="{:.1}" x2="{x:.1}" y2="{:.1}" stroke="{colour}" stroke-width="1.5"/>"#, f.y(hi), f.y(lo));
            }
            for &v in &s.values {
                if v.abs() > f.ymax {
                    off_scale(out, f, x, v, colour);
                } else {
                    marker(out, j, x, f.y(v), colour, v.abs() <= d.bar_b);
                }
            }
        }
        labels(out, x0 + slot / 2.0, f.bottom + 18.0, &g.label);
        if i > 0 {
            let _ = writeln!(
                out,
                r#"<line x1="{x0:.1}" y1="{:.1}" x2="{x0:.1}" y2="{:.1}" stroke="{GRID}" stroke-width="0.8" stroke-dasharray="3 3"/>"#,
                f.top, f.bottom
            );
        }
    }
    // The legend names the series once, taken from the first group that has them.
    if let Some(g) = d.groups.iter().find(|g| !g.series.is_empty()) {
        for (j, s) in g.series.iter().enumerate() {
            let x = LEFT + 8.0 + 250.0 * j as f64;
            let colour = PALETTE.get(j).copied().unwrap_or(INK);
            marker(out, j, x + 6.0, f.top - 14.0, colour, true);
            text_at(out, x + 20.0, f.top - 10.0, 12.0, "start", "normal", &s.name);
        }
    }
}

/// The figure as SVG text; marked as a draft when any item is not yet reproduced.
fn render(d: &Data) -> String {
    let frame_a = Frame { top: 130.0, bottom: 370.0, ymax: d.axis_a };
    let frame_b = Frame { top: 570.0, bottom: 810.0, ymax: d.axis_b };
    let height = 900.0;
    let mut out = String::new();
    let _ = writeln!(
        out,
        r#"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}" width="{WIDTH}" height="{height}" font-family="sans-serif">"#
    );
    let _ = writeln!(out, r##"<rect width="{WIDTH}" height="{height}" fill="#ffffff"/>"##);
    text_at(&mut out, LEFT, 24.0, 16.0, "start", "bold", &format!("Validation scores: develop {} and ours {} ({})", d.develop, d.ours, d.backend));
    text_at(
        &mut out,
        LEFT,
        44.0,
        12.0,
        "start",
        "normal",
        "Shaded band: the bar. Filled marker: inside the bar. Hollow marker: outside it. Arrow: off the scale, value printed.",
    );
    text_at(
        &mut out,
        LEFT,
        62.0,
        12.0,
        "start",
        "normal",
        "Panel A: circle = coarse grid, square = fine grid, diamond = one grid. Panel B: one marker per wall offset.",
    );
    panel_a(&mut out, d, &frame_a);
    panel_b(&mut out, d, &frame_b);
    if is_draft(d) {
        text_at(&mut out, LEFT, 82.0, 13.0, "start", "bold", "DRAFT: not all values are reproduced at the proposed commit");
    }
    out.push_str("</svg>\n");
    out
}

/// True when any item of the data is not marked reproduced at the proposed commit.
fn is_draft(d: &Data) -> bool {
    d.cases.iter().any(|c| !c.reproduced) || d.groups.iter().any(|g| !g.reproduced)
}

/// `cargo xtask figure [--data FILE] [--out DIR] [--final]`.
pub fn run(root: &Path, args: &[&str]) -> Result<(), String> {
    let mut data: PathBuf = root.join("docs/figure/results.json");
    let mut out_dir: PathBuf = root.join("target/figure");
    let mut final_only = false;
    let mut it = args.iter();
    while let Some(&a) = it.next() {
        match a {
            "--data" => data = PathBuf::from(it.next().ok_or("--data needs a path")?),
            "--out" => out_dir = PathBuf::from(it.next().ok_or("--out needs a directory")?),
            "--final" => final_only = true,
            other => return Err(format!("figure: unknown argument `{other}`")),
        }
    }
    let src = std::fs::read_to_string(&data).map_err(|e| format!("figure: cannot read {}: {e}", data.display()))?;
    let json: Value = serde_json::from_str(&src).map_err(|e| format!("figure: {} is not JSON: {e}", data.display()))?;
    let d = parse(&json).map_err(|e| format!("figure: {}: {e}", data.display()))?;
    if final_only && is_draft(&d) {
        return Err("figure: --final: some items are not marked reproduced at the proposed commit".into());
    }
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("figure: cannot create {}: {e}", out_dir.display()))?;
    let path = out_dir.join("figure.svg");
    std::fs::write(&path, render(&d)).map_err(|e| format!("figure: cannot write {}: {e}", path.display()))?;
    println!("wrote {}{}", path.display(), if is_draft(&d) { " (draft)" } else { "" });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(reproduced: bool, value: &str) -> Value {
        let s = format!(
            r#"{{"develop_commit":"aaaa","our_commit":"bbbb","backend":"CPU f64",
            "panel_a":{{"axis":5,"cases":[{{"label":"one|two","bar":1,"reproduced":{reproduced},
              "points":[{{"grid":"coarse","value":{value}}},{{"grid":"fine","value":-7.7}}]}}]}},
            "panel_b":{{"axis":5,"bar":1,"groups":[{{"label":"g","reproduced":{reproduced},
              "series":[{{"name":"a","values":[0.1,-0.2]}},{{"name":"b","values":[0.3]}}]}}]}}}}"#
        );
        serde_json::from_str(&s).unwrap_or(Value::Null)
    }

    #[test]
    fn draws_both_panels_and_marks_off_scale_values() -> Result<(), String> {
        let d = parse(&sample(true, "0.5"))?;
        let svg = render(&d);
        assert!(svg.starts_with("<svg") && svg.trim_end().ends_with("</svg>"));
        assert!(svg.contains("A. Error against the reference") && svg.contains("B. Interpolated walls"));
        assert!(svg.contains("-7.7 %"), "the off-scale value is printed");
        assert!(!svg.contains("DRAFT"));
        assert_eq!(svg, render(&d), "the same input gives the same bytes");
        Ok(())
    }

    #[test]
    fn an_unreproduced_item_makes_a_draft() -> Result<(), String> {
        let d = parse(&sample(false, "0.5"))?;
        assert!(is_draft(&d));
        assert!(render(&d).contains("DRAFT"));
        Ok(())
    }

    #[test]
    fn hostile_data_is_an_error_not_a_panic() {
        assert!(parse(&Value::Null).is_err());
        let mut v = sample(true, "0.5");
        v["panel_a"]["axis"] = Value::from(0);
        assert!(parse(&v).is_err(), "a zero axis is refused");
        let mut v = sample(true, "0.5");
        v["panel_b"]["groups"][0]["series"][0]["values"][0] = Value::from("x");
        assert!(parse(&v).is_err(), "a non-number value is refused");
        let mut v = sample(true, "0.5");
        v["panel_a"]["cases"][0]["points"][0]["value"] = Value::from(f64::NAN);
        assert!(parse(&v).is_err(), "a non-finite value is refused");
    }
}
