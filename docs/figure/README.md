# The validation figure

`cargo xtask figure` draws the figure from `results.json` into `target/figure/figure.svg`
(`--data FILE`, `--out DIR`). It is written in Rust with std and serde_json only, so it needs
nothing beyond the workspace and gives the same bytes for the same input. The code holds no
number: every value, bar, label and commit is in the data file.

`cargo xtask figure --final` refuses to draw while any case or group is not marked
`"reproduced": true`; without it the figure is drawn with a DRAFT mark. Set a case or group to
true only after its numbers were reproduced at `our_commit` (a clean checkout, one run per
number).

Panel A: one column group per test; y = signed error in percent; a shaded band at the test's
bar (none for a test with no bar); filled marker inside the bar, hollow outside; circle for the
coarse grid, square for the fine grid, diamond for a single grid; a value beyond the axis is
drawn at the edge with an arrow and its printed value. Panel B: one group per rotated or
offset channel, three series side by side (develop, ours, ours with the mass correction), one
marker per wall offset with a line spanning them, the bar as a band.

Schema (all numbers percent): `develop_commit`, `our_commit`, optional `our_branch`, `backend`; `panel_a`: `axis`,
`cases[]` of `label` (`|` starts a new line), `bar` (number or null), `reproduced`,
`points[]` of `grid` (`coarse`, `fine` or `single`) and `value`; `panel_b`: `axis`, `bar`,
`groups[]` of `label`, `reproduced`, `series[]` of `name` and `values[]`. `source` fields say
where a case's numbers come from; they are not drawn.
