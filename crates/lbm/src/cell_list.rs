//! A list of cells in the "fill" text format the studies exchange: a header `fill <kind> <nx> <ny> <nz> <count>`, comment lines
//! starting with `#`, then one linear index `i + nx (j + ny k)` per line. Used for cell sets and exclusion masks of the wall-layer
//! studies ([`crate::Flow::set_wall_corner_cells`], [`crate::Flow::set_wall_corner_exclude`]).

/// The most cells a list may hold (the file is input: hostile numbers are capped).
pub const MAX_CELLS: usize = 1 << 28;

/// Parses a cell list for a grid of `cells` cells per axis: the header's grid must match, every index must lie in the grid, and the
/// count must match the header's.
pub fn parse(text: &str, cells: [usize; 3]) -> Result<Vec<usize>, String> {
    let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty() && !l.starts_with('#'));
    let header = lines.next().ok_or("empty cell list")?;
    let fields: Vec<&str> = header.split_whitespace().collect();
    if fields.first() != Some(&"fill") || fields.len() != 6 {
        return Err(format!("cell list header `{header}` is not `fill <kind> <nx> <ny> <nz> <count>`"));
    }
    let num = |k: usize| {
        fields.get(k).and_then(|f| f.parse::<usize>().ok()).ok_or_else(|| format!("cell list header `{header}`: field {k} is not a number"))
    };
    let dims = [num(2)?, num(3)?, num(4)?];
    let count = num(5)?;
    if dims != cells {
        return Err(format!("cell list is for a {dims:?} grid, the flow's is {cells:?}"));
    }
    if count > MAX_CELLS {
        return Err(format!("cell list claims {count} cells, over the cap of {MAX_CELLS}"));
    }
    let total = cells[0].checked_mul(cells[1]).and_then(|v| v.checked_mul(cells[2])).ok_or("grid too large")?;
    let mut out = Vec::with_capacity(count.min(1 << 20));
    for l in lines {
        let v: usize = l.parse().map_err(|_| format!("cell list line `{l}` is not an index"))?;
        if v >= total {
            return Err(format!("cell list index {v} is outside the {total}-cell grid"));
        }
        if out.len() >= MAX_CELLS {
            return Err(format!("cell list is over the cap of {MAX_CELLS} cells"));
        }
        out.push(v);
    }
    if out.len() != count {
        return Err(format!("cell list header says {count} cells, the file holds {}", out.len()));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_fill_format_and_refuses_bad_lists() {
        let ok = "fill 1 4 3 2 3\n# a comment\n0\n5\n23\n";
        assert_eq!(parse(ok, [4, 3, 2]).unwrap(), vec![0, 5, 23]);
        assert!(parse(ok, [4, 3, 3]).unwrap_err().contains("grid"));
        assert!(parse("fill 1 4 3 2 3\n0\n5\n24\n", [4, 3, 2]).unwrap_err().contains("outside"));
        assert!(parse("fill 1 4 3 2 2\n0\n5\n23\n", [4, 3, 2]).unwrap_err().contains("holds"));
        assert!(parse("fill 1 4 3 2 1\nx\n", [4, 3, 2]).is_err());
        assert!(parse("", [4, 3, 2]).is_err());
        assert!(parse("fill 1 4 3 2 99999999999999999999\n", [4, 3, 2]).is_err());
    }
}
