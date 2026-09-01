// Structural validation of every published data file, plus a second
// independent rebuild of the README's gate table.
//
// The seven JSON files under reports/ and app_data/ are the evidence for every
// number in the README. Nothing checked that they are well formed. A truncated
// write, a key written twice, an interval whose ends are the wrong way round or
// a NaN that leaked out of a division would all be invisible until somebody
// read the table and believed it.
//
// Two things happen here. First a structural pass over all seven files:
// duplicate keys, non finite values outside the two places they are legitimate,
// intervals that are not intervals, counts that are negative or impossible, and
// the row counts the README states. Then the gate table is rebuilt from
// app_data/grid_all.json and compared against reports/harness_all.json, which
// verify/gate.sql also does in SQL, so an error would have to be made the same
// way twice.
package main

import (
	"bytes"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Python writes bare NaN and Infinity, which no JSON parser is obliged to
// accept. They are legitimate in exactly two places: the direct method has no
// standard error, and one scenario has a zero estimate so its interval width
// ratio divides by zero. Anywhere else they are a bug.
var allowedNonFinite = map[string]map[string]bool{
	"ope_all.json":  {"se": true, "ci95": true},
	"grid_all.json": {"ci_width_over_estimate": true},
}

var files = []string{
	"reports/baseline_all.json",
	"reports/eda_all.json",
	"reports/harness_all.json",
	"reports/ope_all.json",
	"reports/prepare.json",
	"reports/stress_all.json",
	"app_data/grid_all.json",
}

// The truth values are constants of the dataset and appear in six of the seven
// files. They must be the same bits everywhere; a file regenerated against a
// different slice of the data would show up here first.
const (
	truthBTS    = "$.ctr_bts"
	tolExact    = 0.0
	unloggedCap = 0.01
	essFloor    = 1000.0
)

type problem struct{ file, msg string }

var problems []problem

func fail(file, format string, a ...any) {
	problems = append(problems, problem{file, fmt.Sprintf(format, a...)})
}

// ------------------------------------------------------------------ text ---

// scrub replaces bare NaN / Infinity / -Infinity tokens with null and returns
// the key each one sat under, so the caller can check they are where they
// belong. String contents are skipped, so a name containing the word Infinity
// would not be rewritten.
func scrub(src []byte) ([]byte, []string) {
	var out bytes.Buffer
	var keys []string
	inString := false
	for i := 0; i < len(src); {
		c := src[i]
		if inString {
			if c == '\\' && i+1 < len(src) {
				out.WriteByte(c)
				out.WriteByte(src[i+1])
				i += 2
				continue
			}
			if c == '"' {
				inString = false
			}
			out.WriteByte(c)
			i++
			continue
		}
		if c == '"' {
			inString = true
			out.WriteByte(c)
			i++
			continue
		}
		matched := ""
		for _, tok := range []string{"-Infinity", "Infinity", "NaN"} {
			if bytes.HasPrefix(src[i:], []byte(tok)) {
				matched = tok
				break
			}
		}
		if matched == "" {
			out.WriteByte(c)
			i++
			continue
		}
		keys = append(keys, keyBefore(src, i))
		out.WriteString("null")
		i += len(matched)
	}
	return out.Bytes(), keys
}

// keyBefore finds the nearest "key": to the left of pos.
func keyBefore(src []byte, pos int) string {
	head := src[:pos]
	colon := bytes.LastIndexByte(head, ':')
	if colon < 0 {
		return "?"
	}
	closeQ := bytes.LastIndexByte(head[:colon], '"')
	if closeQ < 0 {
		return "?"
	}
	openQ := bytes.LastIndexByte(head[:closeQ], '"')
	if openQ < 0 {
		return "?"
	}
	return string(head[openQ+1 : closeQ])
}

// duplicateKeys walks the token stream, because encoding/json silently keeps
// the last of two keys with the same name and a file with two "value" fields
// would decode without complaint.
func duplicateKeys(name string, src []byte) {
	dec := json.NewDecoder(bytes.NewReader(src))
	var walk func(path string) error
	walk = func(path string) error {
		tok, err := dec.Token()
		if err != nil {
			return err
		}
		switch d := tok.(type) {
		case json.Delim:
			if d == '{' {
				seen := map[string]bool{}
				for dec.More() {
					kt, err := dec.Token()
					if err != nil {
						return err
					}
					k, _ := kt.(string)
					if seen[k] {
						fail(name, "key %q appears twice under %s", k, path)
					}
					seen[k] = true
					if err := walk(path + "." + k); err != nil {
						return err
					}
				}
				_, err := dec.Token()
				return err
			}
			if d == '[' {
				for i := 0; dec.More(); i++ {
					if err := walk(fmt.Sprintf("%s[%d]", path, i)); err != nil {
						return err
					}
				}
				_, err := dec.Token()
				return err
			}
		}
		return nil
	}
	if err := walk("$"); err != nil && err != io.EOF {
		fail(name, "token walk stopped: %v", err)
	}
}

// ------------------------------------------------------------ structure ---

// walkValues visits every number in the tree with the key it sat under and the
// path to it.
func walkValues(v any, path, key string, visit func(path, key string, f float64)) {
	switch t := v.(type) {
	case map[string]any:
		keys := make([]string, 0, len(t))
		for k := range t {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			walkValues(t[k], path+"."+k, k, visit)
		}
	case []any:
		for i, e := range t {
			walkValues(e, fmt.Sprintf("%s[%d]", path, i), key, visit)
		}
	case float64:
		visit(path, key, t)
	}
}

var countKeys = map[string]bool{
	"rows": true, "clicks": true, "n": true, "n_rows": true, "n_items": true,
	"n_matched": true, "train_rows": true, "test_rows": true, "ess": true,
	"items_dropped": true, "n_scenarios": true, "decisions_correct": true,
	"train_days": true, "shared_days": true, "random_days": true, "bts_days": true,
}

var intervalKeys = map[string]bool{
	"ci95": true, "diff_ci95": true, "would_be_ci": true,
}

func validate(name string, tree any) {
	walkValues(tree, "$", "", func(path, key string, f float64) {
		if math.IsNaN(f) || math.IsInf(f, 0) {
			// scrub already turned the legitimate ones into null, so anything
			// still non finite here came from a literal the parser accepted.
			fail(name, "%s is not finite", path)
		}
		if countKeys[key] && f < 0 {
			fail(name, "%s is a count and is negative (%g)", path, f)
		}
	})

	// Intervals must have two ends the right way round. A null end is the
	// direct method having no standard error, which is allowed.
	var checkIntervals func(v any, path string)
	checkIntervals = func(v any, path string) {
		switch t := v.(type) {
		case map[string]any:
			for k, e := range t {
				if intervalKeys[k] {
					arr, ok := e.([]any)
					if !ok || len(arr) != 2 {
						fail(name, "%s.%s is not a two element interval", path, k)
						continue
					}
					lo, okLo := arr[0].(float64)
					hi, okHi := arr[1].(float64)
					if okLo && okHi && lo > hi {
						fail(name, "%s.%s is inverted: [%g, %g]", path, k, lo, hi)
					}
				}
				checkIntervals(e, path+"."+k)
			}
		case []any:
			for i, e := range t {
				checkIntervals(e, fmt.Sprintf("%s[%d]", path, i))
			}
		}
	}
	checkIntervals(tree, "$")
}

// ---------------------------------------------------------------- helpers ---

func at(tree any, path ...any) any {
	cur := tree
	for _, step := range path {
		switch s := step.(type) {
		case string:
			m, ok := cur.(map[string]any)
			if !ok {
				return nil
			}
			cur = m[s]
		case int:
			a, ok := cur.([]any)
			if !ok || s >= len(a) {
				return nil
			}
			cur = a[s]
		}
	}
	return cur
}

func f64(tree any, path ...any) (float64, bool) {
	v, ok := at(tree, path...).(float64)
	return v, ok
}

func mustF(name string, tree any, path ...any) float64 {
	v, ok := f64(tree, path...)
	if !ok {
		fail(name, "missing number at %v", path)
	}
	return v
}

func arr(tree any, path ...any) []any {
	a, _ := at(tree, path...).([]any)
	return a
}

// ------------------------------------------------------------------ main ---

func main() {
	root := flag.String("root", ".", "repository root")
	flag.Parse()

	trees := map[string]any{}
	fmt.Printf("structural pass over %d published data files\n", len(files))
	for _, rel := range files {
		base := filepath.Base(rel)
		raw, err := os.ReadFile(filepath.Join(*root, rel))
		if err != nil {
			fail(rel, "unreadable: %v", err)
			continue
		}
		clean, nonFinite := scrub(raw)
		for _, k := range nonFinite {
			if !allowedNonFinite[base][k] {
				fail(rel, "non finite value under key %q, which is not one of the "+
					"places that is allowed to have one", k)
			}
		}
		duplicateKeys(rel, clean)
		var tree any
		if err := json.Unmarshal(clean, &tree); err != nil {
			fail(rel, "does not parse: %v", err)
			continue
		}
		validate(rel, tree)
		trees[base] = tree
		fmt.Printf("  %-28s %7d bytes  %d non finite value(s), all expected\n",
			base, len(raw), len(nonFinite))
	}

	eda := trees["eda_all.json"]
	harness := trees["harness_all.json"]
	grid := trees["grid_all.json"]
	ope := trees["ope_all.json"]
	stress := trees["stress_all.json"]
	baseline := trees["baseline_all.json"]
	if eda == nil || harness == nil || grid == nil || ope == nil ||
		stress == nil || baseline == nil {
		report()
		return
	}

	// --- counts the README states -----------------------------------------
	fmt.Printf("\nrow counts, against the shape the README and the code publish\n")
	expect := []struct {
		what string
		got  int
		want int
	}{
		{"gate table rows", len(arr(harness, "rows")), 7},
		{"harness n_scenarios", int(mustF("harness_all.json", harness, "n_scenarios")), 7},
		{"grid scenarios", len(arr(grid, "scenarios")), 15},
		{"shortcut estimators", len(at(baseline, "estimators").(map[string]any)), 4},
		{"clipping sweep rows", len(arr(ope, "clipping")), 8},
		{"broken support steps", len(arr(stress, "broken_support")), 6},
		{"items", int(mustF("eda_all.json", eda, "checks", "n_items")), 80},
		{"shared days", int(mustF("eda_all.json", eda, "checks", "shared_days")), 7},
	}
	for _, e := range expect {
		mark := "ok"
		if e.got != e.want {
			mark = "FAIL"
			fail("counts", "%s is %d, expected %d", e.what, e.got, e.want)
		}
		fmt.Printf("  %-24s %3d   expected %3d   %s\n", e.what, e.got, e.want, mark)
	}

	// --- the truth constants, across files ---------------------------------
	fmt.Printf("\nthe two measured CTRs, across every file that quotes them\n")
	checkConst := func(label string, want float64, refs map[string]float64) {
		agree := 0
		for where, got := range refs {
			if math.Abs(got-want) > tolExact {
				fail(where, "%s is %.17g, eda_all.json says %.17g", label, got, want)
			} else {
				agree++
			}
		}
		fmt.Printf("  %-14s %.17g  identical in %d of %d files\n",
			label, want, agree, len(refs))
	}
	tb := mustF("eda_all.json", eda, "ground_truth", "ctr_bts")
	tr := mustF("eda_all.json", eda, "ground_truth", "ctr_random")
	checkConst("bts CTR", tb, map[string]float64{
		"baseline_all.json": mustF("b", baseline, "truth_bts_ctr"),
		"ope_all.json":      mustF("o", ope, "truth_bts_ctr"),
		"stress_all.json":   mustF("s", stress, "forward", "truth"),
		"harness_all.json":  mustF("h", harness, "rows", 0, "truth"),
		"grid_all.json":     mustF("g", grid, "scenarios", 0, "truth"),
		"eda_all.json":      mustF("e", eda, "bts", "ctr"),
	})
	checkConst("random CTR", tr, map[string]float64{
		"baseline_all.json": mustF("b", baseline, "truth_random_ctr"),
		"stress_all.json":   mustF("s", stress, "reverse", "truth"),
		"harness_all.json":  mustF("h", harness, "rows", 1, "truth"),
		"grid_all.json":     mustF("g", grid, "scenarios", 1, "truth"),
		"eda_all.json":      mustF("e", eda, "random", "ctr"),
	})

	// --- rebuild the gate table --------------------------------------------
	fmt.Printf("\ngate table rebuilt from app_data/grid_all.json\n")
	type entry struct {
		ess, unlogged, truth, relErr, lo, hi float64
	}
	byESS := map[float64]entry{}
	for _, s := range arr(grid, "scenarios") {
		e := entry{
			ess:      mustF("grid_all.json", s, "ess"),
			unlogged: mustF("grid_all.json", s, "unlogged_target_mass"),
			truth:    mustF("grid_all.json", s, "truth"),
			relErr:   mustF("grid_all.json", s, "rel_error"),
			lo:       mustF("grid_all.json", s, "ci95", 0),
			hi:       mustF("grid_all.json", s, "ci95", 1),
		}
		if prev, seen := byESS[e.ess]; seen && prev != e {
			fail("grid_all.json", "two different scenarios share ESS %.10f", e.ess)
		}
		byESS[e.ess] = e
	}

	correct, matched := 0, 0
	for _, r := range arr(harness, "rows") {
		name, _ := at(r, "scenario").(string)
		pubESS := mustF("harness_all.json", r, "ess")
		g, ok := byESS[pubESS]
		if !ok {
			fail("harness_all.json", "%s has no entry in the grid at ESS %.10f",
				name, pubESS)
			continue
		}
		matched++
		status := "ok"
		if g.unlogged > unloggedCap || g.ess < essFloor {
			status = "refuse"
		}
		covers := g.truth >= g.lo && g.truth <= g.hi
		isCorrect := (status == "ok") == covers
		if isCorrect {
			correct++
		}
		bad := []string{}
		if s, _ := at(r, "status").(string); s != status {
			bad = append(bad, fmt.Sprintf("status %s vs %s", status, s))
		}
		if c, _ := at(r, "would_cover_truth").(bool); c != covers {
			bad = append(bad, fmt.Sprintf("coverage %v vs %v", covers, c))
		}
		if c, _ := at(r, "decision_correct").(bool); c != isCorrect {
			bad = append(bad, fmt.Sprintf("verdict %v vs %v", isCorrect, c))
		}
		for _, p := range []struct {
			label string
			got   float64
			want  float64
		}{
			{"unlogged", g.unlogged, mustF("h", r, "unlogged_target_mass")},
			{"rel_error", g.relErr, mustF("h", r, "actual_rel_error")},
			{"ci lo", g.lo, mustF("h", r, "would_be_ci", 0)},
			{"ci hi", g.hi, mustF("h", r, "would_be_ci", 1)},
		} {
			if math.Abs(p.got-p.want) > tolExact {
				bad = append(bad, fmt.Sprintf("%s %.17g vs %.17g", p.label, p.got, p.want))
			}
		}
		mark := "agrees"
		if len(bad) > 0 {
			mark = "FAIL " + strings.Join(bad, "; ")
			fail("harness_all.json", "%s: %s", name, strings.Join(bad, "; "))
		}
		fmt.Printf("  %-40s %-7s %s\n", name, status, mark)
	}
	if matched != len(arr(harness, "rows")) {
		fail("harness_all.json", "only %d of %d rows found in the grid",
			matched, len(arr(harness, "rows")))
	}
	pubCorrect := int(mustF("harness_all.json", harness, "decisions_correct"))
	if correct != pubCorrect {
		fail("harness_all.json", "Go scores %d correct decisions, the file says %d",
			correct, pubCorrect)
	}
	fmt.Printf("  Go scores %d of %d correct, harness_all.json publishes %d of %d\n",
		correct, matched, pubCorrect, int(mustF("h", harness, "n_scenarios")))

	report()
}

func report() {
	if len(problems) == 0 {
		fmt.Printf("\nall %d files are structurally sound and the gate table "+
			"rebuilds exactly\n", len(files))
		return
	}
	fmt.Printf("\n%d problems\n", len(problems))
	for _, p := range problems {
		fmt.Printf("  %s: %s\n", p.file, p.msg)
	}
	os.Exit(1)
}
