-- Recompute the gate table in the README from the rawer per scenario file.
--
-- The README's gate table, and its "6 of 7 right", come from
-- reports/harness_all.json, which src/roo/harness.py wrote. app_data/grid_all.json
-- is a layer below it: for fifteen scenarios it records the three scalars the
-- gate actually decides on, plus the interval that would have been reported.
-- The gate is a pure function of those, so the whole published table can be
-- rebuilt from the grid without touching the harness code.
--
--   refuse  if the target policy puts more than 1% of its probability mass on
--           actions absent from the logs, or if the effective sample size is
--           below 1,000. Both are harness.audit()'s defaults.
--   correct if it reported an interval that covers the truth, or refused one
--           that would not have. That is validate()'s grading rule.
--
-- The two files are joined on effective sample size, which identifies a
-- scenario's log set exactly and does not depend on the two files spelling the
-- scenario names the same way. Seven pairs must match; anything else fails.
--
-- Run: sqlite3 :memory: ".read verify/gate.sql"   from the repository root.

.mode list
.headers off

-- Neither file is RFC 8259 JSON: Python writes bare Infinity and NaN for the
-- non finite values it produces. Both become null here, which is what the JSON
-- functions accept, and verify/gocheck asserts they appear nowhere else.
CREATE TEMP TABLE src AS
SELECT 'grid' AS which,
       replace(replace(CAST(readfile('app_data/grid_all.json') AS TEXT),
                       'Infinity', 'null'), 'NaN', 'null') AS t
UNION ALL
SELECT 'harness',
       replace(replace(CAST(readfile('reports/harness_all.json') AS TEXT),
                       'Infinity', 'null'), 'NaN', 'null');

CREATE TEMP TABLE grid AS
SELECT json_extract(value, '$.name')                  AS name,
       json_extract(value, '$.ess')                   AS ess,
       json_extract(value, '$.unlogged_target_mass')  AS unlogged,
       json_extract(value, '$.truth')                 AS truth,
       json_extract(value, '$.rel_error')             AS rel_error,
       json_extract(value, '$.ci95[0]')               AS lo,
       json_extract(value, '$.ci95[1]')               AS hi
FROM src, json_each(src.t, '$.scenarios')
WHERE src.which = 'grid';

CREATE TEMP TABLE published AS
SELECT json_extract(value, '$.scenario')            AS name,
       json_extract(value, '$.status')              AS status,
       json_extract(value, '$.ess')                 AS ess,
       json_extract(value, '$.unlogged_target_mass') AS unlogged,
       json_extract(value, '$.truth')               AS truth,
       json_extract(value, '$.actual_rel_error')    AS rel_error,
       json_extract(value, '$.would_be_ci[0]')      AS lo,
       json_extract(value, '$.would_be_ci[1]')      AS hi,
       json_extract(value, '$.would_cover_truth')   AS covers,
       json_extract(value, '$.decision_correct')    AS correct
FROM src, json_each(src.t, '$.rows')
WHERE src.which = 'harness';

-- Three of the fifteen grid entries are the same scenario reached from three
-- directions (the forward run, zero items dropped, and the full sample), with
-- every field identical. DISTINCT collapses them, so the join below cannot fan
-- out; the uniqueness of the remaining effective sample sizes is asserted.
CREATE TEMP VIEW distinct_grid AS
SELECT DISTINCT ess, unlogged, truth, rel_error, lo, hi FROM grid;

-- The gate, rebuilt from the grid's three scalars.
CREATE TEMP VIEW rebuilt AS
SELECT ess, unlogged, truth, rel_error, lo, hi,
       CASE WHEN unlogged > 0.01 OR ess < 1000.0 THEN 'refuse' ELSE 'ok' END AS status,
       (truth >= lo AND truth <= hi) AS covers
FROM distinct_grid;

CREATE TEMP VIEW paired AS
SELECT p.name AS scenario,
       r.status AS sql_status, p.status AS py_status,
       r.covers AS sql_covers, p.covers AS py_covers,
       (r.status = 'ok') = r.covers AS sql_correct, p.correct AS py_correct,
       abs(r.unlogged - p.unlogged)  AS d_unlogged,
       abs(r.rel_error - p.rel_error) AS d_rel_error,
       max(abs(r.lo - p.lo), abs(r.hi - p.hi)) AS d_ci
FROM published p JOIN rebuilt r ON r.ess = p.ess;

SELECT 'scenario                                 gate     covers  correct   worst delta';
SELECT printf('%-40s %-8s %-7s %-8s %.1e',
              substr(scenario, 1, 40), sql_status,
              CASE sql_covers WHEN 1 THEN 'yes' ELSE 'no' END,
              CASE WHEN sql_correct = py_correct AND sql_status = py_status
                        AND sql_covers = py_covers THEN 'agrees' ELSE 'FAIL' END,
              max(d_unlogged, d_rel_error, d_ci))
FROM paired ORDER BY scenario;

SELECT '';
SELECT printf('%d of %d scenarios paired by effective sample size, from %d '
              || 'distinct grid entries', count(*),
              (SELECT count(*) FROM published),
              (SELECT count(*) FROM distinct_grid)) FROM paired;
SELECT printf('SQL scores %d correct decisions, harness_all.json publishes %d',
              (SELECT sum(sql_correct) FROM paired),
              (SELECT json_extract(t, '$.decisions_correct') FROM src WHERE which = 'harness'));

-- One number: how many checks failed. verify/verify.sh requires "RESULT 0 7".
SELECT printf('RESULT %d %d',
    (SELECT count(*) FROM paired
        WHERE sql_status <> py_status OR sql_covers <> py_covers
           OR sql_correct <> py_correct
           OR d_unlogged > 0.0 OR d_rel_error > 1e-15 OR d_ci > 0.0)
  + (SELECT CASE WHEN count(*) = (SELECT count(*) FROM published) THEN 0 ELSE 1 END
        FROM paired)
  + (SELECT count(*) FROM (SELECT ess FROM distinct_grid GROUP BY ess HAVING count(*) > 1))
  + (SELECT CASE WHEN (SELECT sum(sql_correct) FROM paired)
                    = json_extract(t, '$.decisions_correct')
                  AND (SELECT count(*) FROM published)
                    = json_extract(t, '$.n_scenarios')
             THEN 0 ELSE 1 END FROM src WHERE which = 'harness'),
    (SELECT count(*) FROM paired));
