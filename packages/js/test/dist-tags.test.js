// The dist-tag check, unit-tested offline against recorded registry responses.
//
// This logic was inline shell in release.yml and shipped a defect that fired on
// the 1.0.0 release. The tag *derivation* had a test and was caught before any
// dispatch; the *check on the tag* had none, because one lived in a file a test
// could import and the other was inline YAML. Extracting it is the fix for the
// shape, not just for the bug.
//
// Every case below uses a recorded dist-tag response, so none of this touches a
// registry. The recordings are real: they are what `npm view xgboost-predictor
// dist-tags --json` actually returned at each stage of this project's releases.

import test from "node:test";
import assert from "node:assert/strict";

import {
  evaluateDistTags,
  evaluateDistTagsHardcodedRc,
  isPrerelease,
  isPrereleasePep440,
} from "../../../tools/check_dist_tags.mjs";

// --- Recorded states, in the order they actually occurred ---------------------

/** After the manual first publish. npm assigned `latest` itself (D061). */
const AFTER_FIRST_MANUAL_PUBLISH = { rc: "1.0.0-rc.1", latest: "1.0.0-rc.1" };

/** After the rc.2 rehearsal through release.yml: rc moved, latest untouched. */
const AFTER_RC2_REHEARSAL = { rc: "1.0.0-rc.2", latest: "1.0.0-rc.1" };

/** After the 1.0.0 release: latest moved off the prereleases. */
const AFTER_FINAL_RELEASE = { rc: "1.0.0-rc.2", latest: "1.0.0" };

// --- The prerelease branch ----------------------------------------------------

test("a prerelease publish is satisfied when rc points at it, whatever latest holds", () => {
  const verdict = evaluateDistTags({
    tags: AFTER_RC2_REHEARSAL,
    version: "1.0.0-rc.2",
    expectedTag: "rc",
  });
  assert.equal(verdict.ok, true, verdict.reason);
  assert.equal(verdict.settled, true);
  // latest is still on rc.1, and that is reported rather than failed.
  assert.match(verdict.notes.join(" "), /another prerelease, 1\.0\.0-rc\.1/);
});

test("a prerelease that npm also put on latest is reported, not failed", () => {
  // The first-publish state. Failing here would make the bootstrap unpublishable,
  // since npm refuses to remove `latest` (E400).
  const verdict = evaluateDistTags({
    tags: AFTER_FIRST_MANUAL_PUBLISH,
    version: "1.0.0-rc.1",
    expectedTag: "rc",
  });
  assert.equal(verdict.ok, true, verdict.reason);
  assert.match(verdict.notes.join(" "), /cannot be undone until a non-prerelease/);
});

test("a prerelease publish fails if rc did not move", () => {
  const verdict = evaluateDistTags({
    tags: AFTER_RC2_REHEARSAL,
    version: "1.0.0-rc.3",
    expectedTag: "rc",
  });
  assert.equal(verdict.ok, false);
  // Unsettled: the registry may simply not have caught up, so the CLI retries.
  assert.equal(verdict.settled, false);
});

// --- The final-release branch, which is the one that had never executed -------

test("a final release is satisfied when latest points at it", () => {
  const verdict = evaluateDistTags({
    tags: AFTER_FINAL_RELEASE,
    version: "1.0.0",
    expectedTag: "latest",
  });
  assert.equal(verdict.ok, true, verdict.reason);
  assert.equal(verdict.settled, true);
  assert.match(verdict.reason, /latest -> 1\.0\.0/);
});

test("a final release published under a NON-DEFAULT tag fails, and says why", () => {
  // This is the reachable form of the final-release branch, and it is exactly the
  // D062 publish-step defect: `--tag rc` hardcoded, so 1.0.0 lands on `rc` and
  // `latest` is left on the last release candidate. The expected tag matches — the
  // publish did what it was told — and the release is still wrong.
  //
  // Writing this test is what established that. My first version passed
  // `expectedTag: "latest"` and asserted this message, which the code cannot
  // produce: when the expected tag itself has not moved, that is indistinguishable
  // from registry lag and is retried instead.
  const verdict = evaluateDistTags({
    tags: { rc: "1.0.0", latest: "1.0.0-rc.1" },
    version: "1.0.0",
    expectedTag: "rc",
  });
  assert.equal(verdict.ok, false);
  assert.equal(verdict.settled, true, "this is a verdict, not a race — do not retry it");
  assert.match(verdict.reason, /npm install` would resolve the wrong version/);
});

test("a final release whose own tag has not appeared yet is a RETRY, not a verdict", () => {
  // Distinguishing "wrong" from "not yet" is why the verdict carries `settled`.
  // Registry visibility is eventually consistent, so a missing tag must not fail
  // the release on the first look.
  const verdict = evaluateDistTags({
    tags: { latest: "0.9.0" },
    version: "1.0.0",
    expectedTag: "latest",
  });
  assert.equal(verdict.ok, false);
  assert.equal(verdict.settled, false, "an unmoved expected tag may just be lag");
  assert.match(verdict.reason, /the latest tag is 0\.9\.0, expected 1\.0\.0/);
});

test("a final release under a non-default tag is caught even when latest is unrelated", () => {
  const verdict = evaluateDistTags({
    tags: { rc: "1.0.0", latest: "0.9.0" },
    version: "1.0.0",
    expectedTag: "rc",
  });
  assert.equal(verdict.ok, false);
  assert.equal(verdict.settled, true);
  assert.match(verdict.reason, /latest points at 0\.9\.0, not the just-published 1\.0\.0/);
});

// --- The regression: the old rule, run against the input that broke it --------

test("the old hardcoded-rc rule fails on the 1.0.0 release; the current rule passes", () => {
  // This is the whole point of the extraction. Both rules are handed the state
  // that actually existed at the moment the 1.0.0 run failed.
  const input = { tags: AFTER_FINAL_RELEASE, version: "1.0.0", expectedTag: "latest" };

  const old = evaluateDistTagsHardcodedRc(input);
  assert.equal(old.ok, false, "the old rule must fail, or this test proves nothing");
  assert.match(old.reason, /the rc tag never came to point at 1\.0\.0/);
  // And it fails as *unsettled*, which is why the workflow retried six times and
  // then reported a timeout about a release that had already succeeded.
  assert.equal(old.settled, false);

  const current = evaluateDistTags(input);
  assert.equal(current.ok, true, `the current rule must pass: ${current.reason}`);

  assert.notEqual(
    old.ok,
    current.ok,
    "the two rules must disagree on this input, or the fix changed nothing",
  );
});

test("the old rule and the current rule agree on every prerelease case", () => {
  // The defect was invisible for as long as it was, because the two rules are
  // indistinguishable while every publish is a release candidate. Asserting that
  // explains why no earlier run caught it.
  for (const [tags, version] of [
    [AFTER_FIRST_MANUAL_PUBLISH, "1.0.0-rc.1"],
    [AFTER_RC2_REHEARSAL, "1.0.0-rc.2"],
  ]) {
    const old = evaluateDistTagsHardcodedRc({ tags, version });
    const current = evaluateDistTags({ tags, version, expectedTag: "rc" });
    assert.equal(old.ok, current.ok, `the rules should agree for ${version}`);
    assert.equal(old.ok, true);
  }
});

// --- Guards on the inputs themselves -----------------------------------------

test("an absent expected tag is a failure, not a pass", () => {
  const verdict = evaluateDistTags({ tags: AFTER_FINAL_RELEASE, version: "1.0.0" });
  assert.equal(verdict.ok, false);
  assert.match(verdict.reason, /no expected tag/);
});

test("prerelease classification matches the workflow's own definition", () => {
  for (const version of ["1.0.0-rc.1", "1.0.0-rc.2", "2.0.0-alpha.1", "1.5.0-beta.3"]) {
    assert.equal(isPrerelease(version), true, version);
  }
  for (const version of ["1.0.0", "1.0.1", "2.0.0", "10.20.30"]) {
    assert.equal(isPrerelease(version), false, version);
  }
});

// --- Both prerelease grammars, because the pipeline spans two ecosystems -------

test("the PEP 440 predicate classifies PyPI versions, including the ones a loose regex got wrong", () => {
  for (const version of ["1.0.0rc1", "1.0.0rc2", "1.0.0a1", "1.0.0b2", "1.0.0.dev0", "0.1.0.dev0"]) {
    assert.equal(isPrereleasePep440(version), true, version);
  }
  for (const version of ["1.0.0", "1.0.1", "2.0.0", "10.20.30"]) {
    assert.equal(isPrereleasePep440(version), false, version);
  }
});

test("the two grammars disagree, which is why there are two of them", () => {
  // A single regex cannot serve both: npm writes `1.0.0-rc.1` and PyPI writes
  // `1.0.0rc1`. Asserting the disagreement stops someone collapsing them.
  assert.equal(isPrerelease("1.0.0-rc.1"), true, "semver form");
  assert.equal(isPrerelease("1.0.0rc1"), false, "PEP 440 form is not semver");
});

test("no version is classified as final by one grammar and prerelease by the other, for the versions this project has used", () => {
  // The real pairs this project shipped. A disagreement here would mean the guard
  // and the tag derivation could take opposite branches on one release.
  for (const [pep440, semver] of [
    ["1.0.0rc1", "1.0.0-rc.1"],
    ["1.0.0rc2", "1.0.0-rc.2"],
    ["1.0.0", "1.0.0"],
  ]) {
    assert.equal(
      isPrereleasePep440(pep440),
      isPrerelease(semver),
      `${pep440} and ${semver} must agree about being a prerelease`,
    );
  }
});
