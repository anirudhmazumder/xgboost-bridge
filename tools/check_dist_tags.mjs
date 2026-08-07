#!/usr/bin/env node
// Decide whether npm's dist-tags are where a publish should have left them.
//
// This lived as inline shell inside `release.yml` and shipped a defect that fired
// on the 1.0.0 release: the retry loop hardcoded `rc`, because every publish
// before that one was a release candidate. A final release uses `latest`, so `rc`
// stayed at `1.0.0-rc.2`, the loop waited six times for something that could not
// happen, and reported "the rc tag never came to point at 1.0.0" about a release
// that had already succeeded — a green publish inside a red run.
//
// The shape of that mistake is the point. The *tag derivation* had a test and was
// caught before any dispatch; the *check on the tag* had none, because one was in
// a file a test could import and the other was inline YAML. So this is a module
// with a CLI attached rather than a shell snippet: anything in a workflow that
// makes a decision belongs where a test can reach it.
//
// Exit status is the CLI contract. `evaluateDistTags` is the testable core and
// throws nothing — it returns a verdict, so a test can assert on the reason
// rather than on a message string.

// Two ecosystems, two spellings, and until this file existed there were THREE
// definitions of "prerelease" in the release pipeline: this one for semver, a
// loose `/(a|b|rc|alpha|beta|dev)/i` in the targets guard for PEP 440, and an
// inline copy in the dist-tag check. They cannot be a single regex -- npm's
// `1.0.0-rc.1` and PyPI's `1.0.0rc1` are genuinely different grammars -- but they
// can live in one tested place, which is the difference that matters.

/** npm / semver: `1.0.0-rc.1`, `2.0.0-alpha.3`. */
export function isPrerelease(version) {
  return /-(?:rc|alpha|beta)\.\d+$/.test(version ?? "");
}

/**
 * PyPI / PEP 440: `1.0.0rc1`, `1.0.0a1`, `1.0.0b2`, `1.0.0.dev0`.
 *
 * Anchored to the pattern rather than searching for letters. The loose predicate
 * this replaces matched any version *containing* `a` or `b`, which happens to be
 * safe for digits-and-dots releases and is the wrong shape: a mis-classification
 * here lets a final release publish to one registry only, because the guard would
 * treat it as a prerelease and skip the both-registries requirement.
 */
export function isPrereleasePep440(version) {
  return /(?:a|b|rc)\d+$|\.dev\d+$|\.post\d+$/.test(version ?? "");
}

/**
 * Evaluate a dist-tag state against what a publish should have produced.
 *
 * @param {object} options
 * @param {Record<string, string>} options.tags - the registry's dist-tags
 * @param {string} options.version - the version just published
 * @param {string} options.expectedTag - the tag `npm publish --tag` was given
 * @returns {{ok: boolean, settled: boolean, reason: string, notes: string[]}}
 *   `settled` is false when the registry has not caught up yet, which is a retry
 *   rather than a verdict — distinguishing "wrong" from "not yet" is why this
 *   returns a shape instead of a boolean.
 */
export function evaluateDistTags({ tags, version, expectedTag }) {
  const notes = [];

  if (!expectedTag) {
    return { ok: false, settled: true, reason: "no expected tag was supplied", notes };
  }
  if (!version) {
    return { ok: false, settled: true, reason: "no version was supplied", notes };
  }

  // The tag the publish actually used. Waiting on any *other* tag is the defect
  // this module exists to prevent: it cannot be satisfied and it is indefinite.
  if (tags[expectedTag] !== version) {
    return {
      ok: false,
      settled: false,
      reason:
        `the ${expectedTag} tag is ${tags[expectedTag] ?? "<absent>"}, expected ${version}`,
      notes,
    };
  }

  if (isPrerelease(version)) {
    // A prerelease must not *take* `latest`, but on a package whose only versions
    // are prereleases npm has already put one there and refuses to remove it
    // (E400: `latest` is protected and every package must have one). So this is
    // reported, not failed — the state is not correctable until a final release.
    if (tags.latest === version) {
      notes.push(
        `latest also points at the prerelease ${version}; that is npm's ` +
          `first-publish behaviour and cannot be undone until a non-prerelease ` +
          `version exists`,
      );
    } else if (isPrerelease(tags.latest)) {
      notes.push(`latest points at another prerelease, ${tags.latest}`);
    } else {
      notes.push(`latest points at ${tags.latest}, a final release`);
    }
    return { ok: true, settled: true, reason: `${expectedTag} -> ${version}`, notes };
  }

  // A final release must own `latest`. This is the branch that matters to a
  // consumer, because `npm install <pkg>` resolves `latest` — an RC left sitting
  // there is a silent wrong-version-installed path.
  if (tags.latest !== version) {
    return {
      ok: false,
      settled: true,
      reason:
        `latest points at ${tags.latest ?? "<absent>"}, not the just-published ` +
        `${version}; \`npm install\` would resolve the wrong version`,
      notes,
    };
  }
  if (isPrerelease(tags.latest)) {
    return {
      ok: false,
      settled: true,
      reason: `latest is the prerelease ${tags.latest} after a final release`,
      notes,
    };
  }

  return {
    ok: true,
    settled: true,
    reason: `latest -> ${tags.latest}, ${expectedTag} -> ${tags[expectedTag]}`,
    notes,
  };
}

/**
 * The old, defective rule, kept so a test can demonstrate it failing.
 *
 * Retained deliberately rather than deleted: a regression test that reimplements
 * the bug inside itself proves only that the test author remembered it. This is
 * the code that shipped, and the test below asserts it disagrees with the current
 * rule on exactly the input that broke the 1.0.0 release.
 */
export function evaluateDistTagsHardcodedRc({ tags, version }) {
  if (tags.rc !== version) {
    return {
      ok: false,
      settled: false,
      reason: `the rc tag never came to point at ${version}`,
      notes: [],
    };
  }
  return { ok: true, settled: true, reason: `rc -> ${version}`, notes: [] };
}

// --- CLI ---------------------------------------------------------------------
// Invoked by release.yml as:
//   node tools/check_dist_tags.mjs <version> <expectedTag> [distTagsJson]
// With no JSON argument it queries the registry itself and retries, because
// dist-tag visibility is eventually consistent and a race is not a verdict.

async function readTagsFromRegistry(packageName) {
  const { execFileSync } = await import("node:child_process");
  try {
    return JSON.parse(
      execFileSync("npm", ["view", packageName, "dist-tags", "--json"], {
        encoding: "utf8",
        stdio: ["ignore", "pipe", "ignore"],
      }),
    );
  } catch {
    return {};
  }
}

async function main(argv) {
  const [version, expectedTag, inlineJson] = argv;
  const packageName = process.env.PACKAGE_NAME ?? "xgboost-predictor";

  if (inlineJson !== undefined) {
    const verdict = evaluateDistTags({
      tags: JSON.parse(inlineJson),
      version,
      expectedTag,
    });
    verdict.notes.forEach((note) => console.log(`NOTE: ${note}`));
    console.log(verdict.ok ? `OK: ${verdict.reason}` : `FAIL: ${verdict.reason}`);
    return verdict.ok ? 0 : 1;
  }

  const attempts = Number(process.env.ATTEMPTS ?? 6);
  const waitMs = Number(process.env.WAIT_MS ?? 10_000);
  let verdict;
  for (let attempt = 1; attempt <= attempts; attempt += 1) {
    const tags = await readTagsFromRegistry(packageName);
    verdict = evaluateDistTags({ tags, version, expectedTag });
    console.log(`attempt ${attempt}: ${JSON.stringify(tags)}`);
    if (verdict.ok || verdict.settled) break;
    if (attempt < attempts) {
      console.log(`not visible yet; waiting ${waitMs / 1000}s`);
      await new Promise((resolve) => setTimeout(resolve, waitMs));
    }
  }
  verdict.notes.forEach((note) => console.log(`NOTE: ${note}`));
  console.log(verdict.ok ? `OK: ${verdict.reason}` : `FAIL: ${verdict.reason}`);
  return verdict.ok ? 0 : 1;
}

// `pathToFileURL`, not string concatenation. `import.meta.url` percent-encodes a
// path -- this repository lives under directories with spaces -- while
// `process.argv[1]` carries them literally, so the naive comparison never matched
// and the CLI silently did nothing. It would have worked in CI, whose paths have no
// spaces, which is the sort of accident that makes a script look fine until it is
// run somewhere real.
const { pathToFileURL } = await import("node:url");
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main(process.argv.slice(2)).then((code) => {
    process.exitCode = code;
  });
}
