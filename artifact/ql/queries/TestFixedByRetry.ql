/**
 * @name Test fixed after retrying
 * @id js/testpilot/test-fixed-by-retry
 * @description Find failing tests that pass after having been refined
 *              with the `RetryWithError` refiner.
 * @kind table
 */

import AssertionQuality

predicate testFixedByRetry(
  ReportJson report, Prompt orig, GeneratedTest failing, Prompt refined, GeneratedTest passing
) {
  orig = report.getAPrompt() and
  failing = orig.getATest(false, _) and
  refined.isRefinedFrom(orig, failing, "RetryWithError") and
  passing = refined.getATest(true, _)
}

query predicate stats(
  ReportJson report, ErrorCategory errorCategory, int failed, int fixed
) {
  failed = count(GeneratedTest t | t = report.getATest() and t.failsDueTo(errorCategory)) and
  fixed =
    count(GeneratedTest t | testFixedByRetry(report, _, t, _, _) and t.failsDueTo(errorCategory))
}
