/**
 * @name Agent generated test non-triviality
 * @id js/agentic-stryker/agent-nontriviality
 * @description Classify generated agent tests whose assertions depend on the
 *              package under test. Reports both the original TestPilot-style
 *              package-name import definition and an agent-compatible definition
 *              that also accepts relative imports from archived generated tests.
 * @kind table
 */

import javascript

/**
 * A minimized or archived agent report.json file.
 */
class AgentReportJson extends JsonObject {
  AgentReportJson() {
    this.isTopLevel() and
    this.getFile().getBaseName() = "report.json"
  }

  /** Gets the `tests/` folder next to this file. */
  Folder getTestFolder() { result = this.getFile().getParentContainer().getFolder("tests") }

  /** Gets the package name for this benchmark run. */
  string getPackageName() {
    result = this.getPropStringValue("packageName")
    or
    result = this.getPropValue("metaData").(JsonObject).getPropStringValue("packageName")
  }
}

/** A generated test stored next to an agent report. */
class AgentGeneratedTest extends File {
  AgentReportJson report;

  AgentGeneratedTest() { this.getParentContainer() = report.getTestFolder() }

  /** Gets the report to which this test belongs. */
  AgentReportJson getReport() { result = report }

  /** Gets the name of the package for which this test was generated. */
  string getPackageName() { result = report.getPackageName() }
}

/**
 * An assertion call in a generated agent test.
 */
class AssertionInAgentGeneratedTest extends DataFlow::Node {
  AgentGeneratedTest test;

  AssertionInAgentGeneratedTest() {
    (
      this = API::moduleImport("assert").getASuccessor*().getACall()
      or
      this = API::moduleImport("assert/strict").getASuccessor*().getACall()
      or
      this = API::moduleImport("node:assert").getASuccessor*().getACall()
      or
      this = API::moduleImport("node:assert/strict").getASuccessor*().getACall()
    ) and
    test = this.getFile()
  }

  AgentGeneratedTest getTest() { result = test }

  /**
   * Gets a node in the intra-procedural backwards slice of this assertion.
   * This mirrors the TestPilot artifact's AssertionQuality predicate.
   */
  DataFlow::Node getANodeInBackwardsSlice() {
    result = this
    or
    DataFlow::localFlowStep(result, this.getANodeInBackwardsSlice())
    or
    TaintTracking::sharedTaintStep(result, this.getANodeInBackwardsSlice())
    or
    result.asExpr().getParent+() = this.getANodeInBackwardsSlice().asExpr()
    or
    exists(DataFlow::InvokeNode call |
      call.getABoundCallbackParameter(_, _) = this.getANodeInBackwardsSlice()
      or
      exists(Function cb | cb = call.getAnArgument().getAFunctionValue().getFunction() |
        cb = this.getANodeInBackwardsSlice().getContainer()
      )
    |
      result = call.getAnArgument() or
      result = call.getCalleeNode()
    )
    or
    exists(DataFlow::InvokeNode call, DataFlow::SsaDefinitionNode v |
      call.getAnArgument().getAPredecessor() = v and
      v = this.getANodeInBackwardsSlice() and
      result = call.getCalleeNode()
    )
  }

  /**
   * Gets evidence for the strict TestPilot-style non-triviality definition:
   * the assertion depends on an import whose imported path is exactly the
   * package name from report.json.
   */
  string getStrictImportPath() {
    exists(Require req |
      req = this.getANodeInBackwardsSlice().asExpr() and
      result = req.getImportedPath().getValue() and
      result = test.getPackageName()
    )
  }

  /**
   * Gets evidence for the agent-compatible definition. Agent tests normally
   * import the package under test through archived relative paths such as
   * ../../lib/index.js rather than through the npm package name.
   */
  string getAgentCompatibleImportPath() {
    result = this.getStrictImportPath()
    or
    exists(Require req |
      req = this.getANodeInBackwardsSlice().asExpr() and
      result = req.getImportedPath().getValue() and
      (result.matches("./%") or result.matches("../%"))
    )
  }
}

query predicate agentNonTriviality(
  string reportPath,
  string testPath,
  string packageName,
  string testFile,
  string definition,
  string importPath
) {
  exists(
    AgentReportJson report, AgentGeneratedTest test, AssertionInAgentGeneratedTest assertion
  |
    test.getReport() = report and
    assertion.getTest() = test and
    reportPath = report.getFile().getRelativePath() and
    testPath = test.getRelativePath() and
    packageName = report.getPackageName() and
    testFile = test.getBaseName() and
    (
      definition = "strict" and
      importPath = assertion.getStrictImportPath()
      or
      definition = "agent_compatible" and
      importPath = assertion.getAgentCompatibleImportPath()
    )
  )
}
