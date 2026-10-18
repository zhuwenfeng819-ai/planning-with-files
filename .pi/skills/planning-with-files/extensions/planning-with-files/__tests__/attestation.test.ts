import { createHash } from "node:crypto";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";
import { checkPlanAttestation } from "../attestation.ts";
import { readPlanStatus } from "../plan.ts";

const tempRoots: string[] = [];

function makeWorkspace(): string {
	const cwd = mkdtempSync(join(tmpdir(), "pwf-pi-attestation-"));
	tempRoots.push(cwd);
	return cwd;
}

function sha256(content: string | Buffer): string {
	return createHash("sha256").update(content).digest("hex");
}

function writePlan(cwd: string, content: string): void {
	const planDir = join(cwd, ".planning", "demo");
	mkdirSync(planDir, { recursive: true });
	writeFileSync(join(planDir, "task_plan.md"), content);
}

afterEach(() => {
	while (tempRoots.length > 0) {
		const root = tempRoots.pop();
		if (root) rmSync(root, { recursive: true, force: true });
	}
});

describe("Pi extension plan attestation", () => {
	for (const scope of ["root", "scoped"] as const) {
		it(`verifies the captured ${scope} plan, not a replacement read from disk`, () => {
			const cwd = makeWorkspace();
			const planDir = scope === "root" ? cwd : join(cwd, ".planning", "demo");
			mkdirSync(planDir, { recursive: true });
			const planPath = join(planDir, "task_plan.md");
			const approved = "# Approved\n### Phase 1\n**Status:** in_progress\n";
			const unapproved = "# Unapproved snapshot\n### Phase 1\n**Status:** complete\n";
			writeFileSync(planPath, unapproved);
			writeFileSync(join(planDir, scope === "root" ? ".plan-attestation" : ".attestation"), sha256(approved));

			const status = readPlanStatus(cwd);
			expect(status.scope).toBe(scope);
			expect(status.firstLines50).toContain("# Unapproved snapshot");
			expect(status.completePhases).toBe(1);
			// The real verifier runs after the reader. Restore approved bytes in
			// between, as a concurrent editor can, without changing the snapshot.
			writeFileSync(planPath, approved);

			expect(checkPlanAttestation(status)).toMatchObject({
				enabled: true,
				tampered: true,
				expected: sha256(approved),
				actual: sha256(unapproved),
			});
		});
	}

	it("hashes the complete snapshot even when both injected prefixes match", () => {
		const cwd = makeWorkspace();
		const prefix = Array.from({ length: 50 }, (_, index) => `line ${index}`).join("\n") + "\n";
		const approved = prefix + "### Phase 1\n**Status:** in_progress\n";
		const unapproved = prefix + "### Phase 1\n**Status:** complete\n";
		writePlan(cwd, unapproved);
		writeFileSync(join(cwd, ".planning", "demo", ".attestation"), sha256(approved));
		const status = readPlanStatus(cwd);
		writePlan(cwd, approved);

		expect(status.firstLines50).toBe(readPlanStatus(cwd).firstLines50);
		expect(status.headLines30).toBe(readPlanStatus(cwd).headLines30);
		expect(status.completePhases).toBe(1);
		expect(checkPlanAttestation(status)).toMatchObject({
			enabled: true, tampered: true, actual: sha256(unapproved),
		});
	});

	it("keeps a successfully captured snapshot valid after the live plan changes", () => {
		const cwd = makeWorkspace();
		const approved = "### Phase 1\n**Status:** in_progress\n";
		writePlan(cwd, approved);
		writeFileSync(join(cwd, ".planning", "demo", ".attestation"), sha256(approved));
		const status = readPlanStatus(cwd);
		writePlan(cwd, "# Later edit\n");

		expect(checkPlanAttestation(status)).toMatchObject({
			enabled: true, tampered: false, actual: sha256(approved),
		});
		expect(status.firstLines50).toBe(approved);
		expect(status.inProgressPhases).toBe(1);
		// The next event gets a fresh snapshot and rejects the later edit.
		expect(checkPlanAttestation(readPlanStatus(cwd)).tampered).toBe(true);
	});

	it("does not turn a failed snapshot read into approval after the file is repaired", () => {
		const cwd = makeWorkspace();
		const planPath = join(cwd, "task_plan.md");
		const approved = "### Phase 1\n**Status:** complete\n";
		// A directory is a deterministic read error on every supported CI OS.
		mkdirSync(planPath);
		writeFileSync(join(cwd, ".plan-attestation"), sha256(approved));
		const status = readPlanStatus(cwd);
		rmSync(planPath, { recursive: true });
		writeFileSync(planPath, approved);

		expect(status.firstLines50).toBe("");
		expect(status.totalPhases).toBe(0);
		expect(checkPlanAttestation(status)).toMatchObject({
			enabled: true, tampered: true, expected: sha256(approved),
		});
		expect(checkPlanAttestation(status).actual).toBeUndefined();
	});

	for (const scope of ["root", "scoped"] as const) {
		it(`keeps ${scope} attestation enabled if its selected file disappears`, () => {
			const cwd = makeWorkspace();
			const planDir = scope === "root" ? cwd : join(cwd, ".planning", "demo");
			mkdirSync(planDir, { recursive: true });
			const plan = "# Plan\n";
			const attestationPath = join(planDir, scope === "root" ? ".plan-attestation" : ".attestation");
			writeFileSync(join(planDir, "task_plan.md"), plan);
			writeFileSync(attestationPath, sha256(plan));
			const status = readPlanStatus(cwd);
			rmSync(attestationPath);
			if (scope === "scoped") {
				// Losing a selected scoped attestation must not switch to root.
				writeFileSync(join(cwd, ".plan-attestation"), sha256(plan));
			}

			expect(checkPlanAttestation(status)).toMatchObject({
				enabled: true, tampered: true, attestationPath,
			});
		});
	}

	it("treats an unreadable attestation as a mismatch without throwing", () => {
		const cwd = makeWorkspace();
		writePlan(cwd, "# Plan\n");
		const attestationPath = join(cwd, ".planning", "demo", ".attestation");
		mkdirSync(attestationPath);

		expect(checkPlanAttestation(readPlanStatus(cwd))).toMatchObject({
			enabled: true, tampered: true, attestationPath,
		});
	});

	it("keeps raw BOM and invalid UTF-8 bytes in the digest", () => {
		const cwd = makeWorkspace();
		const bytes = Buffer.concat([
			Buffer.from([0xef, 0xbb, 0xbf]),
			Buffer.from("# Plan\n### Phase 1\n**Status:** in_progress\n"),
			Buffer.from([0xff, 0xfe]),
		]);
		writePlan(cwd, "");
		writeFileSync(join(cwd, ".planning", "demo", "task_plan.md"), bytes);
		const attestationPath = join(cwd, ".planning", "demo", ".attestation");
		writeFileSync(attestationPath, sha256(bytes));
		const status = readPlanStatus(cwd);

		expect(status.firstLines50).toBe(bytes.toString("utf-8"));
		expect(status.inProgressPhases).toBe(1);
		expect(checkPlanAttestation(status)).toMatchObject({
			enabled: true, tampered: false, actual: sha256(bytes),
		});
		writeFileSync(attestationPath, sha256(bytes.toString("utf-8")));
		expect(checkPlanAttestation(status).tampered).toBe(true);
	});

	it("uses a root attestation for a scoped plan when no scoped attestation exists", () => {
		const cwd = makeWorkspace();
		const plan = "# Scoped plan\n";
		writePlan(cwd, plan);
		const attestationPath = join(cwd, ".plan-attestation");
		writeFileSync(attestationPath, sha256(plan));

		expect(checkPlanAttestation(readPlanStatus(cwd))).toMatchObject({
			enabled: true, tampered: false, attestationPath,
		});
	});

	it("preserves opt-out behavior when no attestation was selected", () => {
		const cwd = makeWorkspace();
		writePlan(cwd, "# Plan\n");

		expect(checkPlanAttestation(readPlanStatus(cwd))).toEqual({ enabled: false, tampered: false });
	});

	it("accepts a known-good SHA-256 attestation", () => {
		const cwd = makeWorkspace();
		const plan = "### Phase 1\n**Status:** complete\n";
		writePlan(cwd, plan);
		writeFileSync(join(cwd, ".planning", "demo", ".attestation"), sha256(plan));

		const result = checkPlanAttestation(readPlanStatus(cwd));

		expect(result).toMatchObject({
			enabled: true,
			tampered: false,
			expected: sha256(plan),
			actual: sha256(plan),
		});
	});

	it("rejects mutated plan content when the attestation hash no longer matches", () => {
		const cwd = makeWorkspace();
		const originalPlan = "### Phase 1\n**Status:** complete\n";
		const mutatedPlan = "### Phase 1\n**Status:** in_progress\n";
		writePlan(cwd, originalPlan);
		writeFileSync(join(cwd, ".planning", "demo", ".attestation"), sha256(originalPlan));
		writeFileSync(join(cwd, ".planning", "demo", "task_plan.md"), mutatedPlan);

		const result = checkPlanAttestation(readPlanStatus(cwd));

		expect(result.enabled).toBe(true);
		expect(result.tampered).toBe(true);
		expect(result.expected).toBe(sha256(originalPlan));
		expect(result.actual).toBe(sha256(mutatedPlan));
	});

	it("treats an invalid attestation file as a blocking mismatch", () => {
		const cwd = makeWorkspace();
		writePlan(cwd, "### Phase 1\n**Status:** complete\n");
		writeFileSync(join(cwd, ".planning", "demo", ".attestation"), "not-a-sha256");

		const result = checkPlanAttestation(readPlanStatus(cwd));

		expect(result.enabled).toBe(true);
		expect(result.tampered).toBe(true);
		expect(result.expected).toBeUndefined();
		expect(result.actual).toBeUndefined();
	});
});
