import { readFileSync } from "node:fs";
import type { PlanStatus } from "./plan.ts";

export interface AttestationCheck {
	enabled: boolean;
	tampered: boolean;
	expected?: string;
	actual?: string;
	attestationPath?: string;
}

function normalizeHash(value: string): string | undefined {
	const hash = value.trim().toLowerCase();
	if (!/^[a-f0-9]{64}$/.test(hash)) return undefined;
	return hash;
}

function readExpectedHash(path: string): string | undefined {
	try {
		return normalizeHash(readFileSync(path, "utf-8"));
	} catch {
		return undefined;
	}
}

export function checkPlanAttestation(status: PlanStatus): AttestationCheck {
	if (!status.exists || !status.planPath) {
		return { enabled: false, tampered: false };
	}

	const attestationPath = status.attestationPath;
	if (!attestationPath) {
		return { enabled: false, tampered: false };
	}

	const expected = readExpectedHash(attestationPath);
	if (!expected) {
		return { enabled: true, tampered: true, attestationPath };
	}

	// Reopening the plan here could verify different bytes from the text and
	// phase state captured by readPlanStatus. Validate that snapshot only.
	const actual = status.planSha256;
	if (!actual) {
		return { enabled: true, tampered: true, expected, attestationPath };
	}

	return {
		enabled: true,
		tampered: actual !== expected,
		expected,
		actual,
		attestationPath,
	};
}
