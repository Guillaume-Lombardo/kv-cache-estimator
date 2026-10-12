import { afterEach, describe, expect, it, vi } from "vitest";

afterEach(() => {
  vi.unstubAllEnvs();
  vi.resetModules();
});

describe("deployment base path", () => {
  it("serves at the root when no prefix is configured", async () => {
    vi.stubEnv("KV_ESTIMATOR_BASE_PATH", undefined);
    const { default: config } = await import("../../next.config");
    expect(config.basePath).toBe("");
  });

  it.each(["", "/kv-estimator", "/tools/kv-estimator", "/tools_v2/kv-estimator.v1"])(
    "accepts the deployment prefix %j",
    async (value) => {
      vi.stubEnv("KV_ESTIMATOR_BASE_PATH", value);
      const { default: config } = await import("../../next.config");
      expect(config.basePath).toBe(value);
      expect(config.output).toBe("standalone");
    },
  );

  it.each([
    "/",
    "kv-estimator",
    "/kv-estimator/",
    "//kv-estimator",
    "/tools//kv-estimator",
    "/tools/./kv-estimator",
    "/tools/../kv-estimator",
    "/tools/%2e%2e/kv-estimator",
    "/kv-estimator?theme=dark",
    "/kv-estimator#results",
    "/kv estimator",
    "/kv-estimator\\other",
    " /kv-estimator",
  ])("rejects the invalid deployment prefix %j", async (value) => {
    vi.stubEnv("KV_ESTIMATOR_BASE_PATH", value);
    await expect(import("../../next.config")).rejects.toThrow("Invalid KV_ESTIMATOR_BASE_PATH");
  });
});
