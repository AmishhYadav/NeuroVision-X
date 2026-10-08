import { describe, expect, it } from "vitest";
import { splitLabel } from "./splitLabel";

describe("splitLabel", () => {
  it("labels the default eval_test directory as test, not val ('eval' contains 'val')", () => {
    expect(splitLabel("outputs/eval_test_baseline_unet3d")).toBe("test split");
  });
  it("labels a val directory as val", () => {
    expect(splitLabel("outputs/neurovision/eval_val")).toBe("val split");
  });
  it("falls back to the basename, handling trailing and Windows separators", () => {
    expect(splitLabel("C:\\runs\\eval_ssa\\")).toBe("eval_ssa");
  });
});
