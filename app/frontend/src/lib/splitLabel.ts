/**
 * `eval_dir` is an arbitrary path (`NVX_EVAL_DIR`), and this project
 * routinely evaluates on val as well as test - so the split shown here must
 * come from the directory name actually in use, never be hardcoded.
 */
export function splitLabel(evalDir: string): string {
  const segments = evalDir.split(/[\\/]/).filter(Boolean);
  const basename = segments[segments.length - 1] ?? evalDir;

  // Match whole WORDS, never substrings. The default directory is
  // `eval_test_baseline_unet3d`, and "eval" contains "val" -- a substring
  // test labels the test split as "val split", which is a false statement
  // about which data is on screen. Same trap as the BraTS modality suffixes,
  // where `"_t1" in name` is also true for `_t1ce`.
  const words = basename.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
  if (words.includes("val")) return "val split";
  if (words.includes("test")) return "test split";
  return basename;
}
