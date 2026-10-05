"""imports_first: a file's leading imports move above a leading comment, so Kimina sees its header
(the OEIS Open spec, O1; 30 of 38 gold proofs failed the first gate run without it)."""

from rlvr_lean.domain.verification import imports_first

LICENSE = "/-\nCopyright 2026 Google LLC\n\nLicensed under the Apache License.\n-/\n\n"


def test_imports_after_a_license_comment_move_to_the_top_and_the_comment_stays():
    source = LICENSE + "import FormalConjectures.Util.ProblemImports\n\nset_option maxHeartbeats 0\n\ntheorem t : True := trivial\n"
    result = imports_first(source)
    assert result.startswith("import FormalConjectures.Util.ProblemImports\n")
    assert result.count("import ") == 1 and "Copyright 2026" in result and result.endswith("theorem t : True := trivial\n")


def test_a_file_that_already_starts_with_its_imports_is_unchanged():
    source = "import Mathlib\nimport Aesop\n\ntheorem t : True := trivial\n"
    assert imports_first(source) == source


def test_several_imports_keep_their_order():
    source = "-- header\nimport A\nimport B\n\ndef x := 1\n"
    assert imports_first(source).splitlines()[:2] == ["import A", "import B"]


def test_an_import_inside_a_comment_or_after_a_command_does_not_move():
    inside = "/- see\nimport Nothing\n-/\nimport Real\n\ntheorem t : True := trivial\n"
    assert imports_first(inside).splitlines()[0] == "import Real"
    assert imports_first(inside).count("import Nothing") == 1
    after = "theorem t : True := trivial\nimport Late\n"
    assert imports_first(after) == after


def test_a_file_without_imports_is_unchanged():
    assert imports_first("theorem t : True := trivial") == "theorem t : True := trivial"
