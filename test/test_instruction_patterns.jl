#!/usr/bin/env julia
"""
    test_instruction_patterns.jl — Tests for compaction-survival instruction patterns.

Verifies: each of the seven patterns matches its canonical example and
rejects plausible non-matches. Action-class extraction is correct.
"""

include(joinpath(@__DIR__, "..", "apps", "credence-governance-sidecar", "instruction_patterns.jl"))

function msg(text::String)
    Dict{String,Any}("role" => "user", "content" => text)
end

# ── Test 1: Pattern 1 — confirm before deleting ──

println("=" ^ 60)
println("TEST 1: confirm before deleting/removing/dropping")
println("=" ^ 60)

r1 = match_instructions([msg("please confirm before deleting any files")])
@assert length(r1) == 1 "Should match, got $(length(r1))"
@assert r1[1][2] == "delete" "Action class should be 'delete', got '$(r1[1][2])'"

r1b = match_instructions([msg("confirm before removing the database")])
@assert length(r1b) == 1 && r1b[1][2] == "delete"

r1c = match_instructions([msg("confirm before dropping the table")])
@assert length(r1c) == 1 && r1c[1][2] == "delete"

r1_neg = match_instructions([msg("I confirmed the deletion already")])
@assert isempty(r1_neg) "'I confirmed the deletion' should NOT match"

r1_neg2 = match_instructions([msg("the file was deleted before I could check")])
@assert isempty(r1_neg2) "'was deleted before' should NOT match"

println("PASSED: confirm-before-delete pattern")
println()

# ── Test 2: Pattern 2 — don't/do not/never delete ──

println("=" ^ 60)
println("TEST 2: don't/do not/never delete/remove/drop")
println("=" ^ 60)

r2 = match_instructions([msg("don't delete anything without asking")])
@assert length(r2) == 1 && r2[1][2] == "delete"

r2b = match_instructions([msg("do not remove files from production")])
@assert length(r2b) == 1 && r2b[1][2] == "delete"

r2c = match_instructions([msg("never drop tables without backup")])
@assert length(r2c) == 1 && r2c[1][2] == "delete"

r2_neg = match_instructions([msg("the user denied my request to delete")])
@assert isempty(r2_neg) "'denied my request to delete' should NOT match"

r2_neg2 = match_instructions([msg("I deleted the file yesterday")])
@assert isempty(r2_neg2) "'I deleted' should NOT match"

println("PASSED: negation-delete pattern")
println()

# ── Test 3: Pattern 3 — always/must ask before ──

println("=" ^ 60)
println("TEST 3: always/must ask before")
println("=" ^ 60)

r3 = match_instructions([msg("always ask before running destructive commands")])
@assert length(r3) == 1 && r3[1][2] == "any-destructive"

r3b = match_instructions([msg("you must ask before making changes")])
@assert length(r3b) == 1 && r3b[1][2] == "any-destructive"

r3_neg = match_instructions([msg("I asked about the deployment")])
@assert isempty(r3_neg) "'I asked about' should NOT match"

r3_neg2 = match_instructions([msg("they should ask the team first")])
@assert isempty(r3_neg2) "'should ask' (no always/must) should NOT match"

println("PASSED: ask-before-any pattern")
println()

# ── Test 4: Pattern 4 — confirm before pushing/deploying/merging ──

println("=" ^ 60)
println("TEST 4: confirm before pushing/deploying/merging")
println("=" ^ 60)

r4 = match_instructions([msg("confirm before pushing to production")])
@assert length(r4) == 1 && r4[1][2] == "deploy"

r4b = match_instructions([msg("please confirm before deploying the service")])
@assert length(r4b) == 1 && r4b[1][2] == "deploy"

r4c = match_instructions([msg("confirm before merging this PR")])
@assert length(r4c) == 1 && r4c[1][2] == "deploy"

r4_neg = match_instructions([msg("I confirmed the push already")])
@assert isempty(r4_neg) "'confirmed the push' should NOT match"

println("PASSED: confirm-before-deploy pattern")
println()

# ── Test 5: Pattern 5 — don't push/deploy/merge to main/master/prod ──

println("=" ^ 60)
println("TEST 5: don't push/deploy/merge to main/master/prod")
println("=" ^ 60)

r5 = match_instructions([msg("never push to main without review")])
@assert length(r5) == 1 && r5[1][2] == "deploy"

r5b = match_instructions([msg("don't deploy to prod directly")])
@assert length(r5b) == 1 && r5b[1][2] == "deploy"

r5c = match_instructions([msg("do not merge into master without approval")])
@assert length(r5c) == 1 && r5c[1][2] == "deploy"

r5_neg = match_instructions([msg("I pushed to main yesterday")])
@assert isempty(r5_neg) "'I pushed to main' should NOT match"

r5_neg2 = match_instructions([msg("don't push to the feature branch")])
@assert isempty(r5_neg2) "'push to feature branch' (not main/master/prod) should NOT match"

println("PASSED: negation-deploy-to-protected pattern")
println()

# ── Test 6: Pattern 6 — don't run as sudo/root ──

println("=" ^ 60)
println("TEST 6: don't run as sudo/root")
println("=" ^ 60)

r6 = match_instructions([msg("don't run anything as root")])
@assert length(r6) == 1 && r6[1][2] == "privileged-exec"

r6b = match_instructions([msg("never run commands with sudo")])
@assert length(r6b) == 1 && r6b[1][2] == "privileged-exec"

r6_neg = match_instructions([msg("I ran the command as root to test")])
@assert isempty(r6_neg) "'I ran as root' should NOT match"

r6_neg2 = match_instructions([msg("the sudo command completed successfully")])
@assert isempty(r6_neg2) "'sudo command completed' should NOT match"

println("PASSED: negation-run-privileged pattern")
println()

# ── Test 7: Pattern 7 — don't install/add/upgrade packages ──

println("=" ^ 60)
println("TEST 7: don't install/add/upgrade packages")
println("=" ^ 60)

r7 = match_instructions([msg("don't install new packages without asking")])
@assert length(r7) == 1 && r7[1][2] == "dependency"

r7b = match_instructions([msg("never add packages to the project")])
@assert length(r7b) == 1 && r7b[1][2] == "dependency"

r7c = match_instructions([msg("do not upgrade any package versions")])
@assert length(r7c) == 1 && r7c[1][2] == "dependency"

r7_neg = match_instructions([msg("I installed the package yesterday")])
@assert isempty(r7_neg) "'I installed the package' should NOT match"

println("PASSED: negation-install-package pattern")
println()

# ── Test 8: Deduplication across messages ──

println("=" ^ 60)
println("TEST 8: Deduplication — same pattern in multiple messages")
println("=" ^ 60)

r8 = match_instructions([
    msg("don't delete anything"),
    msg("never delete files either"),
])
@assert length(r8) == 1 "Same pattern in two messages should deduplicate to 1, got $(length(r8))"

println("PASSED: Deduplication")
println()

# ── Test 9: Multiple distinct patterns ──

println("=" ^ 60)
println("TEST 9: Multiple distinct patterns in one batch")
println("=" ^ 60)

r9 = match_instructions([
    msg("don't delete anything without asking"),
    msg("never push to main without review"),
    msg("don't run anything as root"),
])
@assert length(r9) == 3 "Should find 3 distinct patterns, got $(length(r9))"
classes = Set(x[2] for x in r9)
@assert "delete" in classes
@assert "deploy" in classes
@assert "privileged-exec" in classes

println("PASSED: Multiple distinct patterns")
println()

# ── Test 10: instruction_matches_category ──

println("=" ^ 60)
println("TEST 10: instruction_matches_category helper")
println("=" ^ 60)

@assert instruction_matches_category("delete", "delete") == true
@assert instruction_matches_category("delete", "deploy") == false
@assert instruction_matches_category("any-destructive", "delete") == true
@assert instruction_matches_category("any-destructive", "deploy") == true
@assert instruction_matches_category("any-destructive", "privileged-exec") == true
@assert instruction_matches_category("any-destructive", "dependency") == true
@assert instruction_matches_category("any-destructive", "code") == false
@assert instruction_matches_category("any-destructive", "generic") == false
@assert instruction_matches_category("deploy", "delete") == false

println("PASSED: instruction_matches_category")
println()

# ── Test 11: Content extraction from different message formats ──

println("=" ^ 60)
println("TEST 11: Content extraction (string, array, missing)")
println("=" ^ 60)

r11a = match_instructions([Dict{String,Any}("content" => "don't delete files")])
@assert length(r11a) == 1 "String content should match"

r11b = match_instructions([Dict{String,Any}("content" => [
    Dict{String,Any}("type" => "text", "text" => "don't delete files"),
])])
@assert length(r11b) == 1 "Array content with text block should match"

r11c = match_instructions([Dict{String,Any}("role" => "user")])
@assert isempty(r11c) "Missing content should not crash"

r11d = match_instructions([Dict{String,Any}("content" => 42)])
@assert isempty(r11d) "Non-string content should not crash"

println("PASSED: Content extraction")
println()

# ── Test 12: Empty and edge cases ──

println("=" ^ 60)
println("TEST 12: Empty and edge cases")
println("=" ^ 60)

@assert isempty(match_instructions(Dict{String,Any}[]))
@assert isempty(match_instructions([msg("")]))
@assert isempty(match_instructions([msg("hello world, nothing interesting here")]))

println("PASSED: Edge cases")
println()

println("=" ^ 60)
println("ALL INSTRUCTION PATTERN TESTS PASSED")
println("=" ^ 60)
