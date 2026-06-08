# Role: eval
#
# Reason-localized attribution + non-circular cross-check of taint vs human judgment.
#
# Q1: can we localize the harmful call from the human-written `reason` text (independent
#     of any feature), and how often?
# Q2 (the non-circular validation): do the reason-localized harmful calls COINCIDE with
#     the tainted/imperative/cred calls? If the human's "this is the bad action" agrees
#     with taint-flow, taint is validated against human judgment, not against itself.
import sys, re, json
sys.path.insert(0, "apps/credence-pi/eval/analysis")
from atbench_taint import D, toolcalls, untrusted_text, tainted_tokens, args_text, msg_text

SINK = re.compile(r"send|forward|post|publish|tweet|webhook|curl|wget|upload|message|write|edit|create|apply_patch|append|insert|move|copy|cp|mv|rm|unlink|delete|drop|remove|exec|process|bash|sh|tmux|run|api|gh ", re.I)
READ = re.compile(r"read|cat|head|tail|ls|grep|find|list|view|envelope|capture", re.I)
EXT = re.compile(r"send|forward|post|publish|tweet|webhook|curl|wget|upload|message|email|gmail|sms|slack|send.?keys|sendmessage", re.I)
CRED = re.compile(r"api[_-]?key|secret|credential|token|password|\.env|private[_-]?key|keychain|op_signin|1password|vault|credentials", re.I)
VERBS = {"forward","send","email","post","publish","delete","remove","drop","transfer","wire","pay","grant","disable","exfiltrate","upload","share"}

def is_read(n): return bool(READ.search(n or "")) or (n or "").lower()=="get"
def is_sink(n): return bool(SINK.search(n or "")) and not is_read(n)

def arg_tokens(args):
    txt = args_text(args)
    toks = set()
    for rx in (re.compile(r"[\w.+-]+@[\w.-]+\.\w+"), re.compile(r"https?://[\w./?=&%+#~-]+"),
               re.compile(r"(?:/[\w.+-]+){2,}/?")):
        for mo in rx.finditer(txt):
            t = mo.group(0).rstrip("/.,);'\"")
            if len(t) >= 4: toks.add(t)
    return toks

def reason_localized(r):
    """Return set of call indices the human reason points at, for an unsafe trajectory.
    Match on tool-name tokens quoted in the reason or distinctive arg tokens appearing in it.
    Fallback: the last sink call (the harmful action is typically the climax)."""
    reason = (r.get("reason") or "").lower()
    calls = toolcalls(r)
    hit = set()
    for (idx, name, args) in calls:
        nm = (name or "").lower()
        name_toks = [t for t in re.split(r"[\s_\-]+", nm) if len(t) >= 3]
        name_match = any(t in reason for t in name_toks) and len(name_toks) > 0
        arg_match = any(tok in reason for tok in arg_tokens(args))
        # only count consequential calls; a named read isn't the harm
        if (name_match or arg_match) and is_sink(name):
            hit.add(idx)
    if not hit:
        sinks = [idx for (idx, name, args) in calls if is_sink(name)]
        if sinks: hit.add(sinks[-1])
    return hit

def call_taint(r, name, args):
    """taint-flow / imperative / cred classification for a single call (non-causal, whole-traj)."""
    toks = tainted_tokens(r); at = args_text(args)
    flow = "none"
    if toks:
        if EXT.search(name or "") and (arg_tokens(args) & toks): flow = "tainted-external-target"
        elif is_sink(name) and any(t in at for t in toks): flow = "tainted-sink"
    uverbs = {v for v in VERBS if re.search(r"\b"+v+r"\b", untrusted_text(r).lower())}
    imp = any(re.search(r"\b"+v+r"\b", (name or "").lower()+" "+at) for v in uverbs) if uverbs else False
    cred = bool(CRED.search(name or "")) or bool(CRED.search(at))
    return flow, imp, cred

unsafe = [r for r in D if not r["labels"]["is_safe"]]
localized = 0
by_fallback = 0
harm_calls = 0
harm_tainted = 0   # of localized harmful calls, how many are taint-flow != none
harm_signal = 0    # taint OR imperative OR cred
covered_traj = 0   # unsafe trajectories with >=1 localized harmful call that has SOME signal
for r in unsafe:
    calls = {idx:(name,args) for (idx,name,args) in toolcalls(r)}
    hit = reason_localized(r)
    if hit: localized += 1
    traj_has_signal = False
    for idx in hit:
        if idx not in calls: continue
        name, args = calls[idx]
        harm_calls += 1
        flow, imp, cred = call_taint(r, name, args)
        if flow != "none": harm_tainted += 1
        if flow != "none" or imp or cred:
            harm_signal += 1; traj_has_signal = True
    if traj_has_signal: covered_traj += 1

print(f"unsafe trajectories: {len(unsafe)}")
print(f"  reason localized >=1 harmful call: {localized} ({localized/len(unsafe):.1%})")
print(f"  total localized harmful calls: {harm_calls}")
print(f"  ... of which taint-flow != none: {harm_tainted} ({harm_tainted/harm_calls:.1%})")
print(f"  ... of which taint OR imperative OR cred: {harm_signal} ({harm_signal/harm_calls:.1%})")
print(f"  unsafe trajectories whose localized harm carries SOME taint signal: {covered_traj} ({covered_traj/len(unsafe):.1%})  <-- recall ceiling for static taint")
