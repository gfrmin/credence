# apps/answer-brain/packaging

The `systemd --user` unit that runs the answer-brain decision daemon on `:8799`.

It sits next to the daemon it starts (`../daemon/main.jl`) so the two version together —
`WorkingDirectory` and the `--project=.` in `ExecStart` are a claim about this repo's
layout, and a unit kept anywhere else would silently rot the first time that layout moved.

Until 2026-08-21 this unit existed only in `~/.config/systemd/user/` on one machine: in no
repo, and on no backup source list. life-agent's ask read-path depends on this daemon
answering, so the deployment of a documented dependency was the least durable part of it.

## Install

```bash
cd ~/git/credence
ln -sfn "$PWD"/apps/answer-brain/packaging/answer-brain-daemon.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now answer-brain-daemon.service
```

Symlink, never copy — a copied unit is a second copy free to drift, and the repo stops
being the source.

## Paths

`WorkingDirectory=%h/git/credence` assumes the repo is checked out at `~/git/credence`;
edit it if yours is elsewhere. The installed copy hard-coded `/home/g/...` for both this
and `PATH`, which is fine on one machine and wrong in a public repo — `%h` is systemd's
own specifier for the user's home and is what makes the unit portable.

`ExecStart` uses the system `julia` at `/usr/bin/julia` deliberately: the daemon runs
against the repo's `Project.toml`, not a pinned toolchain, so a Julia upgrade is visible
here rather than silently absorbed.
