param(
      [string]$RepoRoot = (Get-Location).Path
    )

    $ErrorActionPreference = "Stop"

    function Backup-File([string]$p) {
      if (!(Test-Path $p)) { throw "Missing file: $p" }
      $ts = Get-Date -Format "yyyyMMdd_HHmmss"
      $bak = "$p.bak_decision_extract_$ts"
      Copy-Item $p $bak -Force
      Write-Host "[BACKUP] $bak" -ForegroundColor DarkGray
      return $bak
    }

    Set-Location $RepoRoot

    $paper = Join-Path $RepoRoot "paper_trader.py"
    $ptdir = Join-Path $RepoRoot "pt"
    $decision = Join-Path $ptdir "decision_pipeline.py"

    if (!(Test-Path $ptdir)) { throw "Missing pt/ directory at: $ptdir" }

    # 1) Write pt\decision_pipeline.py if not present
    if (!(Test-Path $decision)) {
      throw "pt\decision_pipeline.py not found. Copy it from the provided zip first."
    }

    # 2) Patch paper_trader.py
    Backup-File $paper | Out-Null
    $src = Get-Content $paper -Raw -Encoding UTF8

    # 2a) Ensure import exists (place near other pt imports; safe append if not found)
    if ($src -notmatch '(?m)^\s*from\s+pt\.decision_pipeline\s+import\s+decide_and_maybe_place_entry\s*$') {
      # Insert after existing AIHooks import if present, else after first pt.* import line, else after typing imports.
      $ins = "from pt.decision_pipeline import decide_and_maybe_place_entry`r`n"
      if ($src -match '(?m)^\s*from\s+pt\.ai_hooks\s+import\s+AIHooks\s*$') {
        $src = [regex]::Replace($src, '(?m)^(?<L>\s*from\s+pt\.ai_hooks\s+import\s+AIHooks\s*)$', '${L}`r`n' + $ins, 1)
      } elseif ($src -match '(?m)^\s*from\s+pt\.[a-zA-Z0-9_]+\s+import\s+.*$') {
        $src = [regex]::Replace($src, '(?m)^(?<L>\s*from\s+pt\.[a-zA-Z0-9_]+\s+import\s+.*)$', '${L}`r`n' + $ins, 1)
      } else {
        $src = $ins + $src
      }
      Write-Host "[OK] Added import: pt.decision_pipeline" -ForegroundColor Green
    } else {
      Write-Host "[OK] Import already present" -ForegroundColor Green
    }

    # 2b) Replace the inner "if cand:" decision/AI/placement block with a call
    # We match the unique comment "# 1) Bandit choice as before" inside the cand block.
    $pattern = '(?s)(?m)^(?<indent>\s*)if\s+cand\s*:\s*\r?\n(?<block>(?:\k<indent>\s{4}#\s*1\)\s*Bandit\s+choice\s+as\s+before.*?\r?\n)(?:.*?)(?:\k<indent>\s{4}current_arm\s*=\s*chosen\s*\r?\n))'
    if ($src -notmatch $pattern) {
      throw "Could not locate the target cand decision block (marker '# 1) Bandit choice as before'). No changes applied."
    }

    $src = [regex]::Replace($src, $pattern, {
      param($m)
      $indent = $m.Groups["indent"].Value
      $i2 = $indent + "    "   # inside if cand
      $i3 = $indent + "        " # inside call formatting
      $repl = @"
${indent}if cand:
${i2}# Decision + AI advisory/guardrails extracted to pt.decision_pipeline (behavior-preserving)
${i2}c20_max = max(C[-20:]) if len(C) >= 20 else close
${i2}chosen, advice = decide_and_maybe_place_entry(
${i3}args=args,
${i3}log=log,
${i3}learner=learner,
${i3}ai=ai,
${i3}snapshot=snapshot,
${i3}cand=cand,
${i3}close=float(close),
${i3}last_bar_ts=last_bar_ts,
${i3}net_qty=int(net_qty),
${i3}place_bracket_fn=place_bracket,
${i3}fast=(float(fast) if not math.isnan(fast) else None),
${i3}slow=(float(slow) if not math.isnan(slow) else None),
${i3}c20_max=float(c20_max),
${i3}shadow_veto_learn_fn=_shadow_learn_on_veto,
${i3}strat_path=str(strat_path),
${i3}state=state,
${i3}session_key=k,
${i2})
${i2}if chosen:
${i3}current_arm = chosen
"@
      return $repl
    }, 1)

    Set-Content -Path $paper -Value $src -Encoding UTF8
    Write-Host "[OK] Patched paper_trader.py (decision pipeline extracted)" -ForegroundColor Green

    # 3) Compile check
    & (Join-Path $RepoRoot ".venv\Scripts\python.exe") -m py_compile $paper $decision
    Write-Host "[OK] Compile: paper_trader.py + pt\decision_pipeline.py" -ForegroundColor Green
