<#
.SYNOPSIS
    One entry point for VideoCtrl-F: run the app, run the evaluation, see the figures.

.DESCRIPTION
    VideoCtrl-F needs three things running that are easy to get wrong
    individually: a PostgreSQL that actually has the pgvector extension, a
    backend serving the API, and a generated results directory. This script
    checks all three, fixes what it can, and tells you plainly what it cannot.

    On this machine PostgreSQL+pgvector runs inside WSL2 on port 5433, because
    the Windows-side PostgreSQL 18 cannot be given pgvector (no compiler, and
    share\extension is not writable without administrator rights). WSL2 assigns
    a new IP on every boot, so `doctor` re-reads it and repairs .env.

.EXAMPLE
    .\RUN.ps1                 # same as -Task all
    .\RUN.ps1 -Task doctor    # check everything, change nothing
    .\RUN.ps1 -Task serve     # start backend + frontend, Ctrl+C to stop
    .\RUN.ps1 -Task eval      # run the full evaluation study
    .\RUN.ps1 -Task plots     # regenerate the 8 figures
    .\RUN.ps1 -Task results   # open SUMMARY.md and the figures folder
    .\RUN.ps1 -Task test      # run the pytest suite

.NOTES
    No task writes to the database except `eval`, which only reads it, and
    `init`, which creates schema. `index` is the only task that writes rows.
#>

[CmdletBinding()]
param(
    [ValidateSet(
        'all', 'doctor', 'db', 'init', 'serve', 'test', 'eval', 'plots',
        'results', 'index', 'status', 'help'
    )]
    [string]$Task = 'all',

    # Skip the interactive confirmation before destructive work.
    [switch]$Force,

    [int]$TopK = 10,
    [int]$GraphDepth = 2,

    # Optional video to index. With no value, lists what is already indexed.
    [string]$Video
)

$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

$RepoRoot   = Split-Path -Parent $PSScriptRoot
if (-not (Test-Path (Join-Path $RepoRoot 'backend'))) { $RepoRoot = (Get-Location).Path }

$BackendPy  = Join-Path $RepoRoot 'backend\.venv\Scripts\python.exe'
$EnvFile    = Join-Path $RepoRoot '.env'
$ResultsDir = Join-Path $RepoRoot 'experiments\results'
$FiguresDir = Join-Path $ResultsDir 'figures'
$Dataset    = Join-Path $RepoRoot 'experiments\datasets\evaluation.csv'

$ApiUrl     = 'http://localhost:8000'
$WebUrl     = 'http://localhost:5173'
$DbPort     = 5433

$PlotScripts = @(
    'recall', 'precision', 'mrr', 'ndcg', 'latency',
    'frame_reduction', 'ablation', 'comparison'
)

# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

function Write-Step  { param([string]$m) Write-Host "`n==> $m" -ForegroundColor Cyan }
function Write-Ok    { param([string]$m) Write-Host "    [OK]   $m" -ForegroundColor Green }
function Write-Warn2 { param([string]$m) Write-Host "    [WARN] $m" -ForegroundColor Yellow }
function Write-Fail  { param([string]$m) Write-Host "    [FAIL] $m" -ForegroundColor Red }
function Write-Info  { param([string]$m) Write-Host "           $m" -ForegroundColor DarkGray }

function Confirm-Action {
    param([string]$Question)
    if ($Force) { return $true }
    $ans = Read-Host "    $Question [y/N]"
    return ($ans -match '^(y|yes)$')
}

function Test-Port {
    param([int]$Port)
    try {
        return (Test-NetConnection -ComputerName localhost -Port $Port `
            -WarningAction SilentlyContinue).TcpTestSucceeded
    } catch { return $false }
}

# --- Running Python from PowerShell -----------------------------------------
#
# Passing multi-line Python via `python -c` does NOT work here. PowerShell's
# native-argument marshalling strips the double quotes from the argument, so
# `print("MISSING=" + ...)` reaches the interpreter as `print(MISSING= + ` and
# dies with "SyntaxError: '(' was never closed". Quoting workarounds (single
# quotes, backtick escapes) were all fragile.
#
# The reliable approach is to write the snippet to a temporary .py file and run
# that. `Invoke-PySnippet` does it, and cleans up afterwards.
function Invoke-PySnippet {
    param(
        [Parameter(Mandatory)][string]$Code,
        [string[]]$Arguments = @()
    )
    $tmp = Join-Path ([System.IO.Path]::GetTempPath()) `
                       ("vcqf_{0}.py" -f [guid]::NewGuid().ToString('N').Substring(0, 8))
    try {
        [System.IO.File]::WriteAllText($tmp, $Code, [System.Text.UTF8Encoding]::new($false))
        $all = @($tmp) + $Arguments
        & $BackendPy @all 2>&1
    } finally {
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
    }
}

# Snippets live in single-quoted here-strings so PowerShell never expands $ or
# " inside them, and take their paths from argv rather than being interpolated.
function Get-CheckDatasetCode {
@'
import sys
root, dataset = sys.argv[1], sys.argv[2]
sys.path.insert(0, root)
from experiments.evaluation.dataset import load_dataset, validate_dataset
d = load_dataset(dataset)
v = validate_dataset(d)
print("Q=%d SCORABLE=%d WARN=%d" % (len(d.queries), len(d.scorable), len(v["warnings"])))
'@
}

function Get-CheckPackagesCode {
@'
import importlib.util
mods = ["fastapi", "sqlalchemy", "psycopg2", "pgvector", "torch", "transformers",
        "cv2", "PIL", "matplotlib", "pandas", "alembic", "pytest"]
print("MISSING=" + " ".join(m for m in mods if importlib.util.find_spec(m) is None))
'@
}

function Get-CheckSchemaCode {
@'
import sys
backend = sys.argv[1]
sys.path.insert(0, backend)
try:
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url
    from app.core.config import get_settings
    u = make_url(get_settings().DATABASE_URL)
    e = create_engine("postgresql://%s:%s@%s:%s/%s" % (u.username, u.password, u.host, u.port, u.database))
    with e.connect() as c:
        rows = c.execute(text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema='public' ORDER BY table_name")).scalars().all()
        rev = c.execute(text("SELECT version_num FROM alembic_version")).scalar()
    print("TABLES=" + ",".join(rows) + " ALEMBIC=" + str(rev))
except Exception as exc:
    print("ERR=" + str(exc).replace(chr(10), " ")[:160])
'@
}

function Get-CheckIndexCode {
@'
import sys
backend = sys.argv[1]
sys.path.insert(0, backend)
try:
    from sqlalchemy import func, select
    from app.db.session import SessionLocal
    from app.models.frame import VideoFrame
    from app.models.video import Video
    db = SessionLocal()
    ready = db.execute(select(func.count()).select_from(Video).where(Video.status == "ready")).scalar() or 0
    frames = db.execute(select(func.count()).select_from(VideoFrame).where(VideoFrame.embedding.isnot(None))).scalar() or 0
    print("READY=%d FRAMES=%d" % (ready, frames))
    db.close()
except Exception as exc:
    print("ERR=" + str(exc).replace(chr(10), " ")[:160])
'@
}

function Get-InitSchemaCode {
@'
import pathlib, sys
backend, initfile = sys.argv[1], sys.argv[2]
sys.path.insert(0, backend)
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from app.core.config import get_settings
u = make_url(get_settings().DATABASE_URL)
e = create_engine("postgresql://%s:%s@%s:%s/%s" % (u.username, u.password, u.host, u.port, u.database))
sql = pathlib.Path(initfile).read_text(encoding="utf-8")
with e.begin() as c:
    c.execute(text(sql))
    rows = c.execute(text("SELECT table_name FROM information_schema.tables "
                          "WHERE table_schema='public' ORDER BY table_name")).scalars().all()
print("tables: " + ", ".join(rows))
'@
}

# ---------------------------------------------------------------------------
# Task: doctor — check every prerequisite, repair what is safe to repair
# ---------------------------------------------------------------------------

function Get-WslIp {
    $out = & wsl -u root -- hostname -I 2>$null
    if ($LASTEXITCODE -eq 0 -and $out) { return ($out.Trim() -split '\s+')[0] }
    return $null
}

function Get-DatabaseUrl {
    if (-not (Test-Path $EnvFile)) { return $null }
    $line = Select-String -Path $EnvFile -Pattern '^DATABASE_URL=' | Select-Object -First 1
    if (-not $line) { return $null }
    return ($line.Line -replace '^DATABASE_URL=', '').Trim()
}

function Set-DatabaseUrl {
    param([string]$Url)
    if (-not (Test-Path $EnvFile)) { return $false }
    $content = Get-Content $EnvFile
    $patched = $false
    for ($i = 0; $i -lt $content.Count; $i++) {
        if ($content[$i] -match '^DATABASE_URL=') {
            $content[$i] = "DATABASE_URL=$Url"
            $patched = $true
        }
    }
    if ($patched) {
        Set-Content -Path $EnvFile -Value $content -Encoding UTF8
        return $true
    }
    return $false
}

function Invoke-Doctor {
    $problems = 0

    Write-Step 'Python environment'
    if (Test-Path $BackendPy) {
        Write-Ok "venv present: $BackendPy"
        $ver = & $BackendPy --version 2>&1 | Select-Object -Last 1
        Write-Info "python $ver"
    } else {
        Write-Fail "venv missing. Run: python -m venv backend\.venv"
        Write-Info 'then: backend\.venv\Scripts\pip install -r backend\requirements.txt'
        $problems++
    }

    Write-Step 'Required Python packages'
    if (Test-Path $BackendPy) {
        $out = Invoke-PySnippet (Get-CheckPackagesCode) | Select-Object -Last 1
        $missing = (($out | Out-String) -replace '^MISSING=', '').Trim()
        if ($missing) {
            Write-Fail "missing: $missing"
            Write-Info 'run: backend\.venv\Scripts\pip install -r backend\requirements.txt'
            $problems++
        } else {
            Write-Ok 'all 12 required packages importable'
        }
    }

    Write-Step 'Database (PostgreSQL + pgvector)'
    $dbUp = Test-Port -Port $DbPort
    if (-not $dbUp) {
        $wslIp = Get-WslIp
        if ($wslIp) {
            Write-Warn2 "nothing listening on $DbPort. WSL2 is up (IP $wslIp), so the"
            Write-Info  'cluster is probably stopped. Starting it...'
            & wsl -u root -- pg_ctlcluster 16 main start 2>&1 | Out-Null
            Start-Sleep -Seconds 3
            $dbUp = Test-Port -Port $DbPort
        }
    }

    if ($dbUp) {
        Write-Ok "PostgreSQL listening on $DbPort"
        $wslIp = Get-WslIp
        $dbUrl = Get-DatabaseUrl
        if ($wslIp -and $dbUrl -and $dbUrl -notmatch [regex]::Escape($wslIp)) {
            Write-Warn2 ".env DATABASE_URL points at an old WSL2 IP."
            Write-Info  "current WSL2 IP is $wslIp"
            if (Confirm-Action "Rewrite DATABASE_URL to use $wslIp ?") {
                $new = $dbUrl -replace '@[0-9.]+:', "@$wslIp`:"
                if (Set-DatabaseUrl -Url $new) { Write-Ok 'DATABASE_URL updated' }
                else { Write-Fail 'could not rewrite DATABASE_URL'; $problems++ }
            }
        }
        elseif ($wslIp) {
            Write-Ok "DATABASE_URL matches WSL2 IP $wslIp"
        }
    } else {
        Write-Fail "no database on $DbPort."
        Write-Info 'See MISSING_FILES.md section 3.1 for how PostgreSQL+pgvector is provisioned.'
        $problems++
    }

    Write-Step 'Schema'
    if ($dbUp -and (Test-Path $BackendPy)) {
        $out = Invoke-PySnippet (Get-CheckSchemaCode) @((Join-Path $RepoRoot 'backend')) |
               Select-Object -Last 1
        if ($out -match '^TABLES=([\w,]+) ALEMBIC=(.+)$') {
            $names = $Matches[1] -split ','
            $rev = $Matches[2]
            $need = @('videos', 'video_frames', 'graph_nodes', 'graph_edges')
            $absent = $need | Where-Object { $_ -notin $names }
            if ($absent) {
                Write-Warn2 "missing tables: $($absent -join ', '). Run: .\RUN.ps1 -Task init"
                $problems++
            } else {
                Write-Ok "all 4 tables present ($($names -join ', '))"
            }
            Write-Info "alembic revision: $rev"
        } else {
            Write-Fail "could not query schema: $out"
            $problems++
        }
    }

    Write-Step 'Indexed data'
    if ($dbUp -and (Test-Path $BackendPy)) {
        $out = Invoke-PySnippet (Get-CheckIndexCode) @((Join-Path $RepoRoot 'backend')) |
               Select-Object -Last 1
        if ($out -match 'READY=(\d+) FRAMES=(\d+)') {
            $ready = $Matches[1]; $frames = $Matches[2]
            if ([int]$ready -gt 0) {
                Write-Ok "$ready video(s) ready, $frames embedded keyframes"
            } else {
                Write-Warn2 'no indexed videos, so retrieval metrics would be N/A.'
                Write-Info 'run: .\RUN.ps1 -Task index -Video C:\path\to\clip.mp4'
            }
        } else {
            Write-Warn2 "could not query indexed data: $out"
        }
    }

    Write-Step 'Evaluation dataset'
    if (Test-Path $Dataset) {
        $line = (Invoke-PySnippet (Get-CheckDatasetCode) @($RepoRoot, $Dataset) |
                 Select-Object -Last 1)
        if ($line -match 'Q=(\d+) SCORABLE=(\d+) WARN=(\d+)') {
            $nq = $Matches[1]; $ns = $Matches[2]; $nw = $Matches[3]
            if ([int]$nw -eq 0) {
                Write-Ok "$nq queries, $ns scorable, 0 warnings"
            } else {
                Write-Warn2 "$nq queries, $ns scorable, $nw warnings (check CSV quoting)"
            }
        } else {
            Write-Warn2 "dataset check inconclusive: $line"
        }
    } else {
        Write-Fail "dataset missing: $Dataset"
        $problems++
    }

    Write-Step 'Prior results'
    $csvs = @('retrieval_metrics.csv', 'ablation_results.csv',
              'latency_results.csv', 'indexing_results.csv', 'graph_results.csv')
    $have = $csvs | Where-Object { Test-Path (Join-Path $ResultsDir $_) }
    $figs = @(Get-ChildItem $FiguresDir -Filter *.png -ErrorAction SilentlyContinue).Count
    if ($have.Count -eq $csvs.Count -and $figs -gt 0) {
        Write-Ok "$($have.Count)/$($csvs.Count) result CSVs, $figs figures"
    } else {
        Write-Warn2 "$($have.Count)/$($csvs.Count) CSVs, $figs figures. Run -Task eval then -Task plots"
    }

    Write-Step 'Services'
    if (Test-Port 8000) { Write-Ok "backend  $ApiUrl" } else { Write-Info "backend  $ApiUrl  (not running)" }
    if (Test-Port 5173) { Write-Ok "frontend $WebUrl" } else { Write-Info "frontend $WebUrl  (not running)" }

    Write-Host ''
    if ($problems -eq 0) {
        Write-Host 'DOCTOR: all prerequisites satisfied.' -ForegroundColor Green
    } else {
        Write-Host "DOCTOR: $problems problem(s) need attention." -ForegroundColor Yellow
    }
    return $problems
}

# ---------------------------------------------------------------------------
# Task: init — apply base schema, then the graph migration
# ---------------------------------------------------------------------------

function Invoke-Init {
    Write-Step 'Applying app/db/init.sql (videos, video_frames, HNSW index)'
    if (-not (Confirm-Action 'Continue?')) { return }

    $init = Join-Path $RepoRoot 'backend\app\db\init.sql'
    $out = Invoke-PySnippet (Get-InitSchemaCode) @((Join-Path $RepoRoot 'backend'), $init)
    if ($LASTEXITCODE -ne 0) {
        Write-Fail 'init.sql failed'
        $out | Select-Object -Last 5 | ForEach-Object { Write-Info $_ }
        return
    }
    Write-Ok "base schema applied. $($out | Select-Object -Last 1)"

    Write-Step 'Applying Alembic migration (graph_nodes, graph_edges)'
    Push-Location (Join-Path $RepoRoot 'backend')
    try {
        & $BackendPy -m alembic upgrade head 2>&1 |
            Select-String -Pattern 'Running upgrade|ERROR|error' | ForEach-Object { Write-Info $_ }
        if ($LASTEXITCODE -ne 0) { Write-Fail 'alembic upgrade failed'; return }
    } finally { Pop-Location }
    Write-Ok 'graph layer present (alembic 0001_graph_layer)'
}

# ---------------------------------------------------------------------------
# Task: serve — backend + frontend, Ctrl+C stops both
# ---------------------------------------------------------------------------

function Invoke-Serve {
    Write-Step 'Preflight'
    if ((Invoke-Doctor) -gt 0) {
        Write-Warn2 'Continuing anyway; fix anything above if startup fails.'
    }

    # Refuse to start if something already holds the ports. Without this the
    # uvicorn bind fails with WinError 10048 and Vite silently moves the UI to
    # 5174, so the printed URLs point at the *old* servers, not the ones this
    # task just launched.
    $busy = @()
    if (Test-Port 8000) { $busy += 8000 }
    if (Test-Port 5173) { $busy += 5173 }
    if ($busy.Count -gt 0) {
        Write-Warn2 "port(s) already in use: $($busy -join ', ')"
        Write-Info 'something is already serving them. Stop it first, or:'
        foreach ($p in $busy) {
            $conn = Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue |
                    Select-Object -First 1
            if ($conn) {
                $proc = Get-Process -Id $conn.OwningProcess -ErrorAction SilentlyContinue
                Write-Info ("  to stop the listener on {0}: Stop-Process -Id {1} -Force" -f `
                            $p, $conn.OwningProcess)
            }
        }
        if (Confirm-Action 'Stop those listeners and start fresh?') {
            foreach ($p in $busy) {
                Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue |
                    ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }
            }
            Start-Sleep -Seconds 3
        } else {
            Write-Info 'exiting without starting anything.'
            return
        }
    }

    Write-Step 'Starting backend on :8000 and frontend on :5173'
    Write-Info 'Press Ctrl+C to stop both.'
    Write-Host ''

    $backend = $null
    $frontend = $null
    Push-Location (Join-Path $RepoRoot 'backend')
    try {
        $backend = Start-Process -FilePath $BackendPy `
            -ArgumentList '-m', 'uvicorn', 'app.main:app',
                          '--host', '0.0.0.0', '--port', '8000', '--log-level', 'info' `
            -NoNewWindow -PassThru

        # Wait for the API to actually answer before starting Vite, rather than
        # sleeping a fixed 18 s. Model import on a cold cache makes startup
        # unpredictable; probing the endpoint is deterministic.
        $apiUp = $false
        for ($i = 0; $i -lt 60; $i++) {
            Start-Sleep -Seconds 1
            if ($backend.HasExited) { break }
            try {
                $h = Invoke-RestMethod "$ApiUrl/health" -TimeoutSec 3
                if ($h.status -eq 'ok') { $apiUp = $true; break }
            } catch { }
        }
        if (-not $apiUp) {
            Write-Fail 'backend did not become healthy on :8000'
            return
        }
        Write-Ok "backend healthy on $ApiUrl"

        Push-Location (Join-Path $RepoRoot 'frontend')
        try {
            $env:VITE_API_URL = $ApiUrl
            $frontend = Start-Process -FilePath 'npm.cmd' `
                -ArgumentList 'run', 'dev' -NoNewWindow -PassThru
        } finally { Pop-Location }

        $webUp = $false
        for ($i = 0; $i -lt 60; $i++) {
            Start-Sleep -Seconds 1
            if ($frontend.HasExited) { break }
            if (Test-Port 5173) { $webUp = $true; break }
        }

        Write-Host ''
        Write-Host "    Backend : $ApiUrl"      -ForegroundColor Green
        Write-Host "    API docs: $ApiUrl/docs" -ForegroundColor Green
        if ($webUp) {
            Write-Host "    Frontend: $WebUrl" -ForegroundColor Green
        } else {
            Write-Warn2 "frontend not confirmed on $WebUrl; check the Vite output above"
        }
        Write-Host ''
        Write-Info 'Ctrl+C stops both.'

        while (-not $backend.HasExited -and -not $frontend.HasExited) {
            Start-Sleep -Seconds 3
        }
        Write-Host ''
        Write-Warn2 'A process exited; shutting the other one down.'
    } finally {
        foreach ($p in @($backend, $frontend)) {
            if ($p -and -not $p.HasExited) {
                Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue
            }
        }
        Pop-Location
    }
    Write-Ok 'servers stopped'
}

# ---------------------------------------------------------------------------
# Task: test
# ---------------------------------------------------------------------------

function Invoke-Test {
    Write-Step 'Running pytest suite'
    # pytest-asyncio writes a PytestDeprecationWarning to stderr, and PowerShell
    # surfaces any stderr from a native command as a red error record. That is
    # noise, not a failure, so stderr is folded into stdout and only the
    # summary lines are printed. The exit code is the real signal.
    Push-Location (Join-Path $RepoRoot 'backend')
    try {
        $prev = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $output = & $BackendPy -m pytest -q -p no:warnings 2>&1
        $code = $LASTEXITCODE
        $ErrorActionPreference = $prev

        $output |
            Select-String -Pattern 'passed|failed|error|no tests ran' |
            Select-Object -Last 3 |
            ForEach-Object { Write-Info ($_.Line.Trim()) }

        if ($code -eq 0) { Write-Ok 'test suite passed' }
        else { Write-Fail "pytest exited with code $code" }
    } finally { Pop-Location }
}

# ---------------------------------------------------------------------------
# Task: index — upload a video through the public API
# ---------------------------------------------------------------------------

function Invoke-Index {
    if (-not (Test-Path $BackendPy)) { Write-Fail 'venv missing'; return }

    if (-not $Video) {
        Write-Step 'Indexed videos'
        try {
            $r = Invoke-RestMethod "$ApiUrl/api/v1/videos" -TimeoutSec 30
            if ($r.total -eq 0) {
                Write-Warn2 'no videos indexed.'
                Write-Info 'usage: .\RUN.ps1 -Task index -Video C:\path\to\clip.mp4'
            } else {
                foreach ($v in $r.videos) {
                    Write-Info ("{0}  {1,-12} {2,7:N1}s  {3}" -f `
                        $v.id, $v.status, $v.duration_seconds, $v.original_filename)
                }
            }
        } catch {
            Write-Fail "backend not reachable at $ApiUrl. Start it with -Task serve."
        }
        return
    }

    $path = (Resolve-Path $Video -ErrorAction SilentlyContinue)
    if (-not $path) { Write-Fail "video not found: $Video"; return }
    if (-not (Test-Path $path)) { Write-Fail "video not found: $Video"; return }

    Write-Step "Uploading $path"
    try {
        $r = Invoke-RestMethod "$ApiUrl/api/v1/videos" -Method Post `
            -Form @{ file = Get-Item $path } -TimeoutSec 900
    } catch {
        Write-Fail "upload failed: $($_.Exception.Message)"
        Write-Info 'is the backend running? Start it with -Task serve.'
        return
    }

    $id = $r.id
    Write-Ok "queued. id=$id"
    Write-Info 'indexing runs in a background task; polling...'

    $deadline = (Get-Date).AddMinutes(45)
    while ((Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 5
        try { $d = Invoke-RestMethod "$ApiUrl/api/v1/videos/$id" -TimeoutSec 30 } catch { continue }
        if ($d.status -in @('ready', 'failed')) {
            if ($d.status -eq 'ready') {
                Write-Ok "indexed: $($d.duration_seconds)s, $($d.frame_count) decoded frames"
                Write-Host ''
                Write-Host '    Now annotate experiments\datasets\evaluation.csv' -ForegroundColor Cyan
                Write-Host '    then run:  .\RUN.ps1 -Task eval' -ForegroundColor Cyan
            } else {
                Write-Fail "indexing failed: $($d.error_message)"
            }
            return
        }
    }
    Write-Warn2 'timed out waiting for indexing'
}

# ---------------------------------------------------------------------------
# Task: eval — full evaluation study
# ---------------------------------------------------------------------------

function Invoke-Eval {
    if (-not (Test-Path $BackendPy)) { Write-Fail 'venv missing'; return }

    Write-Step 'Dry run: validating annotations only (no DB, no model)'
    Push-Location $RepoRoot
    try {
        $prev = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'

        $dryOut = & $BackendPy -m experiments.evaluation.evaluate_retrieval `
            --dataset experiments/datasets/evaluation.csv --dry-run 2>&1
        $dry = $LASTEXITCODE
        $ErrorActionPreference = $prev

        $dryOut |
            Select-String -Pattern 'num_queries|num_scorable|num_unscorable|WARNING|metrics_status' |
            ForEach-Object { Write-Info ($_.Line.Trim()) }

        if ($dry -ne 0) {
            Write-Fail 'dataset did not validate; stopping before spending GPU time'
            return
        }

        Write-Step "Running the full study (top_k=$TopK, graph_depth=$GraphDepth)"
        Write-Info 'Ablation A/B/C/D + depth sweep 0-3 + indexing/storage metrics.'
        Write-Info 'First query loads the ~1.5 GB checkpoint (~25 s on this GPU).'

        # stderr carries HuggingFace warnings and the progress bar; folded into
        # stdout so PowerShell does not render them as red error records.
        $prev = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        $evalOut = & $BackendPy -m experiments.evaluation.evaluate_retrieval `
            --dataset experiments/datasets/evaluation.csv `
            --model siglip2 `
            --top-k $TopK `
            --graph --graph-depth $GraphDepth `
            --ablation --depth-sweep --index-metrics `
            --output experiments/results/ 2>&1
        $evalCode = $LASTEXITCODE
        $ErrorActionPreference = $prev

        $evalOut |
            Select-String -Pattern 'Wrote:|\.csv ->|\.json ->|\.md ->|Traceback|ERROR' |
            ForEach-Object { Write-Info ($_.Line.Trim()) }

        if ($evalCode -ne 0) {
            Write-Fail "evaluation failed (exit $evalCode)"
            $evalOut | Select-Object -Last 12 | ForEach-Object { Write-Info $_ }
            return
        }
    } finally { Pop-Location }

    Write-Ok 'results written to experiments\results\'
    Write-Info 'next: .\RUN.ps1 -Task plots'
}

# ---------------------------------------------------------------------------
# Task: plots — regenerate the eight figures
# ---------------------------------------------------------------------------

function Invoke-Plots {
    if (-not (Test-Path $BackendPy)) { Write-Fail 'venv missing'; return }

    if (-not (Test-Path (Join-Path $ResultsDir 'retrieval_metrics.csv'))) {
        Write-Warn2 'no results yet; figures would all read N/A.'
        Write-Info 'run: .\RUN.ps1 -Task eval'
        return
    }

    Write-Step 'Regenerating figures from experiments\results\*.csv'
    Push-Location $RepoRoot
    try {
        $failed = 0
        $prev = $ErrorActionPreference
        $ErrorActionPreference = 'Continue'
        foreach ($s in $PlotScripts) {
            $out = & $BackendPy -m "experiments.visualization.plot_$s" 2>&1
            $line = ($out | Select-String 'wrote' | Select-Object -Last 1)
            if ($line) { Write-Ok ($line.Line -replace '^.*wrote\s+', '') }
            else {
                $failed++
                Write-Fail "plot_$s produced no output"
            }
        }
        $ErrorActionPreference = $prev
    } finally { Pop-Location }

    $n = @(Get-ChildItem $FiguresDir -Filter *.png -ErrorAction SilentlyContinue).Count
    if ($failed -eq 0) { Write-Ok "all $($PlotScripts.Count) figure scripts ran; $n PNGs on disk" }
    else { Write-Warn2 "$failed of $($PlotScripts.Count) figure scripts failed" }
}

# ---------------------------------------------------------------------------
# Task: results — summarise and open
# ---------------------------------------------------------------------------

function Invoke-Results {
    $summary = Join-Path $ResultsDir 'SUMMARY.md'
    $csv = Join-Path $ResultsDir 'ablation_results.csv'

    if (-not (Test-Path $csv)) {
        Write-Warn2 'no results yet.'
        Write-Info 'run: .\RUN.ps1 -Task eval'
        return
    }

    Write-Step 'Headline numbers (from experiments\results\SUMMARY.md)'

    if (Test-Path $csv) {
        $rows = Import-Csv $csv
        Write-Host ''
        Write-Host ('    {0,-5} {1,-16} {2,-8} {3,-8} {4,-8} {5,-9}' -f `
            'cond', 'weights v/g/t/c', 'MRR', 'R@10', 'P@10', 'NDCG@10')
        Write-Host ('    ' + ('-' * 68))
        foreach ($r in $rows) {
            $w = '{0}/{1}/{2}/{3}' -f $r.vector_weight, $r.graph_weight,
                                      $r.temporal_weight, $r.confidence_weight
            Write-Host ('    {0,-5} {1,-16} {2,-8} {3,-8} {4,-8} {5,-9}' -f `
                $r.condition, $w, $r.mrr, $r.'recall@10',
                $r.'precision@10', $r.'ndcg@10')
        }
    }

    Write-Host ''
    if (Test-Path (Join-Path $ResultsDir 'indexing_results.csv')) {
        $ix = Import-Csv (Join-Path $ResultsDir 'indexing_results.csv') | Select-Object -First 1
        Write-Info ("  {0} -> {1} frames kept of {2} decoded ({3}% reduction)" -f `
            $ix.filename, $ix.retained_frames, $ix.original_frames,
            [math]::Round([double]$ix.frame_reduction_percent, 2))
        Write-Info ("  graph: {0} nodes, {1} edges" -f $ix.graph_nodes, $ix.graph_edges)
    }

    $figs = Get-ChildItem $FiguresDir -Filter *.png -ErrorAction SilentlyContinue
    if ($figs) {
        Write-Host ''
        Write-Step "Figures ($($figs.Count))"
        foreach ($f in $figs) {
            Write-Info ("  {0,-28} {1,9:N0} bytes" -f $f.Name, $f.Length)
        }
        Write-Host ''
        Write-Info 'What the numbers mean: see MISSING_FILES.md sections 3.3 and 3.4.'
        Write-Info 'On this corpus the four conditions return identical rankings, and'
        Write-Info 'Recall@10 is below the random baseline. That is a property of a'
        Write-Info 'single screen-recording corpus, not a claim about the method.'
    }

    Write-Host ''
    if (Confirm-Action "Open $summary and the figures folder?") {
        if (Test-Path $summary) { Start-Process notepad.exe -ArgumentList "`"$summary`"" }
        if (Test-Path $FiguresDir) { Invoke-Item $FiguresDir }
    }
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

function Invoke-Status {
    Write-Host ''
    Write-Host '  VideoCtrl-F status' -ForegroundColor Cyan
    Write-Host '  ------------------' -ForegroundColor Cyan
    Write-Host ("  backend  :8000  {0}" -f $(if (Test-Port 8000) { 'RUNNING' } else { 'stopped' }))
    Write-Host ("  frontend :5173  {0}" -f $(if (Test-Port 5173) { 'RUNNING' } else { 'stopped' }))
    Write-Host ("  postgres :$DbPort  {0}" -f $(if (Test-Port $DbPort) { 'RUNNING' } else { 'stopped' }))
    Write-Host ("  wsl2 ip  : {0}" -f (Get-WslIp))
    $shown = Get-DatabaseUrl
    if ($shown) { $shown = $shown -replace '(?<=://)[^:@/]+:[^@/]+(?=@)', '***:***' }
    Write-Host ("  db url   : {0}" -f $shown)
    $figs = @(Get-ChildItem $FiguresDir -Filter *.png -ErrorAction SilentlyContinue).Count
    Write-Host ("  figures  : {0}" -f $figs)
    Write-Host ''
}

switch ($Task) {
    'doctor'  { $null = Invoke-Doctor }
    'db'      { Write-Step "Starting PostgreSQL cluster in WSL2"
                & wsl -u root -- pg_ctlcluster 16 main start 2>&1 | Out-Null
                Start-Sleep -Seconds 3
                if (Test-Port $DbPort) { Write-Ok "PostgreSQL up on $DbPort" }
                else { Write-Fail "still down; try: wsl -u root -- pg_ctlcluster 16 main start" } }
    'init'    { Invoke-Init }
    'serve'   { Invoke-Serve }
    'test'    { Invoke-Test }
    'index'   { Invoke-Index }
    'eval'    { Invoke-Eval }
    'plots'   { Invoke-Plots }
    'results' { Invoke-Results }
    'status'  { Invoke-Status }
    'help'    { Get-Help $PSCommandPath -Detailed }
    'all'     {
        Invoke-Status
        $issues = Invoke-Doctor
        if ($issues -gt 0) {
            Write-Host ''
            Write-Fail "$issues prerequisite problem(s). Fix those, or run tasks individually."
            Write-Info 'e.g. .\RUN.ps1 -Task db     to start PostgreSQL'
            Write-Info '     .\RUN.ps1 -Task init  to apply the schema'
            Write-Info '     .\RUN.ps1 -Task index -Video C:\path\to\clip.mp4'
            break
        }
        Write-Host ''
        Write-Info 'Everything is ready. Next steps:'
        Write-Info '  .\RUN.ps1 -Task serve    start backend + frontend'
        Write-Info '  .\RUN.ps1 -Task eval     run the evaluation study'
        Write-Info '  .\RUN.ps1 -Task plots    generate the 8 figures'
        Write-Info '  .\RUN.ps1 -Task results  summarise and open'
    }
}