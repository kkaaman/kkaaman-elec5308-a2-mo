# CARLA 0.9.16 Windows RGB readback repair

The supported Windows package stopped advancing after six camera ticks and
five RGB receipts. Navigation without RGB continued to work. A fresh client
could still query the server version, but world/settings RPCs timed out.
Updating the NVIDIA driver from 552.22 to 617.42 did not remove this failure.

## Diagnosed task-routing defect

Local executable/PDB stack inspection showed `FStatsThread::Run` executing
the image-readback lambda in `PixelReader.cpp`, repeatedly yielding while
waiting for GPU readback. The game thread waited in `FStats::AdvanceFrame`
and `FThreadStats::WaitForStats`. The compiled `WritePixelsToBuffer` call
passed task target `0x200`: `ENamedThreads::HighTaskPriority` alone. In this
build its thread-index bits select StatsThread instead of an unnamed worker.

The source correction is:

```cpp
AsyncTask(ENamedThreads::AnyNormalThreadHiPriTask,
          [=, Readback=std::move(BackBufferReadback)]() mutable {
    // Existing GPU wait, buffer validation and delivery remain unchanged.
});
```

The complete one-line source diff is in
`patches/carla-0.9.16-rgb-readback-worker.patch`. Apply it to matching CARLA
source and rebuild the Windows server when a source build is available.
The original implementation is visible in the
[CARLA 0.9.16 source](https://github.com/carla-simulator/carla/blob/0.9.16/Unreal/CarlaUE4/Plugins/Carla/Source/Carla/Sensor/PixelReader.cpp).

## D3D11 immediate-context protection

Moving readback to workers removed the deterministic StatsThread deadlock.
Long-session testing then exposed a separate D3D11 device internal error
(`0x887A0020`). Enabling RHI threading alone did not prevent it. The immediate
context was not protected for concurrent access. Microsoft's
[D3D11 threading contract](https://learn.microsoft.com/en-us/windows/win32/api/d3d11_4/nn-d3d11_4-id3d11multithread)
requires synchronization when sharing that context across threads.

The isolated runtime adds the source-built, application-local shim in
`tools/carla_rgb_readback/`. It loads the real D3D11 library by its absolute
System32 path, removes the single-thread-only device flag and enables
`ID3D11Multithread::SetMultithreadProtected(TRUE)`. The native log records
protection before and after creation. It does not replace a Windows DLL.

## Reproduce the verified runtime

For the exact installed package, `scripts/patch_carla_rgb_readback.py` builds
a separate runtime directory. It changes the task target to `0x2ff` and
preserves the remaining executable bytes, compiles the local D3D11 guard,
copies the project descriptor/registry and native dependencies, and shares
the existing assets through directory junctions. Keep the original CARLA
installation available because the runtime uses these shared assets.

| Image | SHA256 |
|---|---|
| Official supported image | `ba05ec36f7f0b548b266865ea2787a1dc5a028e298ebefcf0c49196ac629c4d3` |
| Fixed copy | `9deba9928ffe56ee8c27cc89d88bcdd4e49204b771f86da2e438c613f4205245` |

The utility refuses every other source hash and verifies an existing runtime
before accepting it. It never overwrites the official executable or an
unrecognized runtime directory. Generated executables/DLLs and shared assets
are not committed. This machine already has MinGW-w64 at the compiler path
below; another machine needs an existing Windows x64 MinGW-w64 compiler.

```powershell
Set-Location 'D:\26-S2-Assignment2-Group7'
$py = '.\.venv\Scripts\python.exe'
$carlaRoot = 'D:\CARLA_0.9.16'
$runtime = 'D:\26-S2-Assignment2-Group7\runs\carla-rgb-runtime'

& $py scripts/patch_carla_rgb_readback.py `
    --carla-root $carlaRoot `
    --runtime $runtime `
    --cxx 'D:\mingw64\bin\g++.exe'
if ($LASTEXITCODE -ne 0) { throw 'CARLA RGB repair verification failed.' }

Start-Process `
    -FilePath "$runtime\CarlaUE4\Binaries\Win64\CarlaUE4-Win64-Shipping.exe" `
    -WorkingDirectory $runtime `
    -ArgumentList '-dx11 -windowed -ResX=640 -ResY=360 -quality-level=Low -nosound' `
    -WindowStyle Normal
```

Close existing CARLA instances before starting this copy. Wait for the map,
then run the unchanged mission configuration:

```powershell
& $py scripts/check_baseline_mission.py --verify-map
$runId = Get-Date -Format 'yyyyMMdd-HHmmss'
& $py scripts/run_scenario.py --runs 5 --output "runs/S0/rgb-$runId"
```

Returning to the original installation means closing the isolated process
and starting the official executable in the original installation directory.
No driver or registry changes are made by the repair utility.

## Validation scope

The task-routing correction first passed an independent capture probe, an
RGB-enabled complete mission and five consecutive S0 runs. The final
context-protected runtime passed another 300-tick capture probe (299 image
receipts with the expected GPU pipeline delay). The first saved image hash matched the
unmodified server's first valid image. The repair does not remove the GPU
readiness wait, manufacture sensor data, change acceptance thresholds or alter
the frozen route/controller configuration.

The task-only repair's plain-DX11 long-session failure and subsequent failed
RHI-thread trial are retained. The final context-protected runtime passed
two complete five-run suites and all three RGB HOLD regressions in one
server session, after the independent capture probe. The runtime generated
by the reproducible builder then passed an additional five-run suite.
Consult the latest
[validation report](evidence/TEST-02/README.md) for raw evidence and limits.
The one-line source patch fixes task routing only; a source build used with
D3D11 must also ensure correct immediate-context synchronization.

Startup readiness must retry with a fresh CARLA client after a failed initial
connection. Reusing the failed client produced misleading startup timeouts in
early diagnostic attempts; it did not cause the independent five-frame camera
deadlock.
