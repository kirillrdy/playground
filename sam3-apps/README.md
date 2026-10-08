# SAM 3 native apps

The macOS and Linux desktop apps share this `build.zig` and use the SAM 3 dependency pinned in `build.zig.zon`.
Use Zig 0.17.0 or newer from this directory.

On macOS:

```sh
zig build run --release=fast
```

On Linux with Wayland:

```sh
zig build run --release=fast
```

The default backend is Metal on macOS, and on Linux it defaults to CUDA if `/dev/nvidia0`
is detected (otherwise OpenCL). Pass `-Dbackend=cuda` and optionally `-Dsm=sm_61` for CUDA.
`zig build` installs the app executable under `zig-out/bin`. `zig build test` runs the database tests and, on Linux, the font and Wayland tests.

Both apps cache image-and-text query results and text features by phrase
in `.sam3-zimo` under the working directory. Repeating a query for the same
image skips vision encoding and text lookup; using the same phrase on a different
frame reuses the text features. On Linux, install `ffmpeg` and `ffprobe` on `PATH`
to open videos.

The app opens with a query tab. Enter SQL with a **FROM** path and press
**Run Query** to load that tab's video and results. Use **+** to create another tab.
Each tab retains its editor, results, progress, and cancellation control.

SQL suggestions appear as you type keywords, frame columns, and model functions.
Use **Up/Down** to select a suggestion, **Tab** or a click to accept it, and
**Escape** to dismiss the list. **Return** runs the query; **Shift+Return** adds
a newline. After **FROM**, autocomplete lists matching files and directories from
the filesystem, including relative paths, absolute paths, and `~/` paths. Accept a
directory to browse inside it; accept a file to insert a quoted source path.
Keyword suggestions stay hidden inside quoted prompts and SQL comments.

Create a visual index with:

```sql
CREATE INDEX ON "video.mp4" USING sam3("person");
```

This scans the entire video, caches masks, and saves matching frames and confidence
scores to `video.mp4.vdb`. Existing indexes are loaded when the video opens.
Progress appears during indexing. **Cancel Query** stops the scan after the current
frame; cached masks remain, but the new index is saved only when the scan completes.

Query the indexed video with:

```sql
SELECT frame, sam3(frame, "person") AS mask
FROM "video.mp4"
WHERE sam3(frame, "person") > 0.4;
```

Queries and SQL indexing run in the background. Select a tab to show its source
video, edited SQL, and results. **Run Query** runs in the selected tab; rerunning
cancels that tab's previous scan and replaces its results. Use **+** to run another
query while earlier tabs continue. **Cancel Query** stops only the selected tab.
Model inference is shared and runs one call at a time.
Click **×** on a query tab to close it. Closing a running query cancels its scan
after the current frame and selects a neighboring tab; cached results remain.
Closing the last tab opens a new blank query tab.
Closing the app cancels and joins all workers before releasing the model.

Masks appear only for `sam3(...)` expressions in `SELECT`. A query such as
`SELECT frame WHERE sam3(frame, "hat") > 0.9` filters using SAM 3 and displays
the matching frames without overlays.

Queries that omit `frame` from `SELECT` display a table of the selected columns:

```sql
SELECT timestamp FROM 'foo.mp4' WHERE sam3(frame, "hat") > 0.9;
```

`timestamp` is the decoded frame time in seconds. Column aliases become table
headers. Rows stream into the table while the query runs and remain available
after cancellation. Selecting `frame` (including `frame(...)`) displays video results.

Query an inclusive range of zero-based frame IDs with:

```sql
SELECT frame FROM 'video.mp4' WHERE frame_id BETWEEN 100 AND 200;
```

`frame_id >= 100 AND frame_id <= 200` is equivalent. Frame ranges seek directly
to the starting frame and stop at the upper bound; existing index candidates are
filtered before decoding. Add model predicates or `LIMIT` as usual.

The video controls include **Play/Pause**, a timeline for seeking, **Restart**, and
**Next Frame** (or **Next Match** for query results). Playback and seeking stay
available during indexing. Cached frames show masks; frames waiting for inference
show the plain video. All returned masks appear together in distinct colors.


For an original PyTorch SAM 3 comparison on Intel GPUs, see the
[Python benchmark and Nix environment](benchmarks/sam3-python/README.md).
