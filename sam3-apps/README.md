# SAM 3 native apps

The macOS and Linux desktop apps share this `build.zig` and use the SAM 3 library in `../../sam3`.
Use Zig 0.16.0 or newer from this directory.

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
`zig build` installs the app executable under `zig-out/bin`. `zig build test` runs the Linux font tests on Linux.

Both apps cache image-and-text query results and text features by phrase
in `.sam3-zimo` under the working directory. Repeating a query for the same
image skips vision encoding and text lookup; using the same phrase on a different
frame reuses the text features. On Linux, install `ffmpeg` and `ffprobe` on `PATH`
to open videos.

Use **Open…** to select an image or video; the app detects the file type.
Enter SQL in the query box and press **Run Query**. Create a visual index with:

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

Queries and SQL indexing run in the background. You can open another image or
video while a job continues; its source video, cached masks, and matches stay
separate from the displayed file. Reopen the source video to browse its matches.
Each SQL query opens its own tab with its SQL, matches, status, and cancellation
control. You can edit the input and press **Run Query** to start another query while
earlier tabs continue. Select a tab to show its video and results. **Cancel Query**
stops only the selected tab. Model inference is shared and runs one call at a time.
Closing the app cancels and joins all workers before releasing the model.

Masks appear only for `sam3(...)` expressions in `SELECT`. A query such as
`SELECT frame WHERE sam3(frame, "hat") > 0.9` filters using SAM 3 and displays
the matching frames without overlays.

The video controls include **Play/Pause**, a timeline for seeking, **Restart**, and
**Next Frame** (or **Next Match** for query results). Playback and seeking stay
available during indexing. Cached frames show masks; frames waiting for inference
show the plain video. All returned masks appear together in distinct colors.
Select a mask in the bar to make it more prominent while keeping the others visible.

For an original PyTorch SAM 3 comparison on Intel GPUs, see the
[Python benchmark and Nix environment](benchmarks/sam3-python/README.md).
