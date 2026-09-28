# SAM 3 native apps

The macOS and Linux desktop apps share this `build.zig` and use the SAM 3 library in `../../sam3`.
Use Zig 0.16.0 or newer from this directory.

On macOS:

```sh
zig build run --release=fast
```

On Linux with Wayland:

```sh
zig build run --release=fast -Dbackend=cuda
```

The default backend is Metal on macOS and OpenCL on Linux. Pass `-Dbackend=cuda`
and optionally `-Dsm=sm_61` for CUDA. `zig build` installs the app executable
under `zig-out/bin`. `zig build test` runs the Linux font tests on Linux.

Both apps cache image-and-text query results and text features by phrase
in `.sam3-zimo` under the working directory. Repeating a query for the same
image skips vision encoding and text lookup; using the same phrase on a different
frame reuses the text features. Open a video, enter a word, and press
**Find by Word** to process the current frame. On macOS, **Play** advances with
inference on each frame, following video timestamps when inference is fast enough.
On Linux, **Play** processes each frame before advancing when a word is set.
Cached frames replay at the video's pace; uncached frames wait for inference.
**Pre-cache Video** prepares masks for smooth replay. On Linux, opening or
seeking shows a decoded frame with its cached mask, if available. Press
**Find by Word** while paused to process the current frame. **Play** and
**Pause** control playback, and Play after the end restarts it. Replaying the
same video with the same word reuses cached frame results. On Linux, install
`ffmpeg` and `ffprobe` on `PATH` to open videos.

The video controls also include a timeline for seeking, **Restart**, and
**Next Frame** for stepping while paused. On macOS, opening or seeking a video
shows the decoded frame immediately; when a word is set, its mask appears after
inference. On Linux, **Next Frame** processes the stepped frame.
**Pre-cache Video** scans the entire video for the entered word and shows progress.
On macOS, it shows frames processed and precise percentage progress. Both apps
log scan progress to stdout about once a second.
You can cancel the scan; completed frames remain cached. The scanned word becomes
the playback query. Playback and seeking stay available during the scan: cached
frames show masks, while frames still waiting for inference show the plain video.
All returned masks appear together in distinct colors. Select a mask in the bar
to make it more prominent while keeping the others visible.
