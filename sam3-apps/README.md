# SAM 3 native apps

The macOS and Linux desktop apps share this `build.zig` and use the SAM 3 library in `../../sam3`.
Use Zig 0.16.0 or newer from this directory.

On macOS:

```sh
zig build run-macos --release=fast
```

On Linux with Wayland:

```sh
zig build run-linux --release=fast -Dbackend=cuda
```

The default backend is Metal on macOS and OpenCL on Linux. Pass `-Dbackend=cuda`
and optionally `-Dsm=sm_61` for CUDA. `zig build` installs the app executable
under `zig-out/bin`. `zig build test` runs the Linux font tests on Linux.

The macOS app caches image-and-text query results and text features by phrase
in `.sam3-zimo` under the working directory. Repeating a query for the same
image skips vision encoding and text lookup; using the same phrase on a different
frame reuses the text features. Open a video, enter a word, and press
**Find by Word** to process the current frame, then press **Play** to advance
with inference on each frame. Playback follows the video's timestamps when
inference is fast enough; otherwise it advances as frames finish. **Play** and
**Pause** control playback, and Play after the end restarts it. Replaying the
same video with the same word reuses cached frame results. Linux runs text
queries directly.

The macOS video controls also include a timeline for seeking, **Restart**, and
**Next Frame** for stepping while paused. Opening or seeking a video shows the
decoded frame immediately; when a word is set, its mask appears after inference.
**Pre-cache Video** scans the entire video for the entered word and shows progress.
You can cancel the scan; completed frames remain cached. The scanned word becomes
the playback query. Playback and seeking stay available during the scan: cached
frames show masks, while frames still waiting for inference show the plain video.
All returned masks appear together in distinct colors. Select a mask in the bar
to make it more prominent while keeping the others visible.
