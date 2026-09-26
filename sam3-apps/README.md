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

The macOS app caches image-and-text query results in `.sam3-zimo` under the
working directory. Repeating a query for the same image skips vision encoding
and text lookup. Open a video, enter a word, and press **Find by Word** to play
with inference on every frame. Playback follows the video's timestamps when
inference is fast enough; otherwise it advances as frames finish. **Play** and
**Pause** control playback, and Play after the end restarts it. Replaying the
same video with the same word reuses cached frame results. Linux runs text
queries directly.
