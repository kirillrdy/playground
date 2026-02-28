pub const c = @cImport({
    @cInclude("onnxruntime_c_api.h");
    @cInclude("libavformat/avformat.h");
    @cInclude("libavcodec/avcodec.h");
    @cInclude("libavutil/avutil.h");
    @cInclude("libavutil/hwcontext.h");
    @cInclude("libavutil/hwcontext_cuda.h");
    @cInclude("libavutil/pixdesc.h");
    @cInclude("libavfilter/avfilter.h");
    @cInclude("libavfilter/buffersink.h");
    @cInclude("libavfilter/buffersrc.h");
    @cInclude("cuda_runtime.h");
});
