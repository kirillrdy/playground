const std = @import("std");
const c = @cImport({
    @cInclude("geos_c.h");
});

export fn geosMessageHandler(fmt: [*c]const u8, ...) void {
    _ = fmt;
    // Simple message handler - in a real app you'd want to print these
}

pub fn main() !void {
    // Initialize GEOS
    c.initGEOS(geosMessageHandler, geosMessageHandler);
    defer c.finishGEOS();

    // Two squares that overlap
    const wkt_a = "POLYGON((0 0, 10 0, 10 10, 0 10, 0 0))";
    const wkt_b = "POLYGON((5 5, 15 5, 15 15, 5 15, 5 5))";

    // Read the WKT into geometry objects
    const reader = c.GEOSWKTReader_create();
    defer c.GEOSWKTReader_destroy(reader);

    const geom_a = c.GEOSWKTReader_read(reader, wkt_a.ptr);
    defer c.GEOSGeom_destroy(geom_a);

    const geom_b = c.GEOSWKTReader_read(reader, wkt_b.ptr);
    defer c.GEOSGeom_destroy(geom_b);

    // Calculate the intersection
    const inter = c.GEOSIntersection(geom_a, geom_b);
    defer c.GEOSGeom_destroy(inter);

    // Convert result to WKT
    const writer = c.GEOSWKTWriter_create();
    defer c.GEOSWKTWriter_destroy(writer);

    // Trim trailing zeros off output
    _ = c.GEOSWKTWriter_setTrim(writer, 1);
    const wkt_inter = c.GEOSWKTWriter_write(writer, inter);
    defer c.GEOSFree(wkt_inter);

    // Print answer
    std.debug.print("Geometry A:         {s}\n", .{wkt_a});
    std.debug.print("Geometry B:         {s}\n", .{wkt_b});
    std.debug.print("Intersection(A, B): {s}\n", .{std.mem.span(wkt_inter)});
}
