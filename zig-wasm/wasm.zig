const std = @import("std");
const allocator = std.heap.wasm_allocator;

extern "env" fn jsGet(id: usize, ptr: [*]const u8, len: usize) usize;
extern "env" fn jsSet(id: usize, ptr: [*]const u8, len: usize, ptr2: [*]const u8, len2: usize) void;
extern "env" fn jsInvoke(id: usize, ptr1: [*]const u8, len1: usize, ptr2: [*]const u8, len2: usize) usize;
extern "env" fn jsInvokeValue(self: usize, prt1: [*]const u8, len1: usize, arg1: usize) usize;
extern "env" fn jsAddEventListener(self: usize, context: *anyopaque, func_pointer: *anyopaque) void;

export fn alloc(len: usize) ?[*]u8 {
    return if (allocator.alloc(u8, len)) |slice|
        slice.ptr
    else |_|
        // check how its returned to js, maybe 0 ?
        null;
}

const string = []const u8;

const Handle = struct {
    obj: *anyopaque,
    func: *const fn (*anyopaque) void,
};

const rowItem = struct {
    element: Value,
    selected: bool = false,
    fn new() !*@This() {
        const item = try allocator.create(@This());
        item.element = createElement("div");
        item.onSelected();
        item.element.addEventListener("click", .{ .obj = item, .func = toggleSelect });
        return item;
    }

    fn toggleSelect(ptr: *anyopaque) void {
        const obj: *@This() = @ptrCast(@alignCast(ptr));
        obj.selected = !obj.selected;
        obj.onSelected();
    }
    fn onSelected(self: @This()) void {
        if (self.selected) {
            self.element.set("innerText", "selected");
        } else {
            self.element.set("innerText", "not selected");
        }
    }
};

export fn free(ptr: [*]u8, len: usize) void {
    allocator.free(ptr[0..len]);
}

fn printf(comptime fmt: []const u8, args: anytype) void {
    const message = std.fmt.allocPrint(std.heap.wasm_allocator, fmt, args) catch "failed to allocate string";
    print(message);
}

export fn callZig(context: *anyopaque, function: *anyopaque) void {
    const handle: Handle = .{ .obj = context, .func = @ptrCast(function) };
    handle.func(handle.obj);
}

fn createElement(element_name: []const u8) Value {
    return global.get("document").call("createElement", element_name);
}

fn print(str: []const u8) void {
    //TODO how to free any of these?
    // or some lazy way of stating that value is not needed
    _ = global.get("console").call("log", str);
}

const global = Value{ .id = 0 };

const Value = struct {
    id: usize,

    //TODO have some get that doesn't alloc new Value on js side
    fn get(value: Value, str: []const u8) Value {
        const object = jsGet(value.id, str.ptr, str.len);
        //TODO need to free
        return Value{ .id = object };
    }

    fn addEventListener(value: Value, event_name: []const u8, handle: Handle) void {
        _ = event_name;
        jsAddEventListener(value.id, handle.obj, @ptrCast(@constCast(handle.func)));
    }
    fn set(value: Value, property: []const u8, str: []const u8) void {
        jsSet(value.id, property.ptr, property.len, str.ptr, str.len);
    }

    //TODO move out of value
    // also rename not to be confused with setInnerHtml
    fn setInnerHtml(value: Value, comptime content: string) returnType(content) {
        value.set("innerHTML", content);

        var return_value: returnType(content) = undefined;
        inline for (std.meta.fields(@TypeOf(return_value))) |field| {
            const selector: string = "#" ++ field.name;
            //TODO query in the fragment
            @field(return_value, field.name) = global.get("document").call("querySelector", selector);
        }
        return return_value;
    }

    fn call(value: Value, function_name: []const u8, arg1: anytype) Value {
        if (@TypeOf(arg1) == Value) {
            return Value{ .id = jsInvokeValue(value.id, function_name.ptr, function_name.len, arg1.id) };
        } else if (@TypeOf(arg1) == []const u8) {
            return Value{ .id = jsInvoke(value.id, function_name.ptr, function_name.len, arg1.ptr, arg1.len) };
        } else {
            @compileLog(@TypeOf(arg1));
            @compileError("call only supports Value or []const u8");
        }
    }
};

fn extractIds(html: string, ids: []string) i32 {
    var i: usize = 0;
    var counter = 0;
    while (i < html.len) {
        // look for `id=` pattern
        if (i + 3 <= html.len and html[i] == 'i' and html[i + 1] == 'd' and html[i + 2] == '=') {
            var quote: u8 = 0;
            var start: usize = 0;

            // find whether single or double quote follows
            if (i + 3 < html.len and (html[i + 3] == '"' or html[i + 3] == '\'')) {
                quote = html[i + 3];
                start = i + 4;
                // find closing quote
                var j = start;
                while (j < html.len and html[j] != quote) : (j += 1) {}
                if (j < html.len) {
                    const idValue = html[start..j];
                    ids[counter] = idValue;
                    counter += 1;
                    i = j + 1;
                    continue;
                }
            }
        }
        i += 1;
    }
    return counter;
}

fn returnType(component: string) type {
    //TODO remove limit, or make it big ?
    var ids: [5]string = undefined;
    // inline extractIds
    const number_of_ids = extractIds(component, &ids);
    var fields: [number_of_ids]std.builtin.Type.StructField = undefined;

    for (0..number_of_ids) |i| {
        const name = ids[i];
        fields[i] = std.builtin.Type.StructField{
            .name = name ++ "",
            .type = Value,
            .is_comptime = false,
            .default_value_ptr = null,
            .alignment = @alignOf(u32),
        };
    }

    return @Type(std.builtin.Type{
        .@"struct" = .{
            .layout = .auto,
            .fields = &fields,
            .decls = &.{},
            .is_tuple = false,
        },
    });
}

fn changeText(value_pointer: *anyopaque) void {
    const value: *Value = @ptrCast(@alignCast(value_pointer));
    value.set("innerText", "comptime !!!");
}

export fn main() void {
    const body = global.get("document").get("body");
    var ids = body.setInnerHtml(
        \\<div id="some">
        \\  <div id="person">hello</div>
        \\  <div id="foo">hello</div>
        \\</div>
    );
    ids.person.addEventListener("click", .{ .obj = &ids.foo, .func = changeText });
    for (0..1000) |_| {
        //TODO deinit
        const handler = rowItem.new() catch return;
        _ = body.call("appendChild", handler.element);
    }
}
