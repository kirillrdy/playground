let alloc, free, memory, callZig;

function getView(ptr, len) {
  return new Uint8Array(memory.buffer, ptr, len);
}

// JS strings are UTF-16 and have to be encoded into an
// UTF-8 typed byte array in WASM memory.
function writeStrToWasm(str) {
  const capacity = str.length * 2 + 5; // As per MDN
  const ptr = alloc(capacity);
  //TODO is this correct ?
  if (ptr == null) {
    console.error("failed to allocate memory")
  }
  const { written } = new TextEncoder().encodeInto(str, getView(ptr, capacity));
  return [ptr, written, capacity];
}

// Decode UTF-8 typed byte array in WASM memory into
// UTF-16 JS string.
function readStrFromWasm(ptr, len) {
  // TODO reuse decoder encoder
  return new TextDecoder().decode(getView(ptr, len));
}

let nextId = 1;
let objectsMap = [];

let putObject = (object) => {
  objectsMap[nextId] = object
  nextId++
  return nextId - 1
}

// The environment we export to WASM.
let importObject = {
  env: {
    jsAddEventListener: (id, context, functionPointer) => {
      objectsMap[id].addEventListener("click", function() {
        callZig(context, functionPointer);
      })
    },
    jsInvoke: (id, p1, l1, p2, l2) => {
      const funcName = readStrFromWasm(p1, l1)
      const arg1 = readStrFromWasm(p2, l2)
      return putObject(objectsMap[id][funcName](arg1))
    },
    jsInvokeValue: (id, p1, l1, id2) => {
      const funcName = readStrFromWasm(p1, l1)
      const arg1 = objectsMap[id2]
      return putObject(objectsMap[id][funcName](arg1))
    },

    jsSet: (id, p1, l1, p2, l2) => {
      const arg1 = readStrFromWasm(p1, l1)
      const arg2 = readStrFromWasm(p2, l2)
      objectsMap[id][arg1] = arg2
    },

    jsGet: (id, ptr, len) => {
      if (id == 0) {
        return putObject(window[readStrFromWasm(ptr, len)])
      }
      return putObject(objectsMap[id][readStrFromWasm(ptr, len)])
    }
  }
};
//TODO hardcoded path
WebAssembly.instantiateStreaming(fetch('/main.wasm'), importObject)
  .then((wasm_binary) => {
    ({ alloc, free, memory, callZig } = wasm_binary.instance.exports);

    const [ptr, len, capacity] = writeStrToWasm("Hello from Zig + JS + WASM 🦎⚡!");
    free(ptr, capacity);
    wasm_binary.instance.exports.main()
  });

