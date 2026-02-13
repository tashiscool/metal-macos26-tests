import Metal

let path = CommandLine.arguments[1]
let d = MTLCreateSystemDefaultDevice()!
do {
    let l = try d.makeLibrary(URL: URL(fileURLWithPath: path))
    let p = try d.makeComputePipelineState(function: l.makeFunction(name: "attention")!)
    print("OK")
} catch {
    print("FAIL: \(error)")
}
