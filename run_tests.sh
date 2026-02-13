#!/bin/bash
# macOS 26 Metal GPU Compiler Compatibility Tests
set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "macOS 26 Metal GPU Compiler Compatibility Tests"
echo "================================================"
echo "System: $(sw_vers -productName) $(sw_vers -productVersion)"
echo "Metal: $(xcrun -sdk macosx metal --version 2>&1 | head -1)"
echo ""

# Compile all test cases
echo "Compiling..."
xcrun -sdk macosx metal -c 01_msl_baseline.metal -o /tmp/01_msl.air 2>&1
xcrun -sdk macosx metallib /tmp/01_msl.air -o /tmp/01_msl.metallib 2>&1
echo "  01_msl_baseline.metal -> OK"

xcrun -sdk macosx metal -x ir -c 02_ir_minimal_broken.ll -o /tmp/02_broken.air 2>&1
xcrun -sdk macosx metallib /tmp/02_broken.air -o /tmp/02_broken.metallib 2>&1
echo "  02_ir_minimal_broken.ll -> OK"

xcrun -sdk macosx metal -x ir -c 03_ir_minimal_fixed.ll -o /tmp/03_fixed.air 2>&1
xcrun -sdk macosx metallib /tmp/03_fixed.air -o /tmp/03_fixed.metallib 2>&1
echo "  03_ir_minimal_fixed.ll -> OK"

echo ""
echo "Testing pipeline creation..."

test_pipeline() {
    local name="$1"
    local metallib="$2"
    local expect="$3"

    echo -n "  $name ... "
    # Run in subshell to catch crashes
    local result
    result=$(swift -e "
import Metal
import Foundation
let d = MTLCreateSystemDefaultDevice()!
do {
    let data = try Data(contentsOf: URL(fileURLWithPath: \"$metallib\"))
    let dd = data.withUnsafeBytes { DispatchData(bytes: \$0) }
    let l = try d.makeLibrary(data: dd as __DispatchData)
    let fn = l.makeFunction(name: \"test_kernel\")!
    let _ = try d.makeComputePipelineState(function: fn)
    print(\"PASS\")
} catch {
    print(\"FAIL: \(error.localizedDescription)\")
}
" 2>&1) || result="CRASH"

    if [[ "$result" == *"PASS"* ]]; then
        if [ "$expect" = "pass" ]; then
            echo "PASS"
        else
            echo "UNEXPECTED PASS"
        fi
    else
        if [ "$expect" = "fail" ]; then
            echo "EXPECTED FAIL (GPU compiler crash without SDK Version)"
        else
            echo "FAIL: $result"
        fi
    fi
}

OS_MAJOR=$(sw_vers -productVersion | cut -d. -f1)
if [ "$OS_MAJOR" -ge 26 ]; then
    BROKEN_EXPECT="fail"
else
    BROKEN_EXPECT="pass"
fi

test_pipeline "MSL baseline" "/tmp/01_msl.metallib" "pass"
test_pipeline "IR without SDK Version" "/tmp/02_broken.metallib" "$BROKEN_EXPECT"
test_pipeline "IR with SDK Version" "/tmp/03_fixed.metallib" "pass"

echo ""
echo "Cleanup..."
rm -f /tmp/01_msl.{air,metallib} /tmp/02_broken.{air,metallib} /tmp/03_fixed.{air,metallib}
echo "Done."
