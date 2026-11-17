import Foundation
import CoreAI

struct Fixture: Decodable {
    let input_ids: [Int32]
    let attention_mask: [Int32]
    let option_positions: [Int32]
    let pool_mask: [Float]
    let expected: [Float]
    let labels: [String]
    let output: String
}

@main struct VerifyVela {
    static func main() async throws {
        guard CommandLine.arguments.count == 2 else {
            fatalError("Usage: verify-vela /path/to/model-directory")
        }
        let root = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
        let fixture = try JSONDecoder().decode(
            Fixture.self, from: Data(contentsOf: root.appendingPathComponent("swift_fixture.json"))
        )
        let model = try await AIModel(
            contentsOf: root.appendingPathComponent("model.aimodel"), options: .cpuOnly
        )
        guard let descriptor = model.functionDescriptor(for: "main"),
              descriptor.stateNames.isEmpty,
              let function = try model.loadFunction(named: "main") else {
            fatalError("Missing stateless main function")
        }
        var inputs: [String: NDArray] = [:]
        for (name, values) in [
            ("input_ids", fixture.input_ids),
            ("attention_mask", fixture.attention_mask),
            ("option_positions", fixture.option_positions),
        ] {
            guard case .ndArray = descriptor.inputDescriptor(of: name) else {
                fatalError("Missing input \(name)")
            }
            var array = NDArray(shape: [1, values.count], scalarType: .int32)
            var view = array.mutableView(as: Int32.self)
            view.copyElements(fromContentsOf: values)
            inputs[name] = array
        }
        guard case .ndArray = descriptor.inputDescriptor(of: "pool_mask") else {
            fatalError("Missing pool_mask")
        }
        var pool = NDArray(shape: [1, fixture.pool_mask.count], scalarType: .float32)
        var poolView = pool.mutableView(as: Float.self)
        poolView.copyElements(fromContentsOf: fixture.pool_mask)
        inputs["pool_mask"] = pool
        var result = try await function.run(inputs: inputs)
        guard let output = result.remove(fixture.output)?.ndArray else {
            fatalError("Missing logits")
        }
        let logits: [Float] = output.view(as: Float.self).withUnsafePointer { ptr, _, _ in
            Array(UnsafeBufferPointer(start: ptr, count: fixture.labels.count))
        }
        guard logits.allSatisfy({ $0.isFinite }) else { fatalError("Nonfinite logits") }
        let maximum = logits.max()!
        let exps = logits.map { exp($0 - maximum) }
        let total = exps.reduce(0, +)
        let probabilities = exps.map { $0 / total }
        let error = zip(probabilities, fixture.expected).map { abs($0 - $1) }.max()!
        guard error < 1e-5 else { fatalError("Swift/Python parity failed: \(error)") }
        let report: [String: Any] = [
            "swift_native_coreai": true,
            "device": "cpuOnly",
            "labels": fixture.labels,
            "probabilities": probabilities,
            "maximum_probability_error_vs_python": error,
        ]
        print(String(data: try JSONSerialization.data(
            withJSONObject: report, options: [.prettyPrinted, .sortedKeys]
        ), encoding: .utf8)!)
    }
}
