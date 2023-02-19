
let model
async function loadModel() {
    model = await tf.loadLayersModel('./nn/model.json');
    console.log('Model loaded');
}
loadModel()


const means = [8.365175097, 0.527600681, 0.265058366, 2.398881323, 0.081856031, 16.95598249, 49.2368677, 0.996747702, 3.310569066, 0.641308366, 10.41497244, 5.720817121]
const stds = [1.705389558, 0.173164146, 0.188267286, 0.858823511, 0.023728712, 10.0097098, 32.96114092, 0.001827468, 0.142321445, 0.137941735, 1.028825407, 0.853146182]

function predict(parameters) {
    if (model == null) {
        console.log('Model not loaded');
        return {"error": "Model not loaded"};
    }
    
    parameters = Object.values(parameters);
    // parameters = [8, 0.5, 0.39, 2.2, 0.073, 30, 39, 0.99572, 3.33, 0.77, 12.1] // test data
    parameters = parameters.map((x, i) => (x - means[i]) / stds[i]);
    const x = tf.tensor2d(parameters, [1, 11]);
    const result = model.predict(x);
    let output = result.dataSync()[0];
    output = output * stds[11] + means[11];
    console.log(output)
    return output;
}
