
let model
async function loadModel() {
    model = await tf.loadLayersModel('./nn/model.json');
    console.log('Model loaded');
}
loadModel()


const means = [8.365175097, 0.527600681, 0.265058366, 2.398881323, 0.081856031, 16.95598249, 49.2368677, 0.996747702, 3.310569066, 0.641308366, 10.41497244, 5.720817121]
const stds = [1.705389558, 0.173164146, 0.188267286, 0.858823511, 0.023728712, 10.0097098, 32.96114092, 0.001827468, 0.142321445, 0.137941735, 1.028825407, 0.853146182]

const tests = [[8.9, 0.56, 0.18, 1.9, 0.074, 12, 46, 0.9969, 3.17, 0.92, 9.5, 6],
[10.8, 0.5, 0.55, 2.2, 0.077, 15, 33, 0.9988, 3.18, 0.6, 9.4, 6],
[8, 0.84, 0.07, 2.5, 0.086, 11, 18, 0.99476, 3.37, 0.54, 10, 7],
[6.1, 0.69, 0.24, 2.3, 0.078, 5, 19, 0.9952, 3.54, 0.74, 9.7, 6],
[8.9, 0.45, 0.39, 1.5, 0.05, 3, 12, 0.99808, 3.32, 0.72, 11, 5],
[7.8, 0.41, 0.29, 2.1, 0.104, 6, 16, 0.9967, 3.26, 0.73, 11.8, 7],
[9.2, 0.54, 0.26, 2.3, 0.082, 23, 122, 0.998, 3.12, 0.64, 9.2, 5],
[7, 0.62, 0.06, 1.9, 0.076, 27, 63, 0.9975, 3.34, 0.74, 9.6, 6],
[7.1, 0.36, 0.3, 1.8, 0.081, 24, 52, 0.9978, 3.4, 0.52, 10.3, 6],
[8.7, 0.58, 0.21, 2.5, 0.097, 18, 91, 0.99596, 3.19, 0.68, 9.2, 5],
[12, 0.31, 0.49, 2.8, 0.091, 9, 30, 0.9987, 3.04, 0.81, 10.7, 6]]



function predict(parameters) {
    if (model == null) {
        console.log('Model not loaded');
        return {"error": "Model not loaded"};
    }
    
    if (typeof parameters == "object")
        parameters = Object.values(parameters);
    // parameters = [8, 0.5, 0.39, 2.2, 0.073, 30, 39, 0.99572, 3.33, 0.77, 12.1] // test data
    parameters = parameters.map((x, i) => (x - means[i]) / stds[i]);
    const x = tf.tensor2d(parameters, [1, 11]);
    const result = model.predict(x);
    let output = result.dataSync()
    // find max
    let max = 0;
    let maxIndex = 0;
    for (let i = 0; i < output.length; i++) {
        if (output[i] > max) {
            max = output[i];
            maxIndex = i;
        }
    }
    maxIndex += 3
    console.log(maxIndex)
    return maxIndex
}

// setTimeout(() => {
//     for (let i = 0; i < tests.length; i++) {
//         let testData = tests[i].splice(0, 11)
//         console.log(predict(testData), tests[i][0])
//     }
// }, 1000)




