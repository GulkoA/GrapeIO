
let model
async function loadModel() {
    model = await tf.loadLayersModel('https://foo.bar/tfjs_artifacts/model.json');
}
